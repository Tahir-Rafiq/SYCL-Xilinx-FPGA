# matmul on the U280

A 16x16 integer matrix product C = A * B computed by a 4x4 systolic array. Four kernels run on the card at the same time and pass data to each other through pipes; the host compares all 256 results with a product computed on the CPU. Compared with the first two examples it adds several kernels running together, kernel-to-kernel streams, and a fully unrolled compute array.

## The code

`matmul.cpp`, a simplified version of the `matmul` FPGA reference design from Intel's oneAPI-samples (tag 2025.0.0, MIT license, see `README.md` and `LICENSE.oneAPI-samples` in this folder).

```cpp
// SPDX-License-Identifier: MIT
// Matrix multiply C = A * B on the FPGA with a small systolic array.
//
// Simplified from the "matmul" FPGA reference design of oneAPI-samples
// (tag 2025.0.0, DirectProgramming/C++SYCL_FPGA/ReferenceDesigns/matmul,
// Copyright Intel Corporation, MIT License, see LICENSE.oneAPI-samples).
// Changes: int data, 16x16 matrices, 4x4 tiles, one multiplication, buffers
// and accessors instead of USM, a finite loop in the array kernel, no Intel
// FPGA attributes or ac_int. See README.md.
//
// Four kernels run at the same time and pass data through pipes:
//   FeederA  : A from card memory -> PipeA, one column of a 4-row tile per step
//   FeederB  : B from card memory -> PipeB, one row of a 4-column tile per step
//   Systolic : 4x4 processing elements, each multiply-adds into its own
//              accumulator; a finished tile goes to PipeC column by column
//   Drain    : PipeC -> C in card memory

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

constexpr int kRows = 16;   // rows of A and C
constexpr int kCommon = 16; // columns of A, rows of B
constexpr int kCols = 16;   // columns of B and C
constexpr int kTileA = 4;   // rows of the array (tile height)
constexpr int kTileB = 4;   // columns of the array (tile width)
constexpr int kTilesI = kRows / kTileA;
constexpr int kTilesJ = kCols / kTileB;
constexpr int kPipeDepth = 16;
static_assert(kRows % kTileA == 0 && kCols % kTileB == 0);

// Calls f(0), f(1), ..., f(N-1) with compile-time indices. The loop is fully
// unrolled in the source, so each index becomes its own piece of hardware.
template <typename F, std::size_t... I>
void unrolled_impl(F &&f, std::index_sequence<I...>) {
  (f(std::integral_constant<std::size_t, I>{}), ...);
}
template <std::size_t N, typename F> void unrolled(F &&f) {
  unrolled_impl(f, std::make_index_sequence<N>{});
}

struct ColA { int v[kTileA]; }; // one column of a tile of A (or of C)
struct RowB { int v[kTileB]; }; // one row of a tile of B

using PipeA = sycl::ext::intel::pipe<class PipeAName, ColA, kPipeDepth>;
using PipeB = sycl::ext::intel::pipe<class PipeBName, RowB, kPipeDepth>;
using PipeC = sycl::ext::intel::pipe<class PipeCName, ColA, kPipeDepth>;

int main() {
  // Row-major matrices with small random values, so int never overflows.
  std::vector<int> a(kRows * kCommon), b(kCommon * kCols), c(kRows * kCols, 0);
  std::srand(42);
  for (auto &x : a) x = std::rand() % 10;
  for (auto &x : b) x = std::rand() % 10;

  sycl::queue q;
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>()
            << std::endl;
  {
    sycl::buffer<int> buf_a{a.data(), sycl::range{a.size()}};
    sycl::buffer<int> buf_b{b.data(), sycl::range{b.size()}};
    sycl::buffer<int> buf_c{c.data(), sycl::range{c.size()}};

    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_a{buf_a, cgh, sycl::read_only};
      cgh.single_task<class FeederA>([=] {
        for (int ti = 0; ti < kTilesI; ti++)
          for (int tj = 0; tj < kTilesJ; tj++)
            for (int k = 0; k < kCommon; k++) {
              ColA col;
              unrolled<kTileA>([&](auto r) {
                col.v[r] = acc_a[(ti * kTileA + r) * kCommon + k];
              });
              PipeA::write(col);
            }
      });
    });

    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_b{buf_b, cgh, sycl::read_only};
      cgh.single_task<class FeederB>([=] {
        for (int ti = 0; ti < kTilesI; ti++)
          for (int tj = 0; tj < kTilesJ; tj++)
            for (int k = 0; k < kCommon; k++) {
              RowB row;
              unrolled<kTileB>([&](auto col) {
                row.v[col] = acc_b[k * kCols + tj * kTileB + col];
              });
              PipeB::write(row);
            }
      });
    });

    q.submit([&](sycl::handler &cgh) {
      cgh.single_task<class Systolic>([=] {
        for (int t = 0; t < kTilesI * kTilesJ; t++) {
          int acc[kTileA][kTileB];
          unrolled<kTileA>([&](auto r) {
            unrolled<kTileB>([&](auto col) { acc[r][col] = 0; });
          });
          // Each step every processing element (r, col) does one multiply-add.
          for (int k = 0; k < kCommon; k++) {
            ColA in_a = PipeA::read();
            RowB in_b = PipeB::read();
            unrolled<kTileA>([&](auto r) {
              unrolled<kTileB>([&](auto col) {
                acc[r][col] += in_a.v[r] * in_b.v[col];
              });
            });
          }
          // Send the finished tile out, one column per pipe write. The compiler
          // allows one write site per pipe, so this is a normal loop; the
          // column is picked with constant indices to keep acc in registers.
          for (int col = 0; col < kTileB; col++) {
            ColA out;
            unrolled<kTileA>([&](auto r) {
              int v = 0;
              unrolled<kTileB>([&](auto c) {
                if (c == col) v = acc[r][c];
              });
              out.v[r] = v;
            });
            PipeC::write(out);
          }
        }
      });
    });

    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_c{buf_c, cgh, sycl::write_only};
      cgh.single_task<class Drain>([=] {
        for (int ti = 0; ti < kTilesI; ti++)
          for (int tj = 0; tj < kTilesJ; tj++)
            for (int col = 0; col < kTileB; col++) {
              ColA out = PipeC::read();
              unrolled<kTileA>([&](auto r) {
                acc_c[(ti * kTileA + r) * kCols + tj * kTileB + col] = out.v[r];
              });
            }
      });
    });
  } // buffer destructors wait for the kernels and copy C back into c

  int errors = 0;
  for (int i = 0; i < kRows; i++)
    for (int j = 0; j < kCols; j++) {
      int ref = 0;
      for (int k = 0; k < kCommon; k++)
        ref += a[i * kCommon + k] * b[k * kCols + j];
      if (c[i * kCols + j] != ref && errors++ < 10)
        std::cout << "C[" << i << "][" << j << "] = " << c[i * kCols + j]
                  << ", expected " << ref << std::endl;
    }
  std::cout << "C[0][0..3] = " << c[0] << " " << c[1] << " " << c[2] << " "
            << c[3] << std::endl;
  std::cout << (errors ? "FAILED: " : "PASSED: ") << errors << " errors in "
            << kRows * kCols << " values" << std::endl;
  return errors ? 1 : 0;
}
```

What each part does:

- The constants set the sizes: A, B and C are 16x16, and the array works on tiles of 4x4, so C has 4 x 4 = 16 tiles. `kPipeDepth = 16` is the number of entries each pipe can hold.
- `unrolled<N>(f)` calls `f(0)`, ..., `f(N-1)` with compile-time indices. Each call becomes its own piece of hardware, so an unrolled loop of 4 is 4 copies of the logic working in parallel.
- `ColA` is one column of a 4-row tile of A (4 integers), `RowB` one row of a 4-column tile of B. They are the data types sent through the pipes.
- `PipeA`, `PipeB`, `PipeC` are pipes (FIFOs) between kernels. The compiler turns each into a direct AXI stream connection between two kernels on the FPGA; the data does not go through card memory.
- `main` fills A and B with random values 0 to 9 (`srand(42)`, so every run uses the same matrices and gives the same C).
- `sycl::queue q` picks the device with the default selector; the program prints the device name.
- The three buffers wrap the host vectors. A and B are read only on the card, C is write only.
- FeederA reads A from card memory and, for every tile of C and every step `k`, writes one column of 4 values of A into PipeA (16 tiles x 16 steps = 256 writes).
- FeederB does the same for B: one row of 4 values of B per step into PipeB.
- Systolic is the array: 16 processing elements, element (r, col) keeps its own accumulator `acc[r][col]`. Each step it reads one column of A and one row of B and all 16 elements do one multiply-add at the same time. After 16 steps the 4x4 tile of C is finished and is sent to PipeC, one column per write. The odd-looking selection with `if (c == col)` keeps `acc` in registers: indexing it with a loop variable would turn it into a memory.
- Drain reads the finished columns from PipeC and writes them into C in card memory.
- The four `q.submit` calls start the four kernels without waiting. They have to run at the same time: each pipe holds only 16 entries while FeederA alone sends 256, so a feeder would stop when its pipe is full and wait for the array to take data.
- The closing `}` of the buffer block destroys the buffers, which waits for the kernels and copies C back into the host vector `c`.
- The host then computes every element of C again on the CPU and counts the differences. It prints at most 10 wrong elements, the first four values of row 0, and `PASSED` or `FAILED`; the exit status is 0 only with no errors.

## Build and run

Built for the U280 on a 32-core host with 64 GB of memory (90 minutes; one 49 MB program with the FPGA image inside). Run on 11 November 2026, right after vector_add, on a host whose U280 runs the shell `xilinx_u280_gen3x16_xdma_base_1`. From the repository folder:

```bash
source env.sh
cd examples/matmul
sycl-build matmul.cpp hw
time ./matmul_hw; echo "exit $?"
time ./matmul_hw; echo "exit $?"
```

## Log, first run

```
SYCL_PI_TRACE[basic]: Plugin found and successfully loaded: libpi_xrt.so [ PluginVersion: 12.23 ]
SYCL_PI_TRACE[all]: Requested device_type: info::device_type::automatic
SYCL_PI_TRACE[all]: Selected device: -> final score = 1125
SYCL_PI_TRACE[all]:   platform: Xilinx XRT
SYCL_PI_TRACE[all]:   device: xilinx_u280_gen3x16_xdma_base_1
warning: queue created with unhandled properties
Device: xilinx_u280_gen3x16_xdma_base_1
C[0][0..3] = 256 183 210 264
PASSED: 0 errors in 256 values

real    0m7.942s
user    0m0.041s
sys     0m0.196s
exit 0
```

## Log, second run (right after the first)

The output lines are the same; only the times differ:

```
real    0m0.307s
user    0m0.046s
sys     0m0.118s
exit 0
```

## What the log shows

| Log line | Meaning |
|---|---|
| `Plugin found and successfully loaded: libpi_xrt.so [ PluginVersion: 12.23 ]` | The SYCL runtime (`libsycl.so`) loaded its XRT backend plugin, which talks to the card through AMD's XRT library. 12.23 is the version of the interface between runtime and plugin. The `SYCL_PI_TRACE` lines appear because `env.sh` sets `SYCL_PI_TRACE=1`. |
| `Requested device_type: info::device_type::automatic` | The program creates its queue with `sycl::queue q;` and does not ask for a device type, so the default selector scores every visible device. |
| `Selected device: -> final score = 1125` | The default selector's score: 75 because the device is an accelerator, plus 1000 because the program contains an image this device can run, plus 50 because the XRT backend is preferred to the OpenCL backend for AMD FPGAs. `ONEAPI_DEVICE_SELECTOR=xrt:*` (set by `env.sh`) leaves the card as the only candidate. |
| `platform: Xilinx XRT`, `device: xilinx_u280_gen3x16_xdma_base_1` | The chosen backend and device. On the card the device is named after the shell running on it. In hardware emulation the same device is named after the platform, `xilinx_u280_gen3x16_xdma_1_202211_1`. |
| `warning: queue created with unhandled properties` | Harmless. A SYCL queue is out-of-order unless the program asks for an in-order queue, and the runtime passes that flag to the plugin. The XRT plugin only handles the profiling flag, warns about any other one and ignores it. |
| `Device: xilinx_u280_gen3x16_xdma_base_1` | Printed by the program itself, from the device information of the queue. |
| `C[0][0..3] = 256 183 210 264` | The first four elements of row 0 of C as computed on the FPGA. The inputs come from `srand(42)`, so they are the same in every run, and these are the same values as in hardware emulation. |
| `PASSED: 0 errors in 256 values` | The host computed all 256 elements of C on the CPU and none differs from the FPGA result. With errors, up to 10 lines `C[i][j] = ..., expected ...` would come before a `FAILED` line. The program also finished at all, which shows that the four kernels ran at the same time: with pipes of 16 entries and 256 transfers per feeder, they cannot finish one after the other. |
| `real`, `user`, `sys` | From the shell's `time` command. `real` is the wall-clock time from start to end, including any waiting; `user` is the CPU time in the program's own code and its libraries (SYCL runtime, XRT); `sys` is the CPU time the operating system spent on the program's behalf (calls into the card driver, memory mapping, reading the program file). |
| `exit 0` | The program returns 0 only when all 256 values are correct. This is the test result. |

The times: in the first run `real` (7.9 s) is far larger than `user + sys` (0.24 s), so the program was waiting most of the time. The card still held the vector_add image, so the matmul image had to be loaded onto the card at the first kernel submission, which is the wait. In the second run the image was already there and the whole run took 0.31 s. That is about 0.1 s more than answer_42 and vector_add (0.21 and 0.20 s). This program starts four kernels instead of one and moves three buffers, which is the most likely reason; the extra time was not measured in detail. The kernels themselves need a few thousand clock cycles, about 7 µs at 300 MHz (the feeders are the slowest, 2227 cycles each in the HLS report, see README.md).

## Result

Passed: all 256 elements of C computed by the four kernels on the FPGA equal the CPU product, exit status 0 in both runs. First run 7.9 s (with the image load), second run 0.31 s.
