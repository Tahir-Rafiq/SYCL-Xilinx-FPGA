# Matrix multiply with a systolic array

`matmul.cpp` multiplies two 16x16 integer matrices on the U280 with a 4x4 systolic array, using the same compiler and flow as `answer_42` and `vector_add`.

## Origin

It is a simplified version of the `matmul` FPGA reference design from Intel's oneAPI-samples:

- Repository: https://github.com/oneapi-src/oneAPI-samples
- Path: `DirectProgramming/C++SYCL_FPGA/ReferenceDesigns/matmul` (systolic array in `DirectProgramming/C++SYCL_FPGA/include/streaming_matmul.hpp`)
- Version: tag `2025.0.0`, commit `d3e1ac1829708bedb62c138fb64a9fdf6791fe11` (17 December 2024), https://github.com/oneapi-src/oneAPI-samples/tree/2025.0.0/DirectProgramming/C%2B%2BSYCL_FPGA/ReferenceDesigns/matmul
- License: MIT, Copyright Intel Corporation (`LICENSE.oneAPI-samples` in this folder)

The FPGA samples were removed from the main branch when FPGA support left the oneAPI toolkits in 2025.1, and the main branch now points to https://github.com/altera-fpga/hls-samples. The tag above is the last one in oneAPI-samples that still contains the code.

## What was kept

The structure: four kernels running at the same time and connected by pipes (two feeders that read A and B from card memory, the systolic array, and a drain that writes C), the tiling of the output into tiles the size of the array, and one multiply-add per processing element per step. The fully unrolled loops use the same compile-time index trick as the original `fpga_tools::UnrolledLoop`.

## What was changed

| Original | Here | Why |
|---|---|---|
| float, 64x64 matrices, 8x8 tiles, several matrices and repetitions | int, 16x16, 4x4 tiles, one multiplication | short builds, exact check; a float accumulator would not reach one step per cycle in Vitis HLS |
| USM (`malloc_device`, `malloc_shared`, `memcpy`) | buffers and accessors | USM is not implemented in the XRT plugin of our compiler |
| `annotated_arg` with Intel memory properties, `buffer_location` | plain accessors | not in our compiler |
| `ac_int` loop counters | `int` | `ac_int` header not shipped |
| `while(1)` array kernel with a "done" pipe | loop with a known count, no done pipe | a Vitis kernel has to finish |
| shift-out of results overlapped with the next tile | tile written out after its last step | simpler; costs 4 cycles per tile |
| Intel attributes, `fpga_reg`, `fpga_selector_v`, event profiling | removed; default queue | Intel-only |
| A and B in column-major / transposed layouts, several header files | row-major, one file | easier to read on a slide |

## What it shows

Four kernels run on the card at the same time and pass data through pipes:

1. FeederA reads A from card memory and writes one column of a 4-row tile per step into PipeA.
2. FeederB reads B and writes one row of a 4-column tile per step into PipeB.
3. Systolic has 16 processing elements. Each step, element (r, c) multiplies `a[r]` by `b[c]` and adds the product to its own accumulator. After 16 steps a 4x4 tile of C is finished and goes into PipeC, one column per write.
4. Drain reads PipeC and writes C into card memory.

The host fills A and B with random values from 0 to 9 (`srand(42)`, so every run uses the same matrices), computes the same product on the CPU and compares all 256 values.

Compared with `answer_42` and `vector_add` it adds several kernels running at the same time, kernel-to-kernel streams (the compiler turns each pipe into an AXI stream connection between two kernels at link time), back-pressure through FIFOs of depth 16, and a fully unrolled compute array. FeederA sends 256 columns, far more than its FIFO holds, so the program can only finish if the kernels really run together.

Two compiler rules came up while porting. Each pipe can be read or written from only one place in a kernel: writing PipeC from an unrolled loop made four write sites and stopped the compiler with "multiple reader or writers". And the original array kernel is an endless `while(1)` loop, which a Vitis kernel cannot be, so it became a loop over the 16 tiles.

## Build and run

From the repository folder, after `source env.sh`:

```bash
cd examples/matmul
sycl-build matmul.cpp hw_emu          # about 6 minutes
./matmul_hw_emu; echo "exit $?"       # about 30 seconds
sycl-build matmul.cpp hw              # about 90 minutes
./matmul_hw; echo "exit $?"           # on the card
```

Expected output, in emulation and on the card:

```
C[0][0..3] = 256 183 210 264
PASSED: 0 errors in 256 values
```

and exit status 0. With wrong results the program prints up to 10 lines `C[i][j] = ..., expected ...`, then `FAILED: <n> errors in 256 values`, and exits with 1. [card_run_log.md](card_run_log.md) has the full log of a run on the card with an explanation of every line.

## Results

From the HLS reports (`build_hw_emu/matmul-<hash>/vxx_comp_tmp/.../solution/syn/report/`):

| Kernel | Pipelined loop (trip count) | II | Kernel latency | BRAM_18K | DSP | FF | LUT |
|---|---|---|---|---|---|---|---|
| FeederA | 16, run 4 times | 4 | 2227 cycles | 2 | 0 | 1930 | 2423 |
| FeederB | 16, run 4 times | 4 | 2227 cycles | 2 | 0 | 1303 | 1728 |
| Systolic | 16 (one per tile) | 16 | 262 cycles | 0 | 48 | 7697 | 7079 |
| Drain | 4, run 4 times | 4 | 1443 cycles | 2 | 0 | 2057 | 2771 |

The array kernel takes one pipe value per cycle, the most the pipes deliver, so a tile of 16 steps takes 16 cycles. The 48 DSPs are three per 32-bit multiplier for the 16 processing elements. The feeders are the slow part: each step reads 4 values through one memory port, so their loops start a new step every 4 cycles. Reading wider words from memory, as the original design does, would fix that; it was left out to keep the code short.

The hardware build met timing at the 300 MHz kernel clock with little margin (worst negative slack +0.016 ns; Vivado estimated 304.9 MHz for the kernel clock). On the card the first run took 7.9 s, almost all of it loading the FPGA image, and a second run 0.31 s.
