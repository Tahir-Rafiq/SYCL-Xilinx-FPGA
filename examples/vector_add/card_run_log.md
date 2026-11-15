# single_task_vector_add on the U280

One kernel adds two vectors of 300 integers on the card, and the host checks every result. Compared with answer_42 it adds data copied from the host to the card, a kernel with two inputs and one output, a local array in on-chip memory, two loops that become hardware pipelines, and a check of every output value.

## The code

`vector_add.cpp`, an unchanged copy of `sycl/test/vitis/simple_tests/single_task_vector_add.cpp` from the compiler (triSYCL/sycl, commit 35aa03e0). The `// REQUIRES` and `// RUN` lines at the top are for the compiler's test harness; the compiler ignores them.

```cpp
// REQUIRES: vitis

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx %EXTRA_COMPILE_FLAGS-std=c++20 -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out
// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out
// Test that we can get a fast reports by adding --vitis-ip-part
// RUN: %clangxx %EXTRA_COMPILE_FLAGS-std=c++20 -fsycl -fsycl-targets=%sycl_triple %s -o %t.dir/exec2.out --vitis-ip-part=xcu200-fsgd2104-2-e

/*
   A simple typical FPGA-like kernel adding 2 vectors
*/
#include <sycl/sycl.hpp>
#include <iostream>
#include <numeric>


using namespace sycl;

constexpr size_t N = 300;
using Type = int;

int main(int argc, char *argv[]) {
  buffer<Type> a { N };
  buffer<Type> b { N };
  buffer<Type> c { N };

  {
    sycl::host_accessor a_b{b, sycl::write_only};
    // Initialize buffer with increasing numbers starting at 0
    std::iota(&a_b[0], &a_b[a_b.size()], 0);
  }

  {
    sycl::host_accessor a_c{c, sycl::write_only};
    // Initialize buffer with increasing numbers starting at 5
    std::iota(&a_c[0], &a_c[a_c.size()], 5);
  }

  queue q;

  std::cout << "Queue Device: " << q.get_device().get_info<info::device::name>() << std::endl;
  std::cout << "Queue Device Vendor: " << q.get_device().get_info<info::device::vendor>() << std::endl;

  // Launch a kernel to do the summation
  q.submit([&] (handler &cgh) {
      // Get access to the data
      sycl::accessor a_a{a, cgh, sycl::write_only};
      sycl::accessor a_b{b, cgh, sycl::read_only};
      sycl::accessor a_c{c, cgh, sycl::read_only};

      // A typical FPGA-style pipelined kernel
      cgh.single_task<class add>([=] () {
          // Use an intermediate automatic array
          int array[N];
          for (unsigned int i = 0 ; i < N; ++i)
            array[i] = a_b[i];
          for (unsigned int i = 0 ; i < N; ++i)
            a_a[i] = array[i] + a_c[i];
        });
    });

  // Verify the result
  sycl::host_accessor a_a{a, sycl::read_only};
  for (unsigned int i = 0 ; i < a.size(); ++i) {
    assert(a_a[i] == 5 + 2*i && "invalid result from kernel");
  }

  return 0;
}
```

What each part does:

- `buffer<Type> a { N }`, `b`, `c` declare three buffers of 300 integers that the host and the device can both use.
- The two blocks with `host_accessor ... write_only` fill the inputs on the host: `b` with 0, 1, 2, ..., 299 and `c` with 5, 6, 7, ..., 304. Each accessor lives only inside its `{ }` block, so the buffer is free for the kernel afterwards.
- `queue q` picks the device with the default selector, as in answer_42.
- In `q.submit(...)` the accessors tell the runtime how the kernel uses each buffer: `b` and `c` are read only, so they are copied to the card before the kernel starts; `a` is write only, so nothing is copied for it.
- `cgh.single_task<class add>(...)` is the kernel, one work item running both loops. The first loop copies `b` into the local array `array`, which the compiler places in on-chip block RAM. The second loop computes `a[i] = array[i] + c[i]`. The compiler turns each loop into a pipeline that starts a new iteration every clock cycle (initiation interval 1), so each loop takes about 300 cycles; the whole kernel takes 819 cycles, 2.73 µs at 300 MHz (HLS report, see README.md).
- `host_accessor a_a{a, read_only}` waits for the kernel and copies `a` back to the host. The loop then checks every element: `b[i] + c[i] = i + (i + 5) = 5 + 2i`. A wrong value would stop the program in the `assert`.
- `return 0` is reached only when all 300 values are correct. The build does not define `NDEBUG`, so the asserts are active.

## Build and run

Built for the U280 on a 32-core host with 64 GB of memory (64 minutes; one 50 MB program with the FPGA image inside). Run on 11 November 2026, right after answer_42, on a host whose U280 runs the shell `xilinx_u280_gen3x16_xdma_base_1`. From the repository folder:

```bash
source env.sh
cd examples/vector_add
sycl-build vector_add.cpp hw
time ./vector_add_hw; echo "exit $?"
time ./vector_add_hw; echo "exit $?"
```

## Log, first run

```
SYCL_PI_TRACE[basic]: Plugin found and successfully loaded: libpi_xrt.so [ PluginVersion: 12.23 ]
SYCL_PI_TRACE[all]: Requested device_type: info::device_type::automatic
SYCL_PI_TRACE[all]: Selected device: -> final score = 1125
SYCL_PI_TRACE[all]:   platform: Xilinx XRT
SYCL_PI_TRACE[all]:   device: xilinx_u280_gen3x16_xdma_base_1
warning: queue created with unhandled properties
Queue Device: xilinx_u280_gen3x16_xdma_base_1
Queue Device Vendor: Xilinx

real    0m7.314s
user    0m0.027s
sys     0m0.113s
exit 0
```

## Log, second run (right after the first)

The output lines are the same; only the times differ:

```
real    0m0.200s
user    0m0.020s
sys     0m0.086s
exit 0
```

## What the log shows

| Log line | Meaning |
|---|---|
| `Plugin found and successfully loaded: libpi_xrt.so [ PluginVersion: 12.23 ]` | The SYCL runtime (`libsycl.so`) loaded its XRT backend plugin, which talks to the card through AMD's XRT library. 12.23 is the version of the interface between runtime and plugin. The `SYCL_PI_TRACE` lines appear because `env.sh` sets `SYCL_PI_TRACE=1`. |
| `Requested device_type: info::device_type::automatic` | The program creates its queue with `queue q;` and does not ask for a device type, so the default selector scores every visible device. |
| `Selected device: -> final score = 1125` | The default selector's score: 75 because the device is an accelerator, plus 1000 because the program contains an image this device can run, plus 50 because the XRT backend is preferred to the OpenCL backend for AMD FPGAs. `ONEAPI_DEVICE_SELECTOR=xrt:*` (set by `env.sh`) leaves the card as the only candidate. |
| `platform: Xilinx XRT`, `device: xilinx_u280_gen3x16_xdma_base_1` | The chosen backend and device. On the card the device is named after the shell running on it. In hardware emulation the same device is named after the platform, `xilinx_u280_gen3x16_xdma_1_202211_1`. |
| `warning: queue created with unhandled properties` | Harmless. A SYCL queue is out-of-order unless the program asks for an in-order queue, and the runtime passes that flag to the plugin. The XRT plugin only handles the profiling flag, warns about any other one and ignores it. |
| `Queue Device: ...`, `Queue Device Vendor: Xilinx` | Printed by the program itself, from the device information of the queue. |
| (no further output) | The program prints nothing after the device lines. The 300 results are checked by `assert`, which prints only on a failure (`Assertion ... "invalid result from kernel" failed`, then `Aborted` and exit status 134). No output here means every value was correct. |
| `real`, `user`, `sys` | From the shell's `time` command. `real` is the wall-clock time from start to end, including any waiting; `user` is the CPU time in the program's own code and its libraries (SYCL runtime, XRT); `sys` is the CPU time the operating system spent on the program's behalf (calls into the card driver, memory mapping, reading the program file). |
| `exit 0` | The program reached `return 0`, so all 300 checks passed. This is the test result. |

The times: in the first run `real` (7.3 s) is far larger than `user + sys` (0.14 s), so the program was waiting most of the time. The card still held the answer_42 image, so this program's FPGA image had to be loaded onto the card at the first kernel submission, which is the wait. In the second run the image was already there, XRT skipped the load, and the whole run took 0.20 s: start the runtime, open the card, copy the two 1.2 KB inputs to the card, run the kernel (about 3 µs), copy the result back and check it. The first-run times of answer_42 (7.2 s) and vector_add (7.3 s) are nearly equal although the kernels differ, because the load time depends on the size of the image, and both images are about the same size.

## Result

Passed: exit status 0 in both runs, so all 300 sums computed on the FPGA were correct. First run 7.3 s (with the image load), second run 0.20 s.
