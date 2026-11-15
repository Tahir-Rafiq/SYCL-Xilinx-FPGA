# answer_42 on the U280

The smallest SYCL program for the FPGA: one kernel writes the number 42 into a buffer, and the host reads it back. It checks the whole path from a single C++ source to a kernel running on the card.

## The code

`answer_42.cpp`, an unchanged copy of `sycl/test/vitis/simple_tests/answer_42.cpp` from the compiler (triSYCL/sycl, commit 35aa03e0). The `// REQUIRES` and `// RUN` lines at the top are for the compiler's test harness; the compiler ignores them.

```cpp
// REQUIRES: vitis

// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
// RUN: %clangxx %EXTRA_COMPILE_FLAGS-fsycl -std=c++20 -fsycl-targets=%sycl_triple %s -o %t.dir/exec.out
// RUN: %ACC_RUN_PLACEHOLDER %t.dir/exec.out

#include <sycl/sycl.hpp>
#include <iostream>


int main() {
  // Allocate 1 int of 1D abstract memory
  sycl::buffer<int> answer { 1 };
  // Create a queue on Xilinx FPGA
  sycl::queue q;

  std::cout << "Queue Device: "
            << q.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout << "Queue Device Vendor: "
            << q.get_device().get_info<sycl::info::device::vendor>()
            << std::endl;

  // Submit a kernel on the FPGA
  q.submit([&] (sycl::handler &cgh) {
      // Get a write-only access to the buffer
      sycl::accessor a { answer, cgh, sycl::write_only };
      // The computation on the accelerator
      cgh.single_task<class forty_two>([=] { a[0] = 42; });
    });

  // Verify the result
  sycl::host_accessor ans { answer, sycl::read_only };
  std::cout << "The universal answer to the question is " << ans[0]
            << std::endl;
}
```

What each part does:

- `sycl::buffer<int> answer { 1 }` declares one integer of memory that both the host and the device can use. The runtime decides where it lives and when to copy it.
- `sycl::queue q` picks a device with the default selector (here the U280) and creates a queue to send work to it.
- `q.submit(...)` sends one command group to the device. The accessor `a` says that the kernel only writes the buffer, so nothing has to be copied to the card before the kernel starts.
- `cgh.single_task<class forty_two>(...)` is the kernel: a single work item that runs `a[0] = 42`. The name `forty_two` is the kernel name in the FPGA image. The lambda body is the only part compiled for the FPGA; everything else runs on the host.
- `sycl::host_accessor ans` waits until the kernel has finished and copies the buffer back to the host before the value is printed.

## Build and run

Built for the U280 on a 32-core host with 64 GB of memory (62 minutes; one 49 MB program with the FPGA image inside). Run on 11 November 2026, on a host whose U280 runs the shell `xilinx_u280_gen3x16_xdma_base_1`. From the repository folder:

```bash
source env.sh
cd examples/answer_42
sycl-build answer_42.cpp hw
time ./answer_42_hw; echo "exit $?"
time ./answer_42_hw; echo "exit $?"
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
The universal answer to the question is 42

real    0m7.219s
user    0m0.015s
sys     0m0.124s
exit 0
```

## Log, second run (right after the first)

The output lines are the same; only the times differ:

```
real    0m0.208s
user    0m0.021s
sys     0m0.084s
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
| `Queue Device: ...`, `Queue Device Vendor: Xilinx` | Printed by the program itself, from the device information of the queue. |
| `The universal answer to the question is 42` | The result. The kernel ran on the FPGA and wrote 42 into the buffer in card memory; the `host_accessor` copied it back. |
| `exit 0` | The program ended normally. |

The three times come from the shell's `time` command:

- `real` is the wall-clock time from start to end, including any waiting.
- `user` is the CPU time spent in the program's own code and its libraries (SYCL runtime, XRT).
- `sys` is the CPU time the operating system spent on the program's behalf: calls into the card driver, memory mapping, reading the program file.

In the first run `real` (7.2 s) is far larger than `user + sys` (0.14 s): the program was waiting most of the time. The wait is the loading of the program's FPGA image onto the card, done by the driver and the card at the first kernel submission. In the second run the card already held this image, XRT skipped the load, and the whole run took 0.21 s: start the runtime, open the card, run the kernel, copy one value back. The kernel itself takes microseconds. Running another program in between loads that program's image, and the next run of answer_42 then loads its own again.

## Result

Passed: the program printed 42 and exited with status 0. This program has no automatic check; the printed value is the test. First run 7.2 s (with the image load), second run 0.21 s.
