# answer_42

The smallest SYCL program for the FPGA: one kernel writes the number 42 into a buffer, and the host reads it back.

```cpp
sycl::buffer<int> answer { 1 };
sycl::queue q;
q.submit([&](sycl::handler &cgh) {
  sycl::accessor a { answer, cgh, sycl::write_only };
  cgh.single_task<class forty_two>([=] { a[0] = 42; });
});
sycl::host_accessor ans { answer, sycl::read_only };
std::cout << "The universal answer to the question is " << ans[0] << std::endl;
```

The program makes a buffer of one integer and a queue, which is the connection to a device. It prints the device name and vendor and submits one kernel that writes 42 into the buffer. The `host_accessor` waits for the kernel and copies the value back.

What it tests: the runtime finds the U280, loads the FPGA image, allocates a buffer in card memory, starts a kernel and reads the result back. The kernel itself does almost nothing, so a failure points at the tools or the card setup and not at the kernel code.

`answer_42.cpp` is an unchanged copy of `sycl/test/vitis/simple_tests/answer_42.cpp` from the compiler (triSYCL/sycl, commit 35aa03e0; Apache License v2.0 with LLVM Exceptions, see [ACKNOWLEDGEMENTS.md](../../ACKNOWLEDGEMENTS.md)). The `// REQUIRES` and `// RUN` lines at the top are for the compiler's test harness; the compiler ignores them.

## Build and run

From the repository folder, after `source env.sh`:

```bash
cd examples/answer_42
sycl-build answer_42.cpp hw_emu          # about 6 minutes
./answer_42_hw_emu                       # about 30 seconds
sycl-build answer_42.cpp hw              # about 1 hour
./answer_42_hw                           # on the card
```

Expected output on the card (in emulation the device is named `xilinx_u280_gen3x16_xdma_1_202211_1`):

```
Queue Device: xilinx_u280_gen3x16_xdma_base_1
Queue Device Vendor: Xilinx
The universal answer to the question is 42
```

The program has no automatic check; the printed value is the test. [card_run_log.md](card_run_log.md) has the full log of a run on the card with an explanation of every line.

## Results

The hardware build took 62 minutes on a 32-core host and met timing at the 300 MHz kernel clock (Vivado estimated 306.6 MHz). On the card the first run took 7.2 s, almost all of it loading the FPGA image, and a second run 0.21 s.
