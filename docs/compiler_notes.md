# Notes for writing your own kernels

Limits of this compiler (triSYCL/sycl, commit 35aa03e0) with Vitis 2023.2, found while writing the examples. They matter when you change an example or write a new one.

- Kernels are `single_task`. The XRT plugin starts one work-item per kernel; loops inside the kernel become hardware pipelines.
- Use buffers and accessors. Unified shared memory (`malloc_device`, `malloc_shared`) is not implemented in the XRT plugin.
- The XRT plugin starts a kernel without waiting for the kernels it depends on (`xrt_piEnqueueKernelLaunch` ignores its wait list). If a kernel reads a buffer that an earlier kernel writes, call `.wait()` on the earlier `q.submit(...)` before submitting the next one, as `examples/gcn/gcn.cpp` does.
- Kernels connected by pipes (`sycl::ext::intel::pipe`) must run at the same time, so do not wait between their submissions (see `examples/matmul/matmul.cpp`). Each pipe can be read or written from only one place in a kernel; a write inside an unrolled loop counts as several places and stops the compiler with "multiple reader or writers".
- A kernel must end. An endless `while(1)` kernel, common in Intel FPGA code, does not work with Vitis.
- On-chip arrays split into banks: a two-dimensional `sycl::ext::xilinx::partition_ndarray` makes Vitis HLS crash. Use a one-dimensional `partition_array<T, R * C, partition::cyclic<C, 1>>` and index it as `r * C + c`; element (r, c) then sits in memory c.
- A float accumulation `acc += a * b` in a pipelined loop gives an initiation interval of 7 at 300 MHz, the latency of a float add.
- To index a small array with a loop variable without turning it into a memory, unroll the loop with compile-time indices (the `unrolled<N>` helper in matmul and gcn).
- The default queue is out-of-order, and the XRT plugin prints `warning: queue created with unhandled properties` for it. The warning is harmless.
- The C++ library of gcc 9 (Ubuntu 20.04) has no `<compare>` header, so C++20 three-way comparison does not compile.
