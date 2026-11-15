# 5. Hardware build

Only the target changes from hardware emulation:

```bash
sycl-build <source.cpp> hw
```

The build runs HLS, then links the kernels into the U280 platform with Vivado synthesis, placement and routing. It takes about an hour or more, so run it in the background:

```bash
cd examples/vector_add
nohup sycl-build vector_add.cpp hw > /dev/null 2>&1 &
tail -1 compile_hw.log          # rc=0 minutes=64 when it has finished successfully
```

Build times on a 32-core host with 64 GB of memory:

| Example | Build | Program | Kept build folder |
|---|---|---|---|
| answer_42 | 62 min | 49 MB | 0.7 GB |
| vector_add | 64 min | 50 MB | 0.7 GB |
| matmul | 90 min | 49 MB | 0.9 GB |
| gcn | 99 min | 52 MB | 1.3 GB |

On that host two hardware builds ran at the same time without running short of memory, and so did the gcn hardware build together with a full-graph gcn emulation. The result is one program per example, `<name>_hw`, with the FPGA image (xclbin) inside. To run, it needs only the SYCL runtime libraries (`$SYCL_HOME/lib`) and XRT, so it can be copied to another host with the same operating system, XRT and card shell.

## Check the result

The kernel clock is 300 MHz. Timing is met when `compile_hw.log` has no errors and contains the advisory `AUTO-FREQ-SCALING-08` for `clk_kernel_00_unbuffered_net`, which says that the kernel could run faster than 300 MHz; the kernel still runs at 300 MHz. Vivado's estimates of the possible kernel clock were 306.6 MHz (answer_42), 373.6 MHz (vector_add), 304.9 MHz (matmul) and 304.0 MHz (gcn).

In `build_hw/<name>-<hash>/`:

| Path | Content |
|---|---|
| `vxx_comp_report/<kernel>/hls_reports/<kernel>_csynth.rpt` | HLS latency and resource estimates |
| `vxx_link_report/link/imp/impl_1_hw_bb_locked_timing_summary_routed.rpt` | timing after place and route |
| `vxx_link_tmp/link/vivado/vpl/prj/prj.xpr` | Vivado project, open with `vivado` to see the placed design |
| `<name>-<hash>.xclbin` | the FPGA image |
| `000-...cmd` to `006-vxxlink.cmd` | the commands the compiler ran |
