# edge_detection: the same filter on the CPU and on the FPGA

An exercise rather than a finished example. The program finds the edges in a picture with a 3x3 Sobel filter, once in plain C++ on the CPU and once in a SYCL kernel on the FPGA, checks that both give the same picture, and prints the time of each. The FPGA kernel, as given, is slow. The questions below lead from measuring the two to changing the kernel and checking what the change did.

The kernel is the `edge_detection` example of the triSYCL/sycl compiler repository (`sycl/test/vitis/edge_detection`), itself a SYCL translation of the Xilinx SDAccel `edge_detection` example; the picture `data/eiffel.bmp` (1024 x 1895 = 1,940,480 pixels) comes from the same folder. For this exercise the OpenCV calls were replaced by a small BMP reader and writer, and three things were added: an optional crop, the same filter in plain C++ for the CPU with its time, and the comparison of the two results. See [ACKNOWLEDGEMENTS.md](../../ACKNOWLEDGEMENTS.md).

## The CPU baseline

`sobel_cpu()` is the CPU baseline: a plain loop over the pixels, row by row, on one core, compiled with `-O2`. The compiler does not vectorise it, so it computes one pixel at a time. The program times it on the full picture and prints the best of 5 runs.

## Build and run

The other examples are built with `sycl-build`. Here the compile commands are typed by hand, because the CPU part must be compiled with optimization (`-O2`) for a fair comparison; `cat $(which sycl-build)` shows what the script runs. In the example folder, with the compiler set up (`source env.sh` in the repository folder, or the shared setup of your host):

```bash
mkdir -p build_hw_emu
SYCL_VXX_KEEP_CLUTTER=True TMPDIR=$PWD/build_hw_emu clang++ -std=c++20 -O2 -fsycl -fsycl-targets=fpga64_hls_hw_emu edge_detection.cpp -o edge_detection_hw_emu > compile_hw_emu.log 2>&1; echo "rc=$?"
```

The FPGA emulation build takes about 6 minutes (longer when several builds share the host) and ends with `rc=0`. `SYCL_VXX_KEEP_CLUTTER` and `TMPDIR` keep the build folder, with its reports and logs, in `build_hw_emu/`. The build does not need you: open a second terminal on the same host (or end the command with `&`) and work on the questions meanwhile.

The CPU part alone builds in a second, without SYCL:

```bash
clang++ -std=c++20 -O2 -DCPU_ONLY edge_detection.cpp -o edge_detection_cpu
./edge_detection_cpu data/eiffel.bmp
```

Run the emulation program with a crop size:

```bash
./edge_detection_hw_emu data/eiffel.bmp 32
```

The FPGA filters only the central 32 x 32 pixels, because hardware emulation simulates every clock cycle and is slow (for the kernel as given, about 40 s for 32 x 32 and 80 s for 64 x 64). The CPU filters the crop and the full picture. The run ends with `PASSED: 0 errors in 1024 values` when the FPGA and CPU results agree, and writes `output_fpga.bmp` (the crop) and `output_cpu.bmp` (the full picture). The FPGA time it prints in emulation is the time the simulator needed, not the time of the FPGA.

To change the kernel, copy only the source and the picture into a new folder next to this one (`mkdir ../edge_v1 && cp -r edge_detection.cpp data ../edge_v1/`) and work there, so that both versions keep their own reports, programs and results. Copying the whole folder after a build would also copy the old build folder. Each emulation run leaves a simulator folder `.run/` of several hundred MB; delete `.run/` and `build_hw_emu/` when you are done.

## Where to find the numbers

| What | Where |
|---|---|
| II of the kernel's loop, as scheduled by HLS | the line `Pipelining result` in `compile_hw_emu.log`; the loop table of `build_hw_emu/edge_detection-*/vxx_comp_tmp/*/*/*/solution/syn/report/*Pipeline_1_csynth.rpt` (the loop is called `Loop 1`) |
| HLS messages about that loop | `build_hw_emu/edge_detection-*/vxx_comp_tmp/*/*/vitis_hls.log` |
| Simulated time of a run | the line `$stop called at time` in `*_simulate.log` in the example folder, written after each emulation run. It is the simulated time of the whole run (ps or ns), including a fixed start-up part. |

The Vivado simulator writes that time in the largest unit that still shows the value exactly, so it comes in ps or in ns (for example `918100250 ps` or `2395410 ns`). This line converts the reading into microseconds:

```bash
grep -h 'stop called' *_simulate.log | awk '{f=($7=="ps")?1e-6:($7=="ns")?1e-3:1; printf "%.1f us\n", $6*f}'
```

For two crop sizes `a` and `b`, the simulated cycles per pixel at 300 MHz are

    (T_b - T_a) x 300 MHz / (P_b - P_a)

where `T` is the simulated time and `P` the number of pixels the kernel's loop actually processes (look at the loop bounds: does the loop include the border of the crop?). The difference removes the fixed start-up part. That part also varies by a few tens of microseconds from run to run, so the faster the kernel, the larger the crops must be: if the two times differ by less than a few hundred microseconds, use larger crops, for example 64 and 128, or 256 and 512.

## Questions

While the first build runs:

1. Which part of `edge_detection.cpp` runs on the FPGA and which on the CPU? How does the picture get to the FPGA and back?
2. Build and run the CPU-only program. How long does one core need for the full picture?
3. For two consecutive iterations of the kernel's inner loop, write down the addresses read for the centre pixel of the Sobel window. How far apart are they?

After the first build:

4. Run the crops 32 and 64 and check that both pass. What II does HLS report for the loop, and how many cycles per pixel does the simulated time give? The II is the interval at which the loop can start a new iteration once its pipeline is full; the simulated cycles per pixel describe the whole kernel. Do they agree, and if not, what could account for the difference? Estimate the time for the full picture at 300 MHz and compare it with the CPU.
5. Find the HLS warning about the loop's II in `vitis_hls.log` and quote it. Which access in the source does it refer to?
6. Propose a change to the order in which the kernel walks through the picture, and predict its effect on the memory accesses of question 3, on the II and on the time for the full picture. Write the prediction down, then make the change in the new folder `../edge_v1` and start the second build there.

While the second build runs:

7. At 300 MHz, would an II of 1 be enough to beat the CPU time of question 2? How many pixels per clock cycle, on average, would the FPGA need to match it?

After the second build:

8. Compare the new II and the simulated cycles per pixel with your prediction, and check that the output still passes against the CPU result. What did the change do to the memory accesses, and how far is the FPGA now from one CPU core?

[UG1399, the Vitis HLS user guide](https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls), explains pipelining, the II and the HLS messages. [Xilinx/Vitis_Accel_Examples](https://github.com/Xilinx/Vitis_Accel_Examples) shows how image filters are usually written for AMD FPGAs.

## Home task (optional)

The CPU baseline uses one core and one pixel at a time. Measure how far the CPU can go:

1. Is the loop of `sobel_cpu()` vectorised? Build the CPU-only program with `-Rpass=loop-vectorize -Rpass-missed=loop-vectorize` to see what the compiler reports. Find a way to let the compiler vectorise it and measure again.
2. Use all cores (`nproc` prints the number of hardware threads), for example by giving each of several `std::thread` threads its own band of rows, and measure again.
3. Compare the plain loop, the vectorised loop, all cores and your best FPGA estimate. How many pixels per clock cycle would the FPGA need to match each CPU version, and what would such a kernel have to do differently?
