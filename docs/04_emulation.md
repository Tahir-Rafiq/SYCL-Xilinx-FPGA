# 4. Hardware emulation

Hardware emulation runs the kernels in the Vivado RTL simulator (xsim), against a model of the U280 platform. It uses the same source and the same HLS step as the hardware build, so it shows whether the kernels compute the right result, and the HLS reports are already there. It needs no card and no root access, and a build takes minutes instead of an hour.

`bin/sycl-build` builds one program in the current folder:

```bash
sycl-build <source.cpp> hw_emu
```

It runs `clang++ -std=c++20 -fsycl -fsycl-targets=fpga64_hls_hw_emu <source.cpp>` and writes:

| Output | Content |
|---|---|
| `<name>_hw_emu` | the program, with the emulation image inside |
| `compile_hw_emu.log` | compiler output; the last line is `rc=<exit status> minutes=<build time>` |
| `build_hw_emu/<name>-<hash>/` | the build folder, kept on purpose (`SYCL_VXX_KEEP_CLUTTER=True`) |

The program is written as `<name>_hw_emu.new` and renamed only when the build succeeds, so a failed build never leaves an old program behind under the expected name.

## The four examples

From the repository folder, after `source env.sh`:

```bash
cd examples/answer_42  && sycl-build answer_42.cpp hw_emu  && ./answer_42_hw_emu
cd ../vector_add       && sycl-build vector_add.cpp hw_emu && ./vector_add_hw_emu; echo "exit $?"
cd ../matmul           && sycl-build matmul.cpp hw_emu     && ./matmul_hw_emu; echo "exit $?"
cd ../gcn              && sycl-build gcn.cpp hw_emu        && ./gcn_hw_emu data 128; echo "exit $?"
```

| Example | Build | Run | Pass condition |
|---|---|---|---|
| answer_42 | 6 min | 30 s | prints `The universal answer to the question is 42` |
| vector_add | 6 min | 30 s | exit status 0 |
| matmul | 6 min | 30 s | `PASSED: 0 errors in 256 values`, exit status 0 |
| gcn, 128 nodes | 7 min | 4 min | `PASSED: 0 errors in 896 values`, exit status 0 |
| gcn, full graph (`./gcn_hw_emu data`) | same build | 72 min | `PASSED: 0 errors in 18956 values`, exit status 0 |

The builds are independent and can run at the same time in different folders. The programs find out from their embedded image that they are emulation programs and set `XCL_EMULATION_MODE=hw_emu` themselves. In emulation the device is named after the platform, `xilinx_u280_gen3x16_xdma_1_202211_1`. Each example's README has the expected output.

## What to look at

| Path in `build_hw_emu/<name>-<hash>/` | Content |
|---|---|
| `vxx_comp_report/<kernel>/hls_reports/<kernel>_csynth.rpt` | HLS result per kernel: loops, initiation interval (II), latency, resources |
| `vxx_comp_tmp/.../solution/syn/report/` | detailed HLS reports, one per loop |
| `000-...cmd` to `006-vxxlink.cmd` | the commands the compiler ran, in order ([compile_flow.md](compile_flow.md)) |

The kernel names in the reports are generated from the names in the source with a hash added, for example `handlerEE_clES2_E3add_yN5WAcwV` for the kernel `add` of vector_add.

The run writes simulator files to `.run/<pid>/` in the current folder, plus `emulation_debug.log`, `xcd.log` and two simulator logs whose names end in `_simulate.log` and `_xsc_report.log`. `.gitignore` covers all of them. For gcn on the full graph, `.run/<pid>/hw_emu/device0/binary_0/behav_waveform/xsim/output.txt` grows to about 280 MB of harmless DSP48 warnings. Delete `.run/` after looking at the results.

The emulation run needs the variable `USER`, which a normal login sets. Without it the simulator stops with `ERROR: [HW-EMU 26] $USER variable is not SET`.
