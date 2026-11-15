# 3. Environment

Source `env.sh` from the repository folder in every shell that builds or runs the examples:

```bash
cd SYCL-Xilinx-FPGA
source env.sh
```

`env.sh` sets:

- `SYCL_HOME`, the compiler folder (the one with `bin/clang++` and `lib/libsycl.so`). The default is `~/sycl_workspace/llvm/build`, where `scripts/build_compiler.sh` puts the compiler. To use a compiler installed elsewhere, set the variable before sourcing: `export SYCL_HOME=/path/to/compiler`.
- `PATH`: the repository's `bin/` (with `sycl-build`), the compiler, XRT, and Vitis, Vitis HLS and Vivado 2023.2.
- `LD_LIBRARY_PATH`: the SYCL runtime, XRT and the Vitis libraries.
- `XILINX_PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1`, the platform the compiler builds for.
- `ONEAPI_DEVICE_SELECTOR="xrt:*"` and `SYCL_PI_TRACE=1` (see below).
- an emulation configuration in `~/.Xilinx/emconfig.json`, created the first time.

It prints a line for anything it does not find: the compiler, Vitis, XRT or the posix_ipc module.

XRT's own `/opt/xilinx/xrt/setup.sh` is not used. It checks its own install path and sets nothing when the path differs from what it expects, for example when `/opt/xilinx` is a symbolic link. `env.sh` sets the same variables directly.

## Why the settings

`ONEAPI_DEVICE_SELECTOR="xrt:*"`: an AMD FPGA host usually has an OpenCL driver file for the card (`/etc/OpenCL/vendors/xilinx.icd`), so the same card is visible through the OpenCL and the XRT backend of the SYCL runtime. With both active, `sycl-ls` crashes with a core dump in hardware emulation (XRT issue 7226). Selecting one backend avoids this without root access.

`SYCL_PI_TRACE=1` makes the runtime print which plugin it loaded and which device it picked, at the start of every program. The log explanations in `examples/*/card_run_log.md` describe these lines.

`bin/unshare` is needed on Ubuntu 20.04. The compiler driver runs every `v++` call through `unshare --map-current-user` in its own process namespace, and that option needs util-linux 2.38 or newer. Ubuntu 20.04 has 2.34, where the driver stops with `TypeError: 'NoneType' object is not subscriptable` in `VXXVersion`. The wrapper replaces the option with `--map-root-user`, which 2.34 has. Files written by `v++` still belong to your user. On a newer system the wrapper does no harm.

## Check

```bash
sycl-ls
XCL_EMULATION_MODE=hw_emu sycl-ls
```

The first command lists the card if the host has a U280 with the Vitis shell (`[xrt:acc:0] Xilinx XRT, xilinx_u280_gen3x16_xdma_base_1 ...`), and nothing otherwise. The second lists the emulated card, `[xrt:acc:0] Xilinx XRT, xilinx_u280_gen3x16_xdma_1_202211_1 0.0 [2.16.204]`. Both print `Plugin found and successfully loaded: libpi_xrt.so` first.
