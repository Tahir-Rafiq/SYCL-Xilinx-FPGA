# 1. Requirements

Everything in this repository was built and run with the versions below. Other versions of the AMD tools were not tried; the compiler expects Vitis 2023.2 in particular. Installing the AMD tools, XRT and the U280 platform needs root access and is outside this guide.

| Item | Version used | How to check |
|---|---|---|
| Operating system | Ubuntu 20.04.6, kernel 5.15 | `lsb_release -d; uname -r` |
| Vitis, Vitis HLS, Vivado | 2023.2 in `/opt/Xilinx` | `ls /opt/Xilinx/Vitis /opt/Xilinx/Vitis_HLS /opt/Xilinx/Vivado` |
| XRT | 2.16.204 (2023.2 branch) in `/opt/xilinx/xrt` | `cat /opt/xilinx/xrt/version.json` |
| U280 development platform | `xilinx_u280_gen3x16_xdma_1_202211_1` | `ls /opt/xilinx/platforms` |
| U280 platform packages | `xilinx-u280-gen3x16-xdma-base`, `xilinx-u280-gen3x16-xdma-1-202211-1-dev`, `xilinx-u280-gen3x16-xdma-validate` | `dpkg -l \| grep xilinx-u280` |
| Card shell (only to run on the card) | `xilinx_u280_gen3x16_xdma_base_1` | [06_run_on_card.md](06_run_on_card.md) |
| gcc | 9.4.0 (system) | `gcc --version` |
| Python | 3.8.10 with pip | `python3 -m pip --version` |
| git | 2.25.1 | `git --version` |
| cmake, ninja, posix_ipc | 3.27.9, 1.13.2, 1.3.2, installed by `scripts/install_tools.sh` | [02_compiler.md](02_compiler.md) |

The host used had 32 cores and 64 GB of memory.

The compiler driver calls the old Clang 3.9 tools that ship inside Vitis HLS. Check that they are there:

```bash
ls /opt/Xilinx/Vitis_HLS/2023.2/lnx64/tools/clang-3.9-csynth/bin/{opt,llvm-as,llvm-link,llvm-dis}
ls /opt/Xilinx/Vitis_HLS/2023.2/lnx64/lib/libspir64-39-hls.bc
```

Disk space:

| What | Size |
|---|---|
| compiler sources and build (`~/sycl_workspace`) | about 4.5 GB |
| one hardware emulation build, with its kept build folder | 1.1 to 1.5 GB |
| one hardware build, with its kept build folder | 0.8 to 1.4 GB, of which about 50 MB is the program |

No card and no root access are needed to build the compiler, to build the examples or to run them in hardware emulation. A U280 with the Vitis shell is needed only to run the hardware programs.
