# SYCL-Xilinx-FPGA

Four SYCL examples that go from a single C++ source file to kernels running on an AMD Alveo U280 FPGA card, built with the [triSYCL/sycl](https://github.com/triSYCL/sycl) compiler and AMD Vitis 2023.2. Each example builds for hardware emulation (minutes, no card needed) and for the card (about an hour), and each comes with the log of a run on a U280 and an explanation of every line of it.

The repository was prepared for a seminar on SYCL for FPGAs at the University of Siena.

| Example | What it computes | What it adds | Pass condition |
|---|---|---|---|
| [answer_42](examples/answer_42/) | one kernel writes 42 into a buffer | the whole path from source to card, nothing else | prints 42 |
| [vector_add](examples/vector_add/) | sum of two vectors of 300 integers | data copies, loops as hardware pipelines, on-chip memory | all 300 values checked |
| [matmul](examples/matmul/) | 16x16 integer matrix product on a 4x4 systolic array | four kernels running together, connected by pipes | all 256 values checked |
| [gcn](examples/gcn/) | inference of a trained two-layer graph convolutional network on the Cora citation graph (2708 papers) | real data, float arithmetic, sparse matrices, banked on-chip memory, three dependent kernels | all 18956 outputs checked; 81.1% test accuracy |

## Requirements

- Ubuntu 20.04 (tested: 20.04.6)
- AMD Vitis, Vitis HLS and Vivado 2023.2, XRT 2.16.204, and the U280 platform `xilinx_u280_gen3x16_xdma_1_202211_1`
- gcc 9.4, Python 3.8, git
- for running on the card: an Alveo U280 with the shell `xilinx_u280_gen3x16_xdma_base_1`

No root access is needed to build the compiler or the examples. Details and checks: [docs/01_requirements.md](docs/01_requirements.md).

## Quick start

Clone the repository and enter its folder (`cd SYCL-Xilinx-FPGA`). Then, once:

```bash
scripts/install_tools.sh                                   # cmake, ninja, posix_ipc into ~/.local
scripts/build_compiler.sh 24 > build_compiler.log 2>&1     # about 15 minutes with 24 jobs
```

If your host already has this compiler installed, skip these two steps and set `export SYCL_HOME=<compiler folder>` instead.

In every new shell:

```bash
source env.sh
```

Build an example for hardware emulation and run it:

```bash
cd examples/vector_add
sycl-build vector_add.cpp hw_emu          # about 6 minutes
./vector_add_hw_emu; echo "exit $?"       # exit 0: all 300 values correct
```

Build it for the card and run it there:

```bash
sycl-build vector_add.cpp hw              # about 1 hour
./vector_add_hw; echo "exit $?"
```

The other examples work the same way; their READMEs give the commands and the expected output.

## Documentation

| Page | Content |
|---|---|
| [docs/01_requirements.md](docs/01_requirements.md) | tool versions and how to check them |
| [docs/02_compiler.md](docs/02_compiler.md) | building the compiler |
| [docs/03_environment.md](docs/03_environment.md) | what `env.sh` sets and why |
| [docs/04_emulation.md](docs/04_emulation.md) | hardware emulation builds and runs, HLS reports |
| [docs/05_hardware_build.md](docs/05_hardware_build.md) | hardware builds, timing check, build folder |
| [docs/06_run_on_card.md](docs/06_run_on_card.md) | checking the card and running the programs |
| [docs/07_card_shell.md](docs/07_card_shell.md) | putting the Vitis shell on the card (administrators) |
| [docs/compile_flow.md](docs/compile_flow.md) | what the compiler does, step by step |
| [docs/compiler_notes.md](docs/compiler_notes.md) | limits to know when writing your own kernels |

## Repository layout

| Path | Content |
|---|---|
| `env.sh` | environment for building and running |
| `bin/sycl-build` | builds one program for `hw_emu` or `hw` in the current folder |
| `bin/unshare` | wrapper the compiler needs on Ubuntu 20.04 ([docs/03_environment.md](docs/03_environment.md)) |
| `scripts/install_tools.sh` | installs cmake, ninja and posix_ipc into `~/.local` |
| `scripts/build_compiler.sh` | downloads and builds the compiler |
| `scripts/flash_card.tcl` | writes a shell image into the card's flash over JTAG ([docs/07_card_shell.md](docs/07_card_shell.md)) |
| `examples/<name>/` | source, README with expected output and results, `card_run_log.md` |
| `examples/gcn/data/` | Cora graph, trained weights and reference results (CSV, 1.9 MB) |
| `LICENSES/` | license texts of the third-party files |

## License and acknowledgements

The repository is under the MIT license ([LICENSE](LICENSE)). The two first examples come from the triSYCL/sycl compiler (Apache License v2.0 with LLVM Exceptions), the matrix multiply is simplified from Intel's oneAPI-samples (MIT), and the GCN example uses the model of Kipf and Welling, the Cora dataset in the version distributed with parallel-gcn, and weights trained with parallel-gcn. [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) lists every source with its license and citation.

AMD, Alveo, Vitis, Vivado and Xilinx are trademarks of Advanced Micro Devices, Inc. This repository is not affiliated with AMD.
