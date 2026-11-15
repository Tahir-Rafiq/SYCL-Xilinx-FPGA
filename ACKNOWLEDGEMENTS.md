# Acknowledgements and third-party material

This repository uses code, data and results from the projects and publications below. Files copied or adapted from other projects keep their original licenses; the license texts are in [LICENSES/](LICENSES/). Everything else is under the MIT license in [LICENSE](LICENSE).

## Compiler

triSYCL/sycl, the SYCL compiler for AMD FPGAs used for all examples, a fork of Intel's DPC++ compiler ([intel/llvm](https://github.com/intel/llvm)).

- Repository: https://github.com/triSYCL/sycl, branch `sycl/unified/master`, commit `35aa03e00bb26b685912c2c17c111c6027165520` (4 November 2024).
- License: Apache License v2.0 with LLVM Exceptions ([LICENSES/LLVM-Apache-2.0-WITH-LLVM-exception.txt](LICENSES/LLVM-Apache-2.0-WITH-LLVM-exception.txt)).
- The compiler is not part of this repository. [scripts/build_compiler.sh](scripts/build_compiler.sh) downloads that commit and builds it.
- [examples/answer_42/answer_42.cpp](examples/answer_42/answer_42.cpp) and [examples/vector_add/vector_add.cpp](examples/vector_add/vector_add.cpp) are unchanged copies of `sycl/test/vitis/simple_tests/answer_42.cpp` and `sycl/test/vitis/simple_tests/single_task_vector_add.cpp` from that commit, under the same license.

## Matrix multiply example

[examples/matmul/matmul.cpp](examples/matmul/matmul.cpp) is a simplified version of the `matmul` FPGA reference design of oneAPI-samples by Intel Corporation.

- Repository: https://github.com/oneapi-src/oneAPI-samples, tag `2025.0.0`, commit `d3e1ac1829708bedb62c138fb64a9fdf6791fe11`, folder `DirectProgramming/C++SYCL_FPGA/ReferenceDesigns/matmul`.
- License: MIT, Copyright Intel Corporation ([LICENSES/MIT-oneAPI-samples.txt](LICENSES/MIT-oneAPI-samples.txt), also in [examples/matmul/](examples/matmul/LICENSE.oneAPI-samples)).
- The changes are listed in [examples/matmul/README.md](examples/matmul/README.md).

## Graph convolutional network example

[examples/gcn/gcn.cpp](examples/gcn/gcn.cpp) was written for this repository. Its model, data, trained weights and reference results come from the following sources.

- The model is the two-layer graph convolutional network of T. N. Kipf and M. Welling, "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017, [arXiv:1609.02907](https://arxiv.org/abs/1609.02907). Reference implementation: https://github.com/tkipf/gcn (MIT license).
- The Cora citation dataset was introduced by A. K. McCallum, K. Nigam, J. Rennie and K. Seymore, "Automating the Construction of Internet Portals with Machine Learning", Information Retrieval 3, 127–163, 2000, [doi:10.1023/A:1009953814988](https://doi.org/10.1023/A:1009953814988). The version with 2708 papers, 1433 words and 7 classes used here is the one of P. Sen, G. Namata, M. Bilgic, L. Getoor, B. Gallagher and T. Eliassi-Rad, "Collective Classification in Network Data", 2008, vol. 29, no. 3, [doi:10.1609/aimag.v29i3.2157](https://doi.org/10.1609/aimag.v29i3.2157).
- The node order and the training and test split are those of Z. Yang, W. W. Cohen and R. Salakhutdinov, "Revisiting Semi-Supervised Learning with Graph Embeddings", ICML 2016, [arXiv:1603.08861](https://arxiv.org/abs/1603.08861), as distributed with https://github.com/tkipf/gcn.
- The weights `W0.csv` and `W1.csv` were trained with `gcn-seq` of parallel-gcn by Liwei Cai and Chengze Fan, https://github.com/cai-lw/parallel-gcn (MIT license, Copyright (c) 2019 Liwei Cai).
- The sparse matrices in CSR form, the word features and the labels were converted from the Cora files of parallel-gcn (`data/cora.graph` and `data/cora.svmlight`). The reference results `AH0W1.csv` were computed in double precision by `GCN.cpp`, a sequential C++ program based on `gcn-seq` (not included here); `gcn.cpp` follows its order of operations.

[examples/gcn/README.md](examples/gcn/README.md) describes each data file.

## AMD tools

The examples are built with AMD Vitis, Vitis HLS and Vivado 2023.2 and run with XRT 2.16.204 ([Xilinx/XRT](https://github.com/Xilinx/XRT), Apache License 2.0) on the Alveo U280 with the platform `xilinx_u280_gen3x16_xdma_1_202211_1`. None of these are included here. [scripts/flash_card.tcl](scripts/flash_card.tcl) follows the JTAG flash procedure of AMD/Xilinx answer record 71757.
