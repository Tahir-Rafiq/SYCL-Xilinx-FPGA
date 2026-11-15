# 2. The compiler

The examples are compiled with [triSYCL/sycl](https://github.com/triSYCL/sycl), a fork of Intel's DPC++ compiler that can target AMD FPGAs through Vitis. The project is no longer maintained; the tested commit is `35aa03e00bb26b685912c2c17c111c6027165520` (4 November 2024), the last one of the branch `sycl/unified/master`.

If a compiler is already installed on your host, skip this page and point `SYCL_HOME` to it ([03_environment.md](03_environment.md)).

## Install the small tools

cmake generates the build files for LLVM. The compiler needs cmake 3.20 or newer, and Ubuntu 20.04 and Vitis ship older versions. ninja is the build tool that the compiler's build scripts use. posix_ipc is a Python module that the compiler driver (`sycl_vxx.py`) imports to limit how many `v++` jobs run at once. The script installs all three into your own account (`~/.local`), without root access:

```bash
scripts/install_tools.sh
```

It prints the versions at the end: cmake 3.27.9, ninja 1.13.2 or newer, posix_ipc 1.3.2 or newer. cmake is pinned to 3.27.9, the version that was tested; cmake 4.x dropped support for old `cmake_minimum_required` settings and was not tried.

## Build

```bash
scripts/build_compiler.sh 24 > build_compiler.log 2>&1     # 24 parallel jobs
tail -3 build_compiler.log
```

The script downloads the tested commit without its history (about 2 GB) into `~/sycl_workspace/llvm` and builds it in `~/sycl_workspace/llvm/build`. Set `SYCL_WORKSPACE` to use another folder. With 24 jobs on a 32-core host the build takes about 15 minutes. The last line of the log is `compiler built in .../llvm/build`.

The build runs in a clean environment (`env -i`). If your `.bashrc` sets up XRT or Vitis, a normal shell has the XRT libraries in `LD_LIBRARY_PATH` and the old cmake 3.3.2 of Vitis in `PATH`, and both break the build. Inside the clean environment the script runs the compiler's own configure and compile scripts:

```bash
python3 llvm/buildbot/configure.py --xrt --cmake-opt=-DLLVM_PARALLEL_LINK_JOBS=4
python3 llvm/buildbot/compile.py -j 24
```

`--xrt` builds the XRT plugin (`libpi_xrt.so`); it finds the XRT headers through `XILINX_XRT=/opt/xilinx/xrt`. At most 4 parallel link jobs keep the memory use within 64 GB.

Check the result:

```bash
ls ~/sycl_workspace/llvm/build/bin/{clang++,sycl-ls,sycl_vxx.py}
ls ~/sycl_workspace/llvm/build/lib/{libsycl.so,libpi_xrt.so}
~/sycl_workspace/llvm/build/bin/clang++ --version    # clang version 16.0.0 (https://github.com/triSYCL/sycl.git 35aa03e0...)
```

## Optional: the compiler's own tests

The CPU test suite compiles the Vitis test programs in syntax-only mode, so it needs no card and no FPGA build. It takes about one minute:

```bash
cd ~/sycl_workspace/llvm/build
~/.local/bin/cmake --build . --parallel 24 --target check-sycl-vitis-cpu
```

Expected: 31 passed, 7 unsupported, 1 failed. The failure is `vitis/simple_tests/evaluation_tree.cpp`, which uses `std::weak_ordering` from the C++20 header `<compare>`; the gcc 9 C++ library of Ubuntu 20.04 does not have that header. None of the examples here use it.
