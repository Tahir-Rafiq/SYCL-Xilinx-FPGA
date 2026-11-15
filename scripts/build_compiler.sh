#!/bin/bash
# Download and build the SYCL compiler for AMD FPGAs (triSYCL/sycl) with its XRT plugin.
#
# Usage: scripts/build_compiler.sh [jobs]          (default: 24 parallel jobs)
#   SYCL_WORKSPACE  where the sources and the build go (default ~/sycl_workspace)
#
# Result: $SYCL_WORKSPACE/llvm/build, the default SYCL_HOME of env.sh.
# Needs: scripts/install_tools.sh, XRT in /opt/xilinx/xrt, about 4.5 GB of disk.
# Takes about 15 minutes with 24 jobs on a 32-core host with 64 GB of memory.
#
# The build runs in a clean environment: XRT libraries in LD_LIBRARY_PATH or the
# old cmake of Vitis in PATH (both common when .bashrc sets up the AMD tools)
# break it.
set -e
JOBS=${1:-24}
SYCL_WORKSPACE=${SYCL_WORKSPACE:-$HOME/sycl_workspace}
REPO=https://github.com/triSYCL/sycl.git
COMMIT=35aa03e00bb26b685912c2c17c111c6027165520   # branch sycl/unified/master, 4 Nov 2024

mkdir -p "$SYCL_WORKSPACE"
cd "$SYCL_WORKSPACE"
if [ ! -d llvm/.git ]; then
  # Only the tested commit, without history (about 2 GB)
  git init -q llvm
  git -C llvm remote add origin "$REPO"
  git -C llvm fetch --depth 1 origin "$COMMIT"
  git -C llvm checkout -q FETCH_HEAD
fi
have=$(git -C llvm rev-parse HEAD)
if [ "$have" != "$COMMIT" ]; then
  echo "llvm/ is at $have, expected $COMMIT"
  exit 1
fi

exec env -i HOME="$HOME" USER="$USER" TERM="${TERM:-dumb}" \
  PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin" \
  XILINX_XRT=/opt/xilinx/xrt JOBS="$JOBS" SYCL_WORKSPACE="$SYCL_WORKSPACE" \
  bash --noprofile --norc -c '
set -e
cd "$SYCL_WORKSPACE"
echo "cmake: $(cmake --version | head -1)  ninja: $(ninja --version)  gcc: $(gcc -dumpversion)"
# --xrt builds the XRT plugin (libpi_xrt.so); at most 4 link jobs keep memory use in bounds
python3 llvm/buildbot/configure.py --xrt --cmake-opt=-DLLVM_PARALLEL_LINK_JOBS=4
python3 llvm/buildbot/compile.py -j "$JOBS"
echo "compiler built in $SYCL_WORKSPACE/llvm/build"
'
