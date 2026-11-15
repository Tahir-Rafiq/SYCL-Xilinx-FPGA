# Environment for building and running the SYCL examples on the Alveo U280.
#
# Usage (from the repository folder, in every new shell):
#   source env.sh
#
# SYCL_HOME is the compiler folder, the one with bin/clang++ and lib/libsycl.so.
# Default: the build made by scripts/build_compiler.sh, ~/sycl_workspace/llvm/build.
# To use a compiler installed elsewhere, set it first:
#   export SYCL_HOME=/path/to/compiler && source env.sh

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export SYCL_HOME=${SYCL_HOME:-$HOME/sycl_workspace/llvm/build}

# AMD tools: Vitis, Vitis HLS and Vivado 2023.2, XRT 2.16, U280 platform
export XILINX_VERSION=2023.2
export XILINX_ROOT=${XILINX_ROOT:-/opt/Xilinx}
export XILINX_VITIS=$XILINX_ROOT/Vitis/$XILINX_VERSION
export XILINX_HLS=$XILINX_ROOT/Vitis_HLS/$XILINX_VERSION
export XILINX_VIVADO=$XILINX_ROOT/Vivado/$XILINX_VERSION
export XILINX_XRT=${XILINX_XRT:-/opt/xilinx/xrt}
export XILINX_PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
export PLATFORM_REPO_PATHS=${PLATFORM_REPO_PATHS:-/opt/xilinx/platforms}

# XRT's own setup.sh is not sourced: it refuses some valid install paths (for
# example when /opt/xilinx is a symbolic link) and then sets nothing.
export PATH=$REPO_DIR/bin:$SYCL_HOME/bin:$PATH:$XILINX_XRT/bin:$XILINX_VITIS/bin:$XILINX_HLS/bin:$XILINX_VIVADO/bin
export LD_LIBRARY_PATH=$SYCL_HOME/lib:$XILINX_XRT/lib:$XILINX_VITIS/lib/lnx64.o:$XILINX_HLS/lib/lnx64.o${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$XILINX_XRT/python${PYTHONPATH:+:$PYTHONPATH}
# A shared compiler install may carry the posix_ipc module in $SYCL_HOME/python
if [ -d "$SYCL_HOME/python" ]; then export PYTHONPATH=$SYCL_HOME/python:$PYTHONPATH; fi
# The host linker needs the system library folders
export LIBRARY_PATH=$(ldconfig --verbose 2>/dev/null | grep ':$' | tr -d '\n')

# Use only the XRT backend. The OpenCL backend would list the same card a
# second time, and with both active sycl-ls crashes in hardware emulation.
export ONEAPI_DEVICE_SELECTOR="xrt:*"
# Print which device the runtime picks
export SYCL_PI_TRACE=1

# One emulation configuration per user, for hardware emulation runs
export EMCONFIG_PATH=$HOME/.Xilinx
mkdir -p "$EMCONFIG_PATH"
if [ ! -f "$EMCONFIG_PATH/emconfig.json" ] && command -v emconfigutil > /dev/null; then
  emconfigutil --platform $XILINX_PLATFORM --od "$EMCONFIG_PATH" > /dev/null
fi

# Report what is missing
[ -x "$SYCL_HOME/bin/clang++" ] || echo "env.sh: no compiler in SYCL_HOME=$SYCL_HOME (see docs/02_compiler.md)"
[ -d "$XILINX_VITIS" ] || echo "env.sh: Vitis $XILINX_VERSION not found in $XILINX_ROOT"
[ -d "$XILINX_XRT" ] || echo "env.sh: XRT not found in $XILINX_XRT"
python3 -c "import posix_ipc" 2> /dev/null || echo "env.sh: Python module posix_ipc missing (run scripts/install_tools.sh)"
unset REPO_DIR
