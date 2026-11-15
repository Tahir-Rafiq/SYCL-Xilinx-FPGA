#!/bin/bash
# Install the small tools the compiler build and the compiler driver need, into
# your own account (~/.local), without root access:
#   cmake 3.27.9  (the compiler needs 3.20 or newer; Ubuntu 20.04 and Vitis ship older ones)
#   ninja         (the build tool the compiler's build scripts use)
#   posix_ipc     (Python module imported by the compiler driver, sycl_vxx.py)
# Usage: scripts/install_tools.sh
set -e
python3 -m pip install --user "cmake==3.27.9" ninja posix_ipc
echo "cmake:     $(~/.local/bin/cmake --version | head -1)"
echo "ninja:     $(~/.local/bin/ninja --version)"
echo "posix_ipc: $(python3 -c 'import posix_ipc; print(posix_ipc.VERSION)')"
