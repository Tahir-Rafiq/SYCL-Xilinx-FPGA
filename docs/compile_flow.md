# From SYCL source to the FPGA image

One command does everything:

```bash
clang++ -std=c++20 -fsycl -fsycl-targets=fpga64_hls_hw answer_42.cpp -o answer_42_hw
```

Inside, these steps run in order.

1. Split the source. The compiler reads the file twice. The host pass compiles all of it as normal x86-64 C++. The device pass keeps only the kernel lambdas and what they call, and emits LLVM IR (LLVM 16, the compiler's own version).
2. Prepare the kernels. The driver script `sycl_vxx.py` runs SYCL-specific passes on the IR: it fixes the kernel interface (which arguments are buffers and how they map to card memory), inlines helper code and replaces SYCL built-in functions.
3. Downgrade the IR. Vitis HLS reads only the LLVM IR of its own old front end, so the driver rewrites the IR into that older form. It then assembles and links it with the Clang 3.9 tools and the `libspir64-39-hls.bc` library that come with Vitis HLS.
4. HLS (`v++ -c`), once per kernel. Vitis HLS schedules the kernel into clock cycles, produces RTL (Verilog) and packages it as a kernel object (`.xo`). The latency and resource estimates are written here.
5. Link (`v++ -l`). The kernels are placed into the U280 platform `xilinx_u280_gen3x16_xdma_1_202211_1`. Their memory ports are connected to card memory, pipes between kernels become AXI stream connections, and Vivado runs synthesis, placement, routing and timing analysis. The output is the FPGA image (`.xclbin`). This step takes most of the build time.
6. Package. The `.xclbin` is embedded as data in an object file and linked together with the host code, the SYCL runtime (`libsycl.so`) and the XRT plugin. The result is one executable.

The driver writes each step as a numbered command file in its build folder (`build_hw/<name>-<hash>/` with `bin/sycl-build`):

| File | Step |
|---|---|
| `000-run_preparation.cmd` | SYCL passes on the device IR |
| `001-prepare.cmd` | IR clean-up before the downgrade |
| `002-downgrade.cmd` | rewrite to the old IR form |
| `003-05-asm_ir.cmd.cmd` | assemble with the Clang 3.9 tools |
| `004-link_spir.cmd` | link the HLS built-in library |
| `005-vxxcomp-<kernel>.cmd` | `v++ -c` (HLS) for one kernel |
| `006-vxxlink.cmd` | `v++ -l` (link, synthesis, place and route) |

## Hardware emulation

With `-fsycl-targets=fpga64_hls_hw_emu`, steps 1 to 4 are the same. In step 5 the linker builds a simulation model of the platform and the kernels instead of a bitstream, and at run time the kernels run in the Vivado RTL simulator (xsim). The source is the same; only the target changes.

## At run time

The SYCL runtime hands the work to XRT through the XRT plugin (`libpi_xrt.so`). XRT loads the `.xclbin` into the user part of the card's shell, allocates the buffers in card memory, copies the inputs over PCIe, starts the kernels, waits for them to finish and copies the results back. The host code never touches the FPGA directly.
