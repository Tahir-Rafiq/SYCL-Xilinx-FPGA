# Graph convolutional network on Cora

`gcn.cpp` runs the inference of a trained two-layer graph convolutional network (GCN) on the Cora citation graph on the U280, with the same compiler and flow as the other three examples. It computes

```
Y = Ahat * ReLU(Ahat * X * W0) * W1
```

and takes the predicted topic of each paper as the index of the largest of its 7 outputs.

## The data

Cora has 2708 papers (the nodes), 7 topics and a vocabulary of 1433 words. In this version of the graph there are 5278 citation links; stored in both directions and with one self loop per node, they give the 13264 nonzeros of Ahat. The files in `data/` are:

| File | Contents |
|---|---|
| `X_{rowptr,col,value}_fp.csv` | X, 2708 x 1433 word features, row-normalised, CSR, 49216 nonzeros |
| `Ahat_{rowptr,col,value}_fp.csv` | Ahat = D^-1/2 (A + I) D^-1/2, 2708 x 2708, CSR, 13264 nonzeros |
| `W0.csv`, `W1.csv` | trained weights, 1433 x 16 and 16 x 7 |
| `AH0W1.csv` | Y computed in double precision by the C++ reference `GCN.cpp` (2708 x 7) |
| `labels.csv` | true topic (0 to 6) of each paper |

The node order is the one of the usual Planetoid split of Cora (Yang, Cohen and Salakhutdinov, ICML 2016): nodes 0 to 139 are the training nodes and nodes 1708 to 2707 the 1000 test nodes. The network is the GCN of Kipf and Welling (ICLR 2017).

## Origin

The sources of the model, the data and the weights are listed with their licenses and citations in [ACKNOWLEDGEMENTS.md](../../ACKNOWLEDGEMENTS.md). In short:

- The weights were trained with `gcn-seq` of [parallel-gcn](https://github.com/cai-lw/parallel-gcn), a C++ reimplementation of the original GCN code of Kipf and Welling, with a fixed random seed. The training run reached 811 of 1000 test nodes, the same as this example.
- The CSR files and `labels.csv` were converted from the Cora files of parallel-gcn (`data/cora.graph` and `data/cora.svmlight`). `AH0W1.csv` was computed by `GCN.cpp`, a sequential C++ reference program based on `gcn-seq` that uses double precision. `gcn.cpp` follows its equations and order of operations, in float.
- The output of the training run itself is not used as the reference: the loss function of `gcn-seq` subtracts the row maximum from the outputs in place, so 500 validation rows of its output are shifted. Its predicted classes are the same, and its other rows agree with `GCN.cpp` to within 9e-6.

## The three kernels

| Kernel | Computes | Kept in on-chip memory |
|---|---|---|
| `SpMM_XW0` | XW = X * W0 | W0 (1433 x 16 floats, 92 KB), one memory per column |
| `SpMM_AXW_ReLU_W1` | HW1 = ReLU(Ahat * XW) * W1 | XW (2708 x 16, 173 KB), one memory per column; W1 in registers |
| `SpMM_AHW1` | Y = Ahat * HW1 | HW1 (2708 x 7, 76 KB), one memory per column |

Each kernel walks the nonzeros of its sparse matrix row by row. For each nonzero it reads one full row of the dense matrix from on-chip memory in a single cycle and does all its multiply-adds in parallel (16, or 7 in the last kernel). Each dense matrix is stored row by row in a one-dimensional `sycl::ext::xilinx::partition_array` with `partition::cyclic<16, 1>` (or `<7, 1>`): element (r, c) goes to memory c, so the 16 values of a row sit in 16 different memories. A two-dimensional `partition_ndarray` with `partition::complete<2>` would say this more directly, but in this compiler it attaches the HLS pragma to an array of the wrong type and Vitis HLS crashes on it. The sparse matrices stay in card memory and are read in order.

The kernels run one after the other, and the program waits for each before it submits the next. The XRT plugin of this compiler starts a kernel without waiting for the kernels it depends on (`xrt_piEnqueueKernelLaunch` in `sycl/plugins/xrt/pi_xrt.cpp` ignores its wait list), so without the waits the second kernel could read XW before the first kernel has written it.

## Checks

The program computes the same network on the CPU in float, in the same order of operations, and compares every output (tolerance 1e-4, relative for values above 1). On the full graph it also compares with `AH0W1.csv` (tolerance 1e-3) and prints the accuracy. Expected values for the full graph, from the CPU computation: largest difference to `AH0W1.csv` about 2e-6, 2227 of 2708 nodes correct (82.2%), 811 of 1000 test nodes correct (81.1%).

## Build and run

From the repository folder, after `source env.sh`:

```bash
cd examples/gcn
sycl-build gcn.cpp hw_emu                # about 7 minutes
./gcn_hw_emu data 128; echo "exit $?"    # first 128 nodes, about 4 minutes
sycl-build gcn.cpp hw                    # about 100 minutes
./gcn_hw data; echo "exit $?"            # full graph, on the card
```

The first argument is the data folder. The optional second one is the number of nodes: with fewer than 2708 the program uses the subgraph of the first nodes and checks only against the CPU result, because the outputs of a subgraph differ from those of the full graph. In emulation, 128 nodes take about 4 minutes and end with `PASSED: 0 errors in 896 values`; the full graph takes about 72 minutes.

Expected output for the full graph, in emulation and on the card:

```
Graph: 2708 nodes, 49216 nonzeros in X, 13264 in Ahat
Device: xilinx_u280_gen3x16_xdma_base_1
Kernel times (ms, measured on the host): SpMM_XW0 ..., SpMM_AXW_ReLU_W1 ..., SpMM_AHW1 ...
Largest difference to the CPU result: 0, 0 nodes with a different class
Y[0] = -1.09505 -0.821812 -0.260021 2.1607 -0.100187 -1.17637 -1.16116, class 3
Largest difference to AH0W1.csv (GCN.cpp, double): 2.05361e-06
Accuracy: 2227 of 2708 nodes (82.2378%), test nodes 811 of 1000 (81.1%)
PASSED: 0 errors in 18956 values
```

and exit status 0 (in emulation the device is named `xilinx_u280_gen3x16_xdma_1_202211_1`). The simulator writes a large log of harmless DSP48 warnings to `.run/<pid>/hw_emu/device0/binary_0/behav_waveform/xsim/output.txt` (280 MB for the full graph); delete it after the run. [card_run_log.md](card_run_log.md) has the full log of a run on the card with an explanation of every line.

## Results

From the HLS reports (`build_hw_emu/gcn-<hash>/vxx_comp_tmp/.../solution/syn/report/`):

| Kernel | Copy loop II | Sparse loop II (depth) | BRAM_18K | URAM | DSP | FF | LUT |
|---|---|---|---|---|---|---|---|
| SpMM_XW0 | 1 (22928 values of W0) | 7 (93 cycles) | 66 | 0 | 15 | 7769 | 8521 |
| SpMM_AXW_ReLU_W1 | 1 (n x 16 values of XW) | 7 (93 cycles) | 2 | 16 | 329 | 35853 | 21866 |
| SpMM_AHW1 | 1 (n x 7 values of HW1) | 7 (162 cycles) | 2 | 7 | 25 | 19512 | 16721 |

W0 went into block RAM (16 memories of 1433 words); XW and HW1 (2708 words per column) went into UltraRAM, one per column. Most of the 329 DSPs of the second kernel are the 16 x 7 product with W1, which is fully unrolled. The sparse loops start one nonzero every 7 cycles (II = 7): each `acc += v * w` needs the result of the previous add, and a float add takes 7 cycles at 300 MHz.

The hardware build took 99 minutes on a 32-core host and met timing at 300 MHz (worst negative slack +0.016 ns, Vivado estimate 304.0 MHz). On the card, all 18956 outputs were bit for bit equal to the CPU float result. The first run took 8.05 s (almost all of it loading the FPGA image) and a second run 0.53 s. Kernel times measured on the host in the second run: SpMM_XW0 96 ms (including one-time setup and the copy of X and W0), SpMM_AXW_ReLU_W1 29.6 ms, SpMM_AHW1 27.0 ms.

The last two are about 10 times the times of hardware emulation (2.9 and 2.5 ms of simulated time), about 10 µs per row of Ahat, and much slower than the 1.8 ms of the double precision C++ program on a desktop CPU. The most likely cause is the latency of card memory, paid for every row: each row restarts the sparse loop, which first reads the row pointers and then the nonzeros of the row. This has not been measured. Two changes should speed the kernels up: one flat loop over all nonzeros instead of a loop per row, and several partial sums per output to hide the 7-cycle add. They were left out to keep this version close to the equations.
