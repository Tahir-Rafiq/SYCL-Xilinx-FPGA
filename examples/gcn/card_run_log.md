# gcn on the U280

Inference of a trained two-layer graph convolutional network (GCN) on the Cora citation graph: 2708 papers, 1433 word features per paper, 7 topics. Three kernels run on the card one after the other, and the host checks all 18956 outputs against the same network computed on the CPU, against the double precision results of the original C++ program, and against the true topics of the papers. Compared with the first three examples it adds a real workload with data read from files, float arithmetic, sparse matrices in card memory, large on-chip arrays split into banks, a scalar kernel argument (the node count), and kernels that depend on each other through buffers.

## The code

`gcn.cpp`, written for this repository after `GCN.cpp`, a double precision C++ reference program based on `gcn-seq` of parallel-gcn. `README.md` gives the origin of each file in `data/` (the graph, the trained weights W0 and W1, the true topics and the results of `GCN.cpp`).

The network:

```
Y = Ahat * ReLU(Ahat * X * W0) * W1
```

X (2708 x 1433, 49216 nonzeros) holds the word features of each paper. Ahat (2708 x 2708, 13264 nonzeros) is the normalised adjacency matrix of the citation graph with self loops. Both are sparse and stored in CSR form: a row pointer array, plus the column index and the value of each nonzero. W0 (1433 x 16) and W1 (16 x 7) are the trained weights. Each row of Y holds 7 scores for one paper, and the predicted topic is the largest of them.

```cpp
// SPDX-License-Identifier: MIT
// Two-layer graph convolutional network (GCN), inference on the Cora citation
// graph, on the FPGA:
//
//   Y = Ahat * ReLU(Ahat * X * W0) * W1
//
// Ahat (2708 x 2708, normalised adjacency matrix with self loops) and X
// (2708 x 1433, one row of word features per paper) are sparse and stored in
// CSR form (row pointers, column indices, values). W0 (1433 x 16) and W1
// (16 x 7) are the trained weights. The equations and the order of the
// operations follow the C++ reference GCN.cpp, based on parallel-gcn; see
// README.md. The predicted class of a node is the largest of its 7 outputs.
//
// Three kernels, one per sparse product, run one after the other:
//   SpMM_XW0          XW  = X * W0                (W0 on chip, 16 memories)
//   SpMM_AXW_ReLU_W1  HW1 = ReLU(Ahat * XW) * W1  (XW on chip, 16 memories)
//   SpMM_AHW1         Y   = Ahat * HW1            (HW1 on chip, 7 memories)
//
// Usage: gcn <data folder> [nodes]
//   With nodes < 2708 the program uses the subgraph of the first nodes, which
//   keeps a hardware emulation run short.

#include <sycl/sycl.hpp>
#include <sycl/ext/xilinx/fpga.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace xlx = sycl::ext::xilinx;

constexpr int kNodes = 2708;    // papers
constexpr int kFeatures = 1433; // words
constexpr int kHidden = 16;     // width of the hidden layer
constexpr int kClasses = 7;     // paper topics
constexpr int kTestFirst = 1708; // nodes 1708..2707 are the test nodes

// Calls f(0), f(1), ..., f(N-1) with compile-time indices. The loop is fully
// unrolled in the source, so each index becomes its own piece of hardware.
template <typename F, std::size_t... I>
void unrolled_impl(F &&f, std::index_sequence<I...>) {
  (f(std::integral_constant<std::size_t, I>{}), ...);
}
template <std::size_t N, typename F> void unrolled(F &&f) {
  unrolled_impl(f, std::make_index_sequence<N>{});
}

// A Rows x Cols float matrix in on-chip memory, stored row by row and split
// cyclically into Cols separate memories: element (r, c) goes to memory c, so
// a whole row can be read in one clock cycle. (A 2D partition_ndarray would be
// clearer, but this compiler attaches its pragma to the wrong array type and
// Vitis HLS crashes on it.)
template <std::size_t Rows, std::size_t Cols>
using Banked =
    xlx::partition_array<float, Rows * Cols, xlx::partition::cyclic<Cols, 1>>;

struct Csr {
  std::vector<int> rowptr, col;
  std::vector<float> val;
};

// Reads all numbers of a CSV file, row by row.
template <typename T> std::vector<T> read_csv(const std::string &path) {
  std::ifstream f(path);
  if (!f) {
    std::cerr << "Cannot open " << path << std::endl;
    std::exit(2);
  }
  std::vector<T> v;
  std::string line;
  while (std::getline(f, line)) {
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream s(line);
    T x;
    while (s >> x) v.push_back(x);
  }
  return v;
}

Csr read_csr(const std::string &prefix) {
  return {read_csv<int>(prefix + "_rowptr_fp.csv"),
          read_csv<int>(prefix + "_col_fp.csv"),
          read_csv<float>(prefix + "_value_fp.csv")};
}

// The first n rows of m. With square = true, also only the first n columns.
Csr first_rows(const Csr &m, int n, bool square) {
  Csr s;
  s.rowptr.push_back(0);
  for (int r = 0; r < n; r++) {
    for (int p = m.rowptr[r]; p < m.rowptr[r + 1]; p++)
      if (!square || m.col[p] < n) {
        s.col.push_back(m.col[p]);
        s.val.push_back(m.val[p]);
      }
    s.rowptr.push_back(s.col.size());
  }
  return s;
}

// CPU versions of the kernels, with the same order of operations.
std::vector<float> spmm(const Csr &a, const std::vector<float> &b, int cols) {
  int n = a.rowptr.size() - 1;
  std::vector<float> c(n * cols, 0.0f);
  for (int r = 0; r < n; r++)
    for (int p = a.rowptr[r]; p < a.rowptr[r + 1]; p++)
      for (int j = 0; j < cols; j++)
        c[r * cols + j] += a.val[p] * b[a.col[p] * cols + j];
  return c;
}

std::vector<float> cpu_gcn(const Csr &x, const Csr &a,
                           const std::vector<float> &w0,
                           const std::vector<float> &w1) {
  int n = a.rowptr.size() - 1;
  std::vector<float> h = spmm(a, spmm(x, w0, kHidden), kHidden);
  std::vector<float> hw1(n * kClasses);
  for (int r = 0; r < n; r++)
    for (int j = 0; j < kClasses; j++) {
      float s = 0.0f;
      for (int k = 0; k < kHidden; k++)
        s += std::max(h[r * kHidden + k], 0.0f) * w1[k * kClasses + j];
      hw1[r * kClasses + j] = s;
    }
  return spmm(a, hw1, kClasses);
}

template <typename T> int argmax(const T *v) {
  return std::max_element(v, v + kClasses) - v;
}

int main(int argc, char *argv[]) {
  std::string dir = argc > 1 ? argv[1] : "data";
  int n = argc > 2 ? std::atoi(argv[2]) : kNodes;
  if (n < 1 || n > kNodes) {
    std::cerr << "nodes must be between 1 and " << kNodes << std::endl;
    return 2;
  }

  Csr x_full = read_csr(dir + "/X"), a_full = read_csr(dir + "/Ahat");
  std::vector<float> w0 = read_csv<float>(dir + "/W0.csv");
  std::vector<float> w1 = read_csv<float>(dir + "/W1.csv");
  std::vector<int> labels = read_csv<int>(dir + "/labels.csv");
  if (x_full.rowptr.size() != kNodes + 1 || a_full.rowptr.size() != kNodes + 1 ||
      w0.size() != kFeatures * kHidden || w1.size() != kHidden * kClasses ||
      labels.size() != kNodes) {
    std::cerr << "Unexpected sizes in the data files" << std::endl;
    return 2;
  }
  Csr x = first_rows(x_full, n, false), a = first_rows(a_full, n, true);
  std::cout << "Graph: " << n << " nodes, " << x.col.size()
            << " nonzeros in X, " << a.col.size() << " in Ahat" << std::endl;

  std::vector<float> y(n * kClasses);
  sycl::queue q;
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>()
            << std::endl;
  double ms[3];
  {
    sycl::buffer<int> b_x_rowptr{x.rowptr.data(), sycl::range{x.rowptr.size()}};
    sycl::buffer<int> b_x_col{x.col.data(), sycl::range{x.col.size()}};
    sycl::buffer<float> b_x_val{x.val.data(), sycl::range{x.val.size()}};
    sycl::buffer<int> b_a_rowptr{a.rowptr.data(), sycl::range{a.rowptr.size()}};
    sycl::buffer<int> b_a_col{a.col.data(), sycl::range{a.col.size()}};
    sycl::buffer<float> b_a_val{a.val.data(), sycl::range{a.val.size()}};
    sycl::buffer<float> b_w0{w0.data(), sycl::range{w0.size()}};
    sycl::buffer<float> b_w1{w1.data(), sycl::range{w1.size()}};
    sycl::buffer<float> b_xw{sycl::range{std::size_t(n) * kHidden}};
    sycl::buffer<float> b_hw1{sycl::range{std::size_t(n) * kClasses}};
    sycl::buffer<float> b_y{y.data(), sycl::range{y.size()}};

    // Each kernel is waited for before the next one is submitted: the XRT
    // plugin of this compiler starts a kernel without waiting for the kernels
    // whose results it reads.
    auto t0 = std::chrono::steady_clock::now();

    // Kernel 1: XW = X * W0
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor rowptr{b_x_rowptr, cgh, sycl::read_only};
      sycl::accessor col{b_x_col, cgh, sycl::read_only};
      sycl::accessor val{b_x_val, cgh, sycl::read_only};
      sycl::accessor w0_g{b_w0, cgh, sycl::read_only};
      sycl::accessor xw{b_xw, cgh, sycl::write_only, sycl::no_init};
      cgh.single_task<class SpMM_XW0>([=] {
        Banked<kFeatures, kHidden> w;
        for (int i = 0; i < kFeatures * kHidden; i++)
          w[i] = w0_g[i];
        for (int r = 0; r < n; r++) {
          float acc[kHidden] = {};
          int first = rowptr[r], last = rowptr[r + 1];
          for (int p = first; p < last; p++) {
            int c = col[p];
            float v = val[p];
            // one row of W0 per nonzero: 16 multiply-adds side by side
            unrolled<kHidden>([&](auto j) { acc[j] += v * w[c * kHidden + j]; });
          }
          unrolled<kHidden>([&](auto j) { xw[r * kHidden + j] = acc[j]; });
        }
      });
    }).wait();
    auto t1 = std::chrono::steady_clock::now();

    // Kernel 2: HW1 = ReLU(Ahat * XW) * W1
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor rowptr{b_a_rowptr, cgh, sycl::read_only};
      sycl::accessor col{b_a_col, cgh, sycl::read_only};
      sycl::accessor val{b_a_val, cgh, sycl::read_only};
      sycl::accessor xw_g{b_xw, cgh, sycl::read_only};
      sycl::accessor w1_g{b_w1, cgh, sycl::read_only};
      sycl::accessor hw1{b_hw1, cgh, sycl::write_only, sycl::no_init};
      cgh.single_task<class SpMM_AXW_ReLU_W1>([=] {
        Banked<kNodes, kHidden> xw;
        for (int i = 0; i < n * kHidden; i++)
          xw[i] = xw_g[i];
        float w1[kHidden][kClasses];
        unrolled<kHidden>([&](auto k) {
          unrolled<kClasses>([&](auto j) { w1[k][j] = w1_g[k * kClasses + j]; });
        });
        for (int r = 0; r < n; r++) {
          float h[kHidden] = {};
          int first = rowptr[r], last = rowptr[r + 1];
          for (int p = first; p < last; p++) {
            int c = col[p];
            float v = val[p];
            unrolled<kHidden>([&](auto k) { h[k] += v * xw[c * kHidden + k]; });
          }
          // ReLU, then this row times W1
          unrolled<kHidden>([&](auto k) { h[k] = h[k] > 0.0f ? h[k] : 0.0f; });
          unrolled<kClasses>([&](auto j) {
            float s = 0.0f;
            unrolled<kHidden>([&](auto k) { s += h[k] * w1[k][j]; });
            hw1[r * kClasses + j] = s;
          });
        }
      });
    }).wait();
    auto t2 = std::chrono::steady_clock::now();

    // Kernel 3: Y = Ahat * HW1
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor rowptr{b_a_rowptr, cgh, sycl::read_only};
      sycl::accessor col{b_a_col, cgh, sycl::read_only};
      sycl::accessor val{b_a_val, cgh, sycl::read_only};
      sycl::accessor hw1_g{b_hw1, cgh, sycl::read_only};
      sycl::accessor y_g{b_y, cgh, sycl::write_only, sycl::no_init};
      cgh.single_task<class SpMM_AHW1>([=] {
        Banked<kNodes, kClasses> hw1;
        for (int i = 0; i < n * kClasses; i++)
          hw1[i] = hw1_g[i];
        for (int r = 0; r < n; r++) {
          float acc[kClasses] = {};
          int first = rowptr[r], last = rowptr[r + 1];
          for (int p = first; p < last; p++) {
            int c = col[p];
            float v = val[p];
            unrolled<kClasses>([&](auto j) { acc[j] += v * hw1[c * kClasses + j]; });
          }
          unrolled<kClasses>([&](auto j) { y_g[r * kClasses + j] = acc[j]; });
        }
      });
    }).wait();
    auto t3 = std::chrono::steady_clock::now();

    using ms_t = std::chrono::duration<double, std::milli>;
    ms[0] = ms_t(t1 - t0).count();
    ms[1] = ms_t(t2 - t1).count();
    ms[2] = ms_t(t3 - t2).count();
  } // the buffer destructors copy Y back into y

  std::cout << "Kernel times (ms, measured on the host): SpMM_XW0 " << ms[0]
            << ", SpMM_AXW_ReLU_W1 " << ms[1] << ", SpMM_AHW1 " << ms[2]
            << std::endl;

  // Compare with the same computation on the CPU.
  std::vector<float> ref = cpu_gcn(x, a, w0, w1);
  int errors = 0, class_diff = 0;
  float max_diff = 0.0f;
  for (int i = 0; i < n * kClasses; i++) {
    float d = std::fabs(y[i] - ref[i]);
    max_diff = std::max(max_diff, d);
    if (d > 1e-4f * std::max(1.0f, std::fabs(ref[i])) && errors++ < 10)
      std::cout << "Y[" << i / kClasses << "][" << i % kClasses << "] = " << y[i]
                << ", expected " << ref[i] << std::endl;
  }
  for (int r = 0; r < n; r++)
    if (argmax(&y[r * kClasses]) != argmax(&ref[r * kClasses])) class_diff++;
  std::cout << "Largest difference to the CPU result: " << max_diff << ", "
            << class_diff << " nodes with a different class" << std::endl;
  std::cout << "Y[0] =";
  for (int j = 0; j < kClasses; j++) std::cout << " " << y[j];
  std::cout << ", class " << argmax(&y[0]) << std::endl;

  if (n == kNodes) {
    // The original double precision results of GCN.cpp, and the true classes
    std::vector<double> orig = read_csv<double>(dir + "/AH0W1.csv");
    double max_orig = 0.0;
    for (int i = 0; i < n * kClasses; i++)
      max_orig = std::max(max_orig, std::fabs(y[i] - orig[i]));
    if (max_orig > 1e-3) errors++;
    std::cout << "Largest difference to AH0W1.csv (GCN.cpp, double): "
              << max_orig << std::endl;
    int right = 0, right_test = 0;
    for (int r = 0; r < n; r++)
      if (argmax(&y[r * kClasses]) == labels[r]) {
        right++;
        if (r >= kTestFirst) right_test++;
      }
    std::cout << "Accuracy: " << right << " of " << n << " nodes ("
              << 100.0 * right / n << "%), test nodes " << right_test << " of "
              << n - kTestFirst << " (" << 100.0 * right_test / (n - kTestFirst)
              << "%)" << std::endl;
  }

  std::cout << (errors ? "FAILED: " : "PASSED: ") << errors << " errors in "
            << n * kClasses << " values" << std::endl;
  return errors ? 1 : 0;
}
```

What each part does:

- The constants are the sizes of Cora and of the network. Nodes 1708 to 2707 are the test nodes, the papers that were not used in training.
- `unrolled<N>(f)` calls `f(0)`, ..., `f(N-1)` with compile-time indices, as in matmul. Each call becomes its own piece of hardware.
- `Banked<Rows, Cols>` is a float matrix in on-chip memory, split into `Cols` separate memories with element (r, c) in memory c, so a whole row can be read in one clock cycle. It is a one-dimensional `partition_array` indexed as `r * Cols + c`, because a two-dimensional `partition_ndarray` crashes Vitis HLS with this compiler.
- `Csr`, `read_csv` and `read_csr` read the CSV files into host vectors. `first_rows` cuts the graph down to its first n nodes; the optional second argument of the program uses it to keep hardware emulation runs short. The runs below use the full graph.
- `spmm` and `cpu_gcn` compute the same network on the CPU, in float and in the same order of operations as the kernels, so the FPGA result can be compared value by value.
- `main` reads the data, checks the sizes and prints the `Graph:` line. Then `sycl::queue q` picks the device with the default selector, and the program prints its name.
- Eleven buffers wrap the inputs and the result `y`. `b_xw` (XW, 2708 x 16) and `b_hw1` (HW1, 2708 x 7) have no host data: they carry the intermediate results from one kernel to the next and stay in card memory.
- Kernel 1, `SpMM_XW0`, computes XW = X * W0. It first copies W0 (22928 values) from card memory into a `Banked<1433, 16>` on chip, one value per clock cycle. Then, for each row of X, it walks the nonzeros of that row. For each nonzero it reads the matching row of W0 in one cycle and does 16 multiply-adds side by side. The 16 sums of the row are written to XW.
- Kernel 2, `SpMM_AXW_ReLU_W1`, copies XW on chip the same way and W1 (16 x 7 values) into registers. For each row of Ahat it computes the 16 values of Ahat * XW, applies ReLU (negative values become 0) and multiplies the row by W1 with 112 multiply-adds, all unrolled. The 7 results go to HW1.
- Kernel 3, `SpMM_AHW1`, copies HW1 on chip into 7 memories and computes Y = Ahat * HW1 row by row, with 7 multiply-adds per nonzero.
- The node count `n` is captured by each kernel lambda, so it reaches the kernels as a scalar argument.
- Each `q.submit(...)` is followed by `.wait()`. The XRT plugin of this compiler starts a kernel without waiting for the kernels whose results it reads, so without the wait the second kernel could read XW before the first kernel has written it.
- `t0` to `t3` are host clock readings around each submit and wait. The differences are the times the host waited for each kernel, which include more than the kernel itself (see the log explanation).
- The closing `}` of the buffer block destroys the buffers, which copies Y back into the host vector `y`.
- The host computes the network again with `cpu_gcn` and compares all n x 7 values. A value counts as an error if it differs by more than 1e-4 (relative for values above 1), and up to 10 wrong values are printed. The host also counts the nodes whose predicted topic differs, and prints the outputs of node 0 with its topic.
- With the full graph it also compares Y with `AH0W1.csv`, the double precision output of the original C++ program (one more error if any value differs by more than 1e-3), and compares the predicted topics with the true topics in `labels.csv`.
- The program prints `PASSED` or `FAILED` with the number of errors. The exit status is 0 only with no errors.

## Build and run

Built for the U280 on a 32-core host with 64 GB of memory (99 minutes; one 52 MB program with the FPGA image inside). Run on 11 November 2026, right after matmul, on a host whose U280 runs the shell `xilinx_u280_gen3x16_xdma_base_1`. The data folder is the first argument. From the repository folder:

```bash
source env.sh
cd examples/gcn
sycl-build gcn.cpp hw
time ./gcn_hw data; echo "exit $?"
time ./gcn_hw data; echo "exit $?"
```

## Log, first run

```
Graph: 2708 nodes, 49216 nonzeros in X, 13264 in Ahat
SYCL_PI_TRACE[basic]: Plugin found and successfully loaded: libpi_xrt.so [ PluginVersion: 12.23 ]
SYCL_PI_TRACE[all]: Requested device_type: info::device_type::automatic
SYCL_PI_TRACE[all]: Selected device: -> final score = 1125
SYCL_PI_TRACE[all]:   platform: Xilinx XRT
SYCL_PI_TRACE[all]:   device: xilinx_u280_gen3x16_xdma_base_1
warning: queue created with unhandled properties
Device: xilinx_u280_gen3x16_xdma_base_1
Kernel times (ms, measured on the host): SpMM_XW0 7723.29, SpMM_AXW_ReLU_W1 32.1883, SpMM_AHW1 29.0329
Largest difference to the CPU result: 0, 0 nodes with a different class
Y[0] = -1.09505 -0.821812 -0.260021 2.1607 -0.100187 -1.17637 -1.16116, class 3
Largest difference to AH0W1.csv (GCN.cpp, double): 2.05361e-06
Accuracy: 2227 of 2708 nodes (82.2378%), test nodes 811 of 1000 (81.1%)
PASSED: 0 errors in 18956 values

real    0m8.050s
user    0m0.221s
sys     0m0.168s
exit 0
```

## Log, second run (right after the first)

```
Graph: 2708 nodes, 49216 nonzeros in X, 13264 in Ahat
SYCL_PI_TRACE[basic]: Plugin found and successfully loaded: libpi_xrt.so [ PluginVersion: 12.23 ]
SYCL_PI_TRACE[all]: Requested device_type: info::device_type::automatic
SYCL_PI_TRACE[all]: Selected device: -> final score = 1125
SYCL_PI_TRACE[all]:   platform: Xilinx XRT
SYCL_PI_TRACE[all]:   device: xilinx_u280_gen3x16_xdma_base_1
warning: queue created with unhandled properties
Device: xilinx_u280_gen3x16_xdma_base_1
Kernel times (ms, measured on the host): SpMM_XW0 96.102, SpMM_AXW_ReLU_W1 29.6226, SpMM_AHW1 27.0476
Largest difference to the CPU result: 0, 0 nodes with a different class
Y[0] = -1.09505 -0.821812 -0.260021 2.1607 -0.100187 -1.17637 -1.16116, class 3
Largest difference to AH0W1.csv (GCN.cpp, double): 2.05361e-06
Accuracy: 2227 of 2708 nodes (82.2378%), test nodes 811 of 1000 (81.1%)
PASSED: 0 errors in 18956 values

real    0m0.534s
user    0m0.191s
sys     0m0.111s
exit 0
```

Only the kernel times and the run times differ from the first run.

## What the log shows

| Log line | Meaning |
|---|---|
| `Graph: 2708 nodes, 49216 nonzeros in X, 13264 in Ahat` | Printed by the program after it has read the data files: the full graph and the number of nonzeros of the two sparse matrices. It comes before the `SYCL_PI_TRACE` lines because the program reads the data before it creates the queue, and the runtime loads its plugin only when the first queue is created. |
| `Plugin found and successfully loaded: libpi_xrt.so [ PluginVersion: 12.23 ]` | The SYCL runtime (`libsycl.so`) loaded its XRT backend plugin, which talks to the card through AMD's XRT library. 12.23 is the version of the interface between runtime and plugin. The `SYCL_PI_TRACE` lines appear because `env.sh` sets `SYCL_PI_TRACE=1`. |
| `Requested device_type: info::device_type::automatic` | The program creates its queue with `sycl::queue q;` and does not ask for a device type, so the default selector scores every visible device. |
| `Selected device: -> final score = 1125` | The default selector's score: 75 because the device is an accelerator, plus 1000 because the program contains an image this device can run, plus 50 because the XRT backend is preferred to the OpenCL backend for AMD FPGAs. `ONEAPI_DEVICE_SELECTOR=xrt:*` (set by `env.sh`) leaves the card as the only candidate. |
| `platform: Xilinx XRT`, `device: xilinx_u280_gen3x16_xdma_base_1` | The chosen backend and device. On the card the device is named after the shell running on it. In hardware emulation the same device is named after the platform, `xilinx_u280_gen3x16_xdma_1_202211_1`. |
| `warning: queue created with unhandled properties` | Harmless. A SYCL queue is out-of-order unless the program asks for an in-order queue, and the runtime passes that flag to the plugin. The XRT plugin only handles the profiling flag, warns about any other one and ignores it. |
| `Device: xilinx_u280_gen3x16_xdma_base_1` | Printed by the program itself, from the device information of the queue. |
| `Kernel times (ms, measured on the host): SpMM_XW0 ..., SpMM_AXW_ReLU_W1 ..., SpMM_AHW1 ...` | For each kernel, the host time from its submission to the end of its `wait()`. This includes more than the kernel: the copies of the kernel's inputs that are not yet on the card, starting the kernel through XRT, and waiting for it. The first value also includes the work the runtime does at the first kernel submission. In the first run that includes loading the program's FPGA image onto the card (the card still held the matmul image), which is almost all of the 7723 ms. The times are not part of the test. The second run is discussed below the table. |
| `Largest difference to the CPU result: 0, 0 nodes with a different class` | The largest absolute difference between the 18956 FPGA outputs and the CPU float computation is 0, so the FPGA results are bit for bit equal to the CPU result. The kernels and `cpu_gcn` do the same float operations in the same order, so they round the same way. Because no value differs, no node gets a different predicted topic. |
| `Y[0] = -1.09505 -0.821812 -0.260021 2.1607 -0.100187 -1.17637 -1.16116, class 3` | The 7 outputs of node 0 (the first paper), one score per topic. The largest is the fourth, 2.1607, so the predicted topic is 3 (topics are numbered 0 to 6). The true topic of this paper in `labels.csv` is also 3. The values are the same as in hardware emulation. |
| `Largest difference to AH0W1.csv (GCN.cpp, double): 2.05361e-06` | Comparison with the output of the original C++ program, which computes in double precision. The difference of about 2e-6 comes from rounding in float versus double. A difference above 1e-3 would count as an error. |
| `Accuracy: 2227 of 2708 nodes (82.2378%), test nodes 811 of 1000 (81.1%)` | The predicted topic of each paper (its largest output) compared with the true topic in `labels.csv`: 2227 of all 2708 papers and 811 of the 1000 test papers are right. These are the same numbers as for the C++ program and the training run, so the network on the FPGA classifies exactly like the trained network. The accuracy is reported but is not part of the pass test. |
| `PASSED: 0 errors in 18956 values` | 18956 = 2708 nodes x 7 outputs. No value differs from the CPU result by more than the tolerance, and the difference to `AH0W1.csv` is below 1e-3. With errors, up to 10 lines `Y[node][output] = ..., expected ...` would come first and this line would start with `FAILED`. |
| `real`, `user`, `sys` | From the shell's `time` command. `real` is the wall-clock time from start to end, including any waiting; `user` is the CPU time in the program's own code and its libraries (SYCL runtime, XRT); `sys` is the CPU time the operating system spent on the program's behalf (calls into the card driver, memory mapping, reading the program file). |
| `exit 0` | The program returns 0 only when there are no errors. This is the test result. |

The times: in the first run `real` (8.05 s) is far larger than `user + sys` (0.39 s), so the program was waiting most of the time, and the kernel times show where: 7.7 s in the first kernel submission, for the image load. `user` (0.19 to 0.22 s) is higher than in the other three programs (0.02 to 0.05 s) because the host reads about 1.9 MB of CSV text and computes the whole network a second time on the CPU.

In the second run the image was already on the card and the whole run took 0.53 s. The first kernel time, 96 ms, still includes the one-time work at the first submission. XRT reads the image from the program, finds it already loaded and skips the load. The runtime creates the kernel objects and copies the inputs of the first kernel, X and W0 (about 0.5 MB), to the card. This log does not show how much of the 96 ms is the kernel itself.

The other two kernel times, 29.6 and 27.0 ms, are about 10 times what hardware emulation showed for these kernels (2.9 and 2.5 ms of simulated time at 300 MHz). Before kernel 2 the runtime copies only Ahat and W1 (about 120 KB), and before kernel 3 nothing, because all its inputs are already on the card. Between start and wait the XRT plugin only calls XRT's own start and wait functions. So most of the 27 ms of kernel 3 is the kernel running on the card, about 10 µs for each of the 2708 rows. The most likely cause is the latency of card memory. For every row the kernel first reads the two row pointers and then the column indices and values of that row from card memory, and the loop restarts for each row. HLS planned the loops for a memory latency of 64 cycles, and the memory model of hardware emulation is approximate. Neither has been checked against the card yet. What is certain is that this first version is slower on the card than the C++ program on a desktop CPU: kernels 2 and 3 alone take about 57 ms, against 1.8 ms for the whole C++ program. The two changes listed in README.md (one loop over all nonzeros instead of one loop per row, and several partial sums per output) aim at this per-row cost. The small difference between the two runs for kernels 2 and 3 (32.2 and 29.0 ms in the first run) is run-to-run variation.

## Result

Passed: all 18956 outputs computed by the three kernels on the FPGA are bit for bit equal to the CPU float result and within 2.1e-6 of the double precision C++ program. 811 of the 1000 test papers are classified correctly (81.1%), and the exit status was 0 in both runs. First run 8.05 s (with the image load), second run 0.53 s. The kernel times measured on the host in the second run were 96 ms (including the one-time setup), 29.6 ms and 27.0 ms.
