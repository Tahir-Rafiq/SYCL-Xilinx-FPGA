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
