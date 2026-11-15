// SPDX-License-Identifier: MIT
// Matrix multiply C = A * B on the FPGA with a small systolic array.
//
// Simplified from the "matmul" FPGA reference design of oneAPI-samples
// (tag 2025.0.0, DirectProgramming/C++SYCL_FPGA/ReferenceDesigns/matmul,
// Copyright Intel Corporation, MIT License, see LICENSE.oneAPI-samples).
// Changes: int data, 16x16 matrices, 4x4 tiles, one multiplication, buffers
// and accessors instead of USM, a finite loop in the array kernel, no Intel
// FPGA attributes or ac_int. See README.md.
//
// Four kernels run at the same time and pass data through pipes:
//   FeederA  : A from card memory -> PipeA, one column of a 4-row tile per step
//   FeederB  : B from card memory -> PipeB, one row of a 4-column tile per step
//   Systolic : 4x4 processing elements, each multiply-adds into its own
//              accumulator; a finished tile goes to PipeC column by column
//   Drain    : PipeC -> C in card memory

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

constexpr int kRows = 16;   // rows of A and C
constexpr int kCommon = 16; // columns of A, rows of B
constexpr int kCols = 16;   // columns of B and C
constexpr int kTileA = 4;   // rows of the array (tile height)
constexpr int kTileB = 4;   // columns of the array (tile width)
constexpr int kTilesI = kRows / kTileA;
constexpr int kTilesJ = kCols / kTileB;
constexpr int kPipeDepth = 16;
static_assert(kRows % kTileA == 0 && kCols % kTileB == 0);

// Calls f(0), f(1), ..., f(N-1) with compile-time indices. The loop is fully
// unrolled in the source, so each index becomes its own piece of hardware.
template <typename F, std::size_t... I>
void unrolled_impl(F &&f, std::index_sequence<I...>) {
  (f(std::integral_constant<std::size_t, I>{}), ...);
}
template <std::size_t N, typename F> void unrolled(F &&f) {
  unrolled_impl(f, std::make_index_sequence<N>{});
}

struct ColA { int v[kTileA]; }; // one column of a tile of A (or of C)
struct RowB { int v[kTileB]; }; // one row of a tile of B

using PipeA = sycl::ext::intel::pipe<class PipeAName, ColA, kPipeDepth>;
using PipeB = sycl::ext::intel::pipe<class PipeBName, RowB, kPipeDepth>;
using PipeC = sycl::ext::intel::pipe<class PipeCName, ColA, kPipeDepth>;

int main() {
  // Row-major matrices with small random values, so int never overflows.
  std::vector<int> a(kRows * kCommon), b(kCommon * kCols), c(kRows * kCols, 0);
  std::srand(42);
  for (auto &x : a) x = std::rand() % 10;
  for (auto &x : b) x = std::rand() % 10;

  sycl::queue q;
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>()
            << std::endl;
  {
    sycl::buffer<int> buf_a{a.data(), sycl::range{a.size()}};
    sycl::buffer<int> buf_b{b.data(), sycl::range{b.size()}};
    sycl::buffer<int> buf_c{c.data(), sycl::range{c.size()}};

    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_a{buf_a, cgh, sycl::read_only};
      cgh.single_task<class FeederA>([=] {
        for (int ti = 0; ti < kTilesI; ti++)
          for (int tj = 0; tj < kTilesJ; tj++)
            for (int k = 0; k < kCommon; k++) {
              ColA col;
              unrolled<kTileA>([&](auto r) {
                col.v[r] = acc_a[(ti * kTileA + r) * kCommon + k];
              });
              PipeA::write(col);
            }
      });
    });

    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_b{buf_b, cgh, sycl::read_only};
      cgh.single_task<class FeederB>([=] {
        for (int ti = 0; ti < kTilesI; ti++)
          for (int tj = 0; tj < kTilesJ; tj++)
            for (int k = 0; k < kCommon; k++) {
              RowB row;
              unrolled<kTileB>([&](auto col) {
                row.v[col] = acc_b[k * kCols + tj * kTileB + col];
              });
              PipeB::write(row);
            }
      });
    });

    q.submit([&](sycl::handler &cgh) {
      cgh.single_task<class Systolic>([=] {
        for (int t = 0; t < kTilesI * kTilesJ; t++) {
          int acc[kTileA][kTileB];
          unrolled<kTileA>([&](auto r) {
            unrolled<kTileB>([&](auto col) { acc[r][col] = 0; });
          });
          // Each step every processing element (r, col) does one multiply-add.
          for (int k = 0; k < kCommon; k++) {
            ColA in_a = PipeA::read();
            RowB in_b = PipeB::read();
            unrolled<kTileA>([&](auto r) {
              unrolled<kTileB>([&](auto col) {
                acc[r][col] += in_a.v[r] * in_b.v[col];
              });
            });
          }
          // Send the finished tile out, one column per pipe write. The compiler
          // allows one write site per pipe, so this is a normal loop; the
          // column is picked with constant indices to keep acc in registers.
          for (int col = 0; col < kTileB; col++) {
            ColA out;
            unrolled<kTileA>([&](auto r) {
              int v = 0;
              unrolled<kTileB>([&](auto c) {
                if (c == col) v = acc[r][c];
              });
              out.v[r] = v;
            });
            PipeC::write(out);
          }
        }
      });
    });

    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc_c{buf_c, cgh, sycl::write_only};
      cgh.single_task<class Drain>([=] {
        for (int ti = 0; ti < kTilesI; ti++)
          for (int tj = 0; tj < kTilesJ; tj++)
            for (int col = 0; col < kTileB; col++) {
              ColA out = PipeC::read();
              unrolled<kTileA>([&](auto r) {
                acc_c[(ti * kTileA + r) * kCols + tj * kTileB + col] = out.v[r];
              });
            }
      });
    });
  } // buffer destructors wait for the kernels and copy C back into c

  int errors = 0;
  for (int i = 0; i < kRows; i++)
    for (int j = 0; j < kCols; j++) {
      int ref = 0;
      for (int k = 0; k < kCommon; k++)
        ref += a[i * kCommon + k] * b[k * kCols + j];
      if (c[i * kCols + j] != ref && errors++ < 10)
        std::cout << "C[" << i << "][" << j << "] = " << c[i * kCols + j]
                  << ", expected " << ref << std::endl;
    }
  std::cout << "C[0][0..3] = " << c[0] << " " << c[1] << " " << c[2] << " "
            << c[3] << std::endl;
  std::cout << (errors ? "FAILED: " : "PASSED: ") << errors << " errors in "
            << kRows * kCols << " values" << std::endl;
  return errors ? 1 : 0;
}
