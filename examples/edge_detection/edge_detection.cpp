// Edge detection with a 3x3 Sobel filter: the same filter on the CPU and on the FPGA.
//
// The kernel is the one of the triSYCL example sycl/test/vitis/edge_detection
// (https://github.com/triSYCL/sycl, Apache License 2.0 with LLVM exceptions), itself a
// SYCL translation of the Xilinx SDAccel edge_detection example. Changes for the seminar:
// a small BMP reader and writer in place of OpenCV, an optional crop for emulation, the
// same filter as plain C++ on the CPU with its time, and a check of the FPGA output.
//
// Usage: ./edge_detection_hw_emu <image.bmp> [crop]
//   crop: the FPGA filters only the central crop x crop pixels (emulation is slow);
//         the CPU filters both the crop and the full image.
// Writes output_fpga.bmp (FPGA result) and output_cpu.bmp (CPU result, full image).
//
// Built with -DCPU_ONLY and without -fsycl, the program has only the CPU part and builds in
// seconds, for example while the FPGA build runs:
//   clang++ -std=c++20 -O2 -DCPU_ONLY edge_detection.cpp -o edge_detection_cpu

#ifndef CPU_ONLY
#include <sycl/sycl.hpp>
#include <sycl/ext/xilinx/fpga.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using uchar = unsigned char;

// Read a 24-bit BMP file and convert it to 8-bit gray (rows top to bottom)
static bool read_bmp_gray(const std::string &name, std::vector<uchar> &gray,
                          size_t &width, size_t &height) {
  std::ifstream f(name, std::ios::binary);
  if (!f)
    return false;
  unsigned char h[54];
  if (!f.read(reinterpret_cast<char *>(h), 54) || h[0] != 'B' || h[1] != 'M')
    return false;
  auto u32 = [&](int o) { return uint32_t(h[o]) | uint32_t(h[o + 1]) << 8 |
                                 uint32_t(h[o + 2]) << 16 | uint32_t(h[o + 3]) << 24; };
  uint32_t offset = u32(10);
  int32_t w = int32_t(u32(18)), hh = int32_t(u32(22));
  int bpp = h[28] | h[29] << 8;
  if (bpp != 24 || w <= 0 || hh == 0)
    return false;
  bool bottom_up = hh > 0;
  width = size_t(w);
  height = size_t(bottom_up ? hh : -hh);
  size_t row = (width * 3 + 3) / 4 * 4;
  std::vector<uchar> data(row * height);
  f.seekg(offset);
  if (!f.read(reinterpret_cast<char *>(data.data()), data.size()))
    return false;
  gray.resize(width * height);
  for (size_t y = 0; y < height; ++y) {
    const uchar *p = &data[(bottom_up ? height - 1 - y : y) * row];
    for (size_t x = 0; x < width; ++x) {
      unsigned b = p[3 * x], g = p[3 * x + 1], r = p[3 * x + 2];
      gray[y * width + x] = uchar((77 * r + 150 * g + 29 * b + 128) >> 8);
    }
  }
  return true;
}

// Write an 8-bit gray image as a 24-bit BMP file
static void write_bmp_gray(const std::string &name, const std::vector<uchar> &gray,
                           size_t width, size_t height) {
  size_t row = (width * 3 + 3) / 4 * 4;
  uint32_t size = uint32_t(54 + row * height);
  unsigned char h[54] = {'B', 'M'};
  auto put = [&](int o, uint32_t v) { for (int i = 0; i < 4; ++i) h[o + i] = uchar(v >> (8 * i)); };
  put(2, size); put(10, 54); put(14, 40); put(18, uint32_t(width)); put(22, uint32_t(height));
  h[26] = 1; h[28] = 24; put(34, uint32_t(row * height));
  std::ofstream f(name, std::ios::binary);
  f.write(reinterpret_cast<char *>(h), 54);
  std::vector<uchar> line(row, 0);
  for (size_t y = height; y-- > 0;) {
    for (size_t x = 0; x < width; ++x)
      line[3 * x] = line[3 * x + 1] = line[3 * x + 2] = gray[y * width + x];
    f.write(reinterpret_cast<char *>(line.data()), row);
  }
}

// The same Sobel filter in plain C++ for the CPU, row by row (the cache-friendly order).
// This plain loop is the CPU baseline: one core, one pixel at a time. The compiler does not
// vectorise it, because it cannot rule out that in and out overlap in memory (noinline keeps it a
// separate function, so that this holds wherever it is called from).
__attribute__((noinline)) static void sobel_cpu(const std::vector<uchar> &in, std::vector<uchar> &out,
                      size_t width, size_t height) {
  static const int gX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  static const int gY[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
  for (size_t y = 1; y < height - 1; ++y)
    for (size_t x = 1; x < width - 1; ++x) {
      int magX = 0, magY = 0;
      for (size_t k = 0; k < 3; ++k)
        for (size_t l = 0; l < 3; ++l) {
          int p = in[(x + k - 1) + (y + l - 1) * width];
          magX += gX[k * 3 + l] * p;
          magY += gY[k * 3 + l] * p;
        }
      out[x + y * width] = uchar(std::min(std::abs(magX) + std::abs(magY), 0xFF));
    }
}

// Best time of a few CPU runs, in milliseconds
static double time_cpu(const std::vector<uchar> &in, std::vector<uchar> &out,
                       size_t width, size_t height) {
  double best = 1e30;
  for (int r = 0; r < 5; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    sobel_cpu(in, out, width, height);
    auto t1 = std::chrono::steady_clock::now();
    best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return best;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "usage: " << argv[0] << " <image.bmp> [crop]\n";
    return 2;
  }
  std::vector<uchar> full;
  size_t fw, fh;
  if (!read_bmp_gray(argv[1], full, fw, fh)) {
    std::cout << "cannot read " << argv[1] << " (24-bit BMP expected)\n";
    return 2;
  }

  // The part of the image filtered on the FPGA: the full image, or a central crop
  size_t width = fw, height = fh;
  std::vector<uchar> input = full;
  if (argc > 2) {
    size_t c = std::strtoul(argv[2], nullptr, 10);
    width = std::clamp<size_t>(c, 3, fw);
    height = std::clamp<size_t>(c, 3, fh);
    size_t x0 = (fw - width) / 2, y0 = (fh - height) / 2;
    input.resize(width * height);
    for (size_t y = 0; y < height; ++y)
      for (size_t x = 0; x < width; ++x)
        input[y * width + x] = full[(y0 + y) * fw + x0 + x];
  }
#ifdef CPU_ONLY
  // CPU-only build: the filter on the full picture, no FPGA part
  std::cout << "Image " << argv[1] << ": " << fw << " x " << fh << " = " << fw * fh << " pixels\n";
  std::vector<uchar> cpu_full(fw * fh, 0);
  std::cout << "CPU, one core: " << time_cpu(full, cpu_full, fw, fh) << " ms for the full image\n";
  write_bmp_gray("output_cpu.bmp", cpu_full, fw, fh);
  std::cout << "Wrote output_cpu.bmp; CPU-only build, no FPGA part\n";
  return 0;
#else
  const size_t area = height * width;
  std::cout << "Image " << argv[1] << ": " << fw << " x " << fh << " pixels; FPGA part: "
            << width << " x " << height << " = " << area << " pixels\n";

  // CPU: the same filter on the FPGA part and on the full image
  std::vector<uchar> cpu_out(area, 0), cpu_full(fw * fh, 0);
  double cpu_ms = time_cpu(input, cpu_out, width, height);
  double cpu_full_ms = time_cpu(full, cpu_full, fw, fh);
  std::cout << "CPU, one core: " << cpu_ms << " ms for " << width << " x " << height
            << ", " << cpu_full_ms << " ms for the full image " << fw << " x " << fh << "\n";

  // FPGA
  sycl::queue q;
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
  sycl::buffer<uchar> ib{input.data(), sycl::range<1>{area}};
  std::vector<uchar> fpga_out(area, 0);
  sycl::buffer<uchar> ob{fpga_out.data(), sycl::range<1>{area}};

  auto t0 = std::chrono::steady_clock::now();
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor pixel_rb{ib, cgh, sycl::read_only};
    sycl::accessor pixel_wb{ob, cgh, sycl::write_only};
    cgh.single_task<class krnl_sobel>([=] {
      // Partition completely the following arrays along their first dimension
      auto gX = sycl::ext::xilinx::partition_array<
          char, 9, sycl::ext::xilinx::partition::complete<1>>(
          {-1, 0, 1, -2, 0, 2, -1, 0, 1});
      auto gY = sycl::ext::xilinx::partition_array<
          char, 9, sycl::ext::xilinx::partition::complete<1>>(
          {1, 2, 1, 0, 0, 0, -1, -2, -1});
      int magX, magY, gI, pIndex;
      for (size_t x = 1; x < width - 1; ++x) {
        for (size_t y = 1; y < height - 1; ++y) {
          magX = 0; magY = 0;
          sycl::ext::xilinx::pipeline([&] {
            for (size_t k = 0; k < 3; ++k) {
              for (size_t l = 0; l < 3; ++l) {
                gI = k * 3 + l;
                pIndex = (x + k - 1) + (y + l - 1) * width;
                magX += gX[gI] * pixel_rb[pIndex];
                magY += gY[gI] * pixel_rb[pIndex];
              }
            }
          });
          pixel_wb[x + y * width] = sycl::min((int)(sycl::abs(magX) + sycl::abs(magY)), 0xFF);
        }
      }
    });
  });
  q.wait();
  auto t1 = std::chrono::steady_clock::now();
  double host_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "FPGA kernel: " << host_ms << " ms measured on the host for "
            << (width - 2) * (height - 2) << " filtered pixels\n";
  if (std::getenv("XCL_EMULATION_MODE"))
    std::cout << "(emulation: this is how long the simulator needed, not the time the FPGA needs)\n";
  {
    sycl::host_accessor r{ob, sycl::read_only};  // copy the result back to fpga_out
  }

  // Compare the FPGA result with the CPU result (the border pixels stay 0 in both)
  size_t errors = 0;
  for (size_t i = 0; i < area; ++i)
    errors += fpga_out[i] != cpu_out[i];
  write_bmp_gray("output_fpga.bmp", fpga_out, width, height);
  write_bmp_gray("output_cpu.bmp", cpu_full, fw, fh);
  std::cout << "Wrote output_fpga.bmp (" << width << " x " << height << ") and output_cpu.bmp ("
            << fw << " x " << fh << ")\n";
  std::cout << (errors ? "FAILED: " : "PASSED: ") << errors << " errors in " << area << " values\n";
  return errors ? 1 : 0;
#endif
}
