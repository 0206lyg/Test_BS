#ifndef SLURRY_FFT_PLANS_H
#define SLURRY_FFT_PLANS_H

// This header is included outside SlurryPeriodic: FFTW declares a C API.
#ifdef SLURRY_HAVE_FFTW
#include <fftw3.h>
#include <array>
#include <complex>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace SlurryFFT {
// FFTW execution is thread safe; planning, initialization and destruction are
// serialized. No cleanup_threads() call is made: another operator or plugin
// may still own plans. Each operator owns its plans and mutable grid buffers.
inline std::mutex &planner_mutex() {
  static std::mutex mutex;
  return mutex;
}

class Plans {
public:
  Plans() = default;
  Plans(const Plans &) = delete;
  Plans &operator=(const Plans &) = delete;
  ~Plans() { clear(); }

  void clear() {
    std::lock_guard<std::mutex> lock(planner_mutex());
    release();
  }

  void reset(const std::array<int, 3> &dimensions, int threads,
             std::array<std::vector<std::complex<double>>, 2> &grids) {
    std::array<void *, 2> addresses{{grids[0].data(), grids[1].data()}};
    if (plans_[0] && dimensions_ == dimensions && threads_ == threads &&
        addresses_ == addresses)
      return;
    std::lock_guard<std::mutex> lock(planner_mutex());
    release();
    // The C++ complex representation is the interleaved real/imaginary layout
    // supported by FFTW. Plans execute on the exact arrays used for planning.
    static_assert(sizeof(std::complex<double>) == sizeof(fftw_complex),
                  "FFTW requires interleaved complex storage");
    static const bool initialized = fftw_init_threads() != 0;
    if (!initialized)
      throw std::runtime_error("FFTW thread initialization failed");
    fftw_plan_with_nthreads(threads);
    for (int component = 0; component < 2; ++component)
      for (int inverse = 0; inverse < 2; ++inverse) {
        auto *data = reinterpret_cast<fftw_complex *>(grids[component].data());
        plans_[2 * component + inverse] = fftw_plan_dft_3d(
            dimensions[2], dimensions[1], dimensions[0], data, data,
            inverse ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_MEASURE);
        if (!plans_[2 * component + inverse]) {
          release();
          throw std::runtime_error("FFTW 3D plan creation failed");
        }
      }
    dimensions_ = dimensions;
    addresses_ = addresses;
    threads_ = threads;
    ++builds_;
  }

  void execute(bool inverse) const {
    if (!plans_[0])
      throw std::logic_error("FFTW execution requires an initialized plan");
    // Both transforms are unnormalised, matching the physical Fourier weights.
    fftw_execute(plans_[inverse ? 1 : 0]);
    fftw_execute(plans_[inverse ? 3 : 2]);
  }

  std::size_t builds() const { return builds_; }

private:
  void release() {
    for (auto &plan : plans_) {
      if (plan) fftw_destroy_plan(plan);
      plan = nullptr;
    }
    addresses_ = {{nullptr, nullptr}};
    dimensions_ = {{0, 0, 0}};
    threads_ = 0;
  }
  std::array<fftw_plan, 4> plans_{{nullptr, nullptr, nullptr, nullptr}};
  std::array<void *, 2> addresses_{{nullptr, nullptr}};
  std::array<int, 3> dimensions_{{0, 0, 0}};
  int threads_ = 0;
  std::size_t builds_ = 0;
};
} // namespace SlurryFFT
#endif
#endif
