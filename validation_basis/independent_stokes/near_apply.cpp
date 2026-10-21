// Exact application of stored six-entry symmetric Stokes quadrature blocks.
// No quadrature, jump term, or physical-model change is made here.
#include <cstdint>
#include <limits>
#include <omp.h>

extern "C" int spheroid_near_apply_abi() { return 1; }

// weights[6*(target*n + source) + k], k=(xx,xy,xz,yy,yz,zz).
// vectors[(3*source + component)*nrhs + rhs]; output uses the same ordering.
// Targets are independent. Every target retains increasing source order,
// regardless of OpenMP thread count. Compile without -ffast-math.
extern "C" int spheroid_near_apply(
    std::int64_t n, std::int64_t nrhs, const double *weights,
    const double *vectors, double *output, int threads) {
  if (n < 1 || nrhs < 1 || !weights || !vectors || !output || threads < 0)
    return 1;
  if (n > std::numeric_limits<std::int64_t>::max() / n / 6 ||
      n > std::numeric_limits<std::int64_t>::max() / nrhs / 3)
    return 2;
  const int thread_count = threads > 0 ? threads : omp_get_max_threads();
  // A fixed RHS has three scalar accumulators, which the compiler keeps in
  // registers. The single-RHS path makes exactly one sequential weights pass.
  if (nrhs == 1) {
#pragma omp parallel for schedule(static) num_threads(thread_count)
    for (std::int64_t t = 0; t < n; ++t) {
      double x = 0., y = 0., z = 0.;
      const double *w = weights + 6 * t * n;
      for (std::int64_t s = 0; s < n; ++s, w += 6) {
        const double qx = vectors[3*s], qy = vectors[3*s+1], qz = vectors[3*s+2];
        x += w[0]*qx + w[1]*qy + w[2]*qz;
        y += w[1]*qx + w[3]*qy + w[4]*qz;
        z += w[2]*qx + w[4]*qy + w[5]*qz;
      }
      output[3*t] = x;
      output[3*t+1] = y;
      output[3*t+2] = z;
    }
  } else {
#pragma omp parallel for schedule(static) num_threads(thread_count)
    for (std::int64_t t = 0; t < n; ++t) {
      double *u = output + 3*t*nrhs;
      for (std::int64_t r = 0; r < 3*nrhs; ++r) u[r] = 0.;
      const double *w = weights + 6*t*n;
      for (std::int64_t s = 0; s < n; ++s, w += 6) {
        const double *q = vectors + 3*s*nrhs;
        for (std::int64_t r = 0; r < nrhs; ++r) {
          const double qx = q[r], qy = q[nrhs+r], qz = q[2*nrhs+r];
          u[r] += w[0]*qx + w[1]*qy + w[2]*qz;
          u[nrhs+r] += w[1]*qx + w[3]*qy + w[4]*qz;
          u[2*nrhs+r] += w[2]*qx + w[4]*qy + w[5]*qz;
        }
      }
    }
  }
  return 0;
}
