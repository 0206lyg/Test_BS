
#include <stdint.h>
#include <math.h>
extern void dgemm_(const char *, const char *, const int32_t *, const int32_t *,
                  const int32_t *, const double *, const double *, const int32_t *,
                  const double *, const int32_t *, const double *, double *, const int32_t *);
extern void dgesv_(const int32_t *, const int32_t *, double *, const int32_t *,
                  int32_t *, double *, const int32_t *, int32_t *);
int reference_blas_probe(void) {
    /* The zero upper words allow an ILP64 routine to execute the tiny problem;
       guarded output integers then detect an incompatible 64-bit write. */
    struct { int32_t value, upper; } two = {2, 0}, one = {1, 0};
    struct { int32_t value, guard; } info = {-99, 0x13579bdf};
    struct { int32_t value[2]; int32_t guard[2]; } piv = {{0, 0}, {0x13579bdf, 0x2468ace}};
    char normal = 'N';
    double alpha = 1., beta = 0.;
    double a[4] = {1., 3., 2., 4.}, identity[4] = {1., 0., 0., 1.}, c[4] = {0.};
    dgemm_(&normal, &normal, &two.value, &two.value, &two.value, &alpha,
           a, &two.value, identity, &two.value, &beta, c, &two.value);
    for (int i = 0; i < 4; ++i) if (!isfinite(c[i]) || fabs(c[i] - a[i]) > 1e-12) return 1;
    double matrix[4] = {3., 1., 1., 2.}, rhs[2] = {9., 8.};
    dgesv_(&two.value, &one.value, matrix, &two.value, piv.value, rhs, &two.value, &info.value);
    if (info.guard != 0x13579bdf || piv.guard[0] != 0x13579bdf || piv.guard[1] != 0x2468ace) return 2;
    if (info.value != 0 || piv.value[0] < 1 || piv.value[0] > 2 || piv.value[1] < 1 || piv.value[1] > 2) return 3;
    if (!isfinite(rhs[0]) || !isfinite(rhs[1]) || fabs(rhs[0] - 2.) > 1e-12 || fabs(rhs[1] - 3.) > 1e-12) return 4;
    return 0;
}
