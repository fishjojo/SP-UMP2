#if defined __cplusplus
extern "C" {
#endif
#include <complex.h>

void dgemm_(const char*, const char*,
            const int*, const int*, const int*,
            const double*, const double*, const int*,
            const double*, const int*,
            const double*, double*, const int*);
void zgemm_(const char*, const char*,
            const int*, const int*, const int*,
            const double complex*, const double complex*, const int*,
            const double complex*, const int*,
            const double complex*, double complex*, const int*);

#if defined __cplusplus
} // end extern "C"
#endif

