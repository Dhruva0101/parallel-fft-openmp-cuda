/*
 * fft2d_omp.c
 * OpenMP 2D FFT.
 *
 * Row FFTs are distributed across threads, then column FFTs, with one
 * barrier between the two phases. Each thread uses its own column buffer (D8).
 *
 * Author: Om Kumar Singh
 */
#include "fft_common.h"
#include "fft2d.h"

#ifdef _OPENMP
#include <omp.h>
#endif

double fft2d_omp(cpx *x, int n) {
    (void)x; (void)n;
    /* TODO: implement */
    return -1.0;
}
