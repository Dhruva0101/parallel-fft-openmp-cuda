/*
 * fft1d_omp.c
 * OpenMP radix-2 FFT (1D).
 *
 * Each stage is a flat loop over its n/2 butterflies, split across threads.
 * The implicit barrier of each omp for separates stages. A single parallel
 * region wraps all stages so the thread team is created once (D4).
 *
 * Author: Dhruva Kamble
 */
#include "fft_common.h"
#include "fft1d.h"

#ifdef _OPENMP
#include <omp.h>
#endif

double fft1d_omp(cpx *x, int n) {
    (void)x; (void)n;
    /* TODO: implement */
    return -1.0;
}
