/* fft1d.h: CPU 1D implementations. Author: Dhruva Kamble */
#ifndef FFT1D_H
#define FFT1D_H

#include "fft_common.h"

double fft1d_seq(cpx *x, int n);   /* sequential baseline */
double fft1d_omp(cpx *x, int n);   /* OpenMP              */

#endif
