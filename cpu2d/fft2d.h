/* fft2d.h: CPU 2D implementations. Author: Om Kumar Singh
 * n = matrix side; buffer holds n*n elements, row-major. */
#ifndef FFT2D_H
#define FFT2D_H

#include "fft_common.h"

double fft2d_seq(cpx *x, int n);   /* sequential baseline */
double fft2d_omp(cpx *x, int n);   /* OpenMP              */

#endif
