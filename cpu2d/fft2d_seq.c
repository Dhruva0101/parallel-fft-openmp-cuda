/*
 * fft2d_seq.c
 * Sequential 2D FFT using row-column decomposition. Baseline for 2D speedups.
 *
 * 1D FFT on every row, then on every column. Columns are gathered into a
 * contiguous buffer before the FFT and scattered back afterwards.
 *
 * Author: Om Kumar Singh
 */
#include "fft_common.h"
#include "fft2d.h"

double fft2d_seq(cpx *x, int n) {
    (void)x; (void)n;
    /* TODO: implement */
    return -1.0;
}
