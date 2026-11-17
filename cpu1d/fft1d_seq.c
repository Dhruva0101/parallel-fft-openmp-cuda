/*
 * fft1d_seq.c
 * Sequential iterative radix-2 Cooley-Tukey FFT (1D). Baseline for 1D speedups.
 *
 * 1. Bit-reversal permutation of the input.
 * 2. log2(n) butterfly stages; stage s combines pairs of size 2^(s-1) DFTs.
 * Twiddles come from a cached table (see docs/DESIGN.md, D3).
 *
 * Author: Dhruva Kamble
 */
#include "fft_common.h"
#include "fft1d.h"

double fft1d_seq(cpx *x, int n) {
    (void)x; (void)n;
    /* TODO: implement */
    return -1.0;
}
