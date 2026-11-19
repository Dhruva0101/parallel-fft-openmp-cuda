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

#include <stdlib.h>

/* Twiddle table cache (D3): rebuilt only when n changes, so the cost lands in
 * the warm-up run (run 0) and never in the timed runs. */
static cpx *tw_cache = NULL;
static int  tw_n = 0;

static const cpx *get_twiddles(int n) {
    if (n != tw_n) {
        free(tw_cache);
        tw_cache = malloc(sizeof(cpx) * (size_t)(n / 2));
        if (!tw_cache) { fprintf(stderr, "fft1d_seq: out of memory\n"); exit(1); }
        make_twiddles(tw_cache, (size_t)n);
        tw_n = n;
    }
    return tw_cache;
}

double fft1d_seq(cpx *x, int n) {
    const cpx *tw = get_twiddles(n);
    const int logn = ilog2((size_t)n);

    /* Step 1: bit-reversal permutation. i < j so each pair is swapped once. */
    for (int i = 0; i < n; i++) {
        int j = (int)bit_reverse((uint32_t)i, logn);
        if (i < j) { cpx t = x[i]; x[i] = x[j]; x[j] = t; }
    }

    /* Step 2: log2(n) stages. Stage s merges sub-FFTs of size half into size m. */
    for (int s = 1; s <= logn; s++) {
        const int m = 1 << s;
        const int half = m >> 1;
        const int stride = n / m;          /* W_m^j = tw[j * stride] */
        for (int k = 0; k < n; k += m) {
            for (int j = 0; j < half; j++) {
                cpx w = tw[j * stride];
                cpx t = cpx_mul(w, x[k + j + half]);
                cpx u = x[k + j];
                x[k + j]        = cpx_add(u, t);
                x[k + j + half] = cpx_sub(u, t);
            }
        }
    }
    return -1.0;   /* CPU: no separate kernel time */
}
