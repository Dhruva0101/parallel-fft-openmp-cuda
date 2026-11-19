/*
 * fft1d_omp.c
 * OpenMP radix-2 FFT (1D).
 *
 * Each stage is a flat loop over its n/2 butterflies, split across threads.
 * The implicit barrier of each omp for separates stages. A single parallel
 * region wraps all stages so the thread team is created once (D4).
 *
 * Why a flat loop: parallelising over groups gives n/m groups per stage, which
 * drops to 1 in the last stage, so the largest stages would run on one thread.
 * Numbering the butterflies b = 0 .. n/2-1 gives every stage n/2 equal work items.
 *
 * Author: Dhruva Kamble
 */
#include "fft_common.h"
#include "fft1d.h"

#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Same caching rule as the sequential version (D3). Kept separate so the two
 * implementations don't share state. */
static cpx *tw_cache = NULL;
static int  tw_n = 0;

static const cpx *get_twiddles(int n) {
    if (n != tw_n) {
        free(tw_cache);
        tw_cache = malloc(sizeof(cpx) * (size_t)(n / 2));
        if (!tw_cache) { fprintf(stderr, "fft1d_omp: out of memory\n"); exit(1); }
        make_twiddles(tw_cache, (size_t)n);
        tw_n = n;
    }
    return tw_cache;
}

double fft1d_omp(cpx *x, int n) {
    const cpx *tw = get_twiddles(n);
    const int logn = ilog2((size_t)n);
    const int nhalf = n / 2;

    #pragma omp parallel
    {
        /* Bit reversal: each i independent; only the i < j thread swaps a pair. */
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            int j = (int)bit_reverse((uint32_t)i, logn);
            if (i < j) { cpx t = x[i]; x[i] = x[j]; x[j] = t; }
        }
        /* implicit barrier */

        for (int s = 1; s <= logn; s++) {       /* every thread runs this loop */
            const int half = 1 << (s - 1);
            const int stride = n >> s;          /* n / m */
            #pragma omp for schedule(static)
            for (int b = 0; b < nhalf; b++) {   /* ...and splits this one */
                int j = b & (half - 1);              /* position in group: b % half */
                int i = ((b >> (s - 1)) << s) + j;   /* group start + j             */
                cpx w = tw[j * stride];
                cpx t = cpx_mul(w, x[i + half]);
                cpx u = x[i];
                x[i]        = cpx_add(u, t);
                x[i + half] = cpx_sub(u, t);
            }
            /* implicit barrier = stage boundary */
        }
    }
    return -1.0;   /* CPU: no separate kernel time */
}
