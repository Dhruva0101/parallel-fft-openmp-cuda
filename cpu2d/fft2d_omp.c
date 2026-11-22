/*
 * fft2d_omp.c
 * OpenMP parallel 2D FFT.
 *
 * 1. Parallel FFTs across rows.
 * 2. Synchronize after row phase.
 * 3. Parallel FFTs across columns.
 * 4. Each thread owns a private column buffer.
 *
 * Author: Om Kumar Singh
 */

#include "fft_common.h"
#include "fft2d.h"

#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Cached twiddle factors. */
static cpx *tw_cache = NULL;
static int tw_n = 0;

static const cpx *get_twiddles(int n) {
    if (n != tw_n) {
        free(tw_cache);

        tw_cache = malloc(sizeof(cpx) * (size_t)(n / 2));

        if (!tw_cache) {
            fprintf(stderr, "fft2d_omp: out of memory\n");
            exit(1);
        }

        make_twiddles(tw_cache, (size_t)n);
        tw_n = n;
    }

    return tw_cache;
}

/* Sequential FFT helper used independently by each thread. */
static void fft1d(cpx *x, int n, const cpx *tw) {
    const int logn = ilog2((size_t)n);

    for (int i = 0; i < n; i++) {
        int j = (int)bit_reverse((uint32_t)i, logn);

        if (i < j) {
            cpx temp = x[i];
            x[i] = x[j];
            x[j] = temp;
        }
    }

    for (int s = 1; s <= logn; s++) {
        const int m = 1 << s;
        const int half = m >> 1;
        const int stride = n / m;

        for (int k = 0; k < n; k += m) {
            for (int j = 0; j < half; j++) {
                cpx w = tw[j * stride];

                cpx t = cpx_mul(w, x[k + j + half]);
                cpx u = x[k + j];

                x[k + j] = cpx_add(u, t);
                x[k + j + half] = cpx_sub(u, t);
            }
        }
    }
}

double fft2d_omp(cpx *x, int n) {
    const cpx *tw = get_twiddles(n);

    #pragma omp parallel
    {
        /*
         * Phase 1: Parallel row FFTs.
         *
         * Each row occupies a disjoint memory region.
         */
        #pragma omp for schedule(static)
        for (int r = 0; r < n; r++) {
            fft1d(&x[(size_t)r * n], n, tw);
        }

        /*
         * Implicit barrier:
         * All row FFTs finish before column processing.
         */

        /* Each thread gets its own column buffer. */
        cpx *column = malloc(sizeof(cpx) * (size_t)n);

        if (!column) {
            fprintf(stderr, "fft2d_omp: out of memory\n");
            exit(1);
        }

        /*
         * Phase 2: Parallel column FFTs.
         */
        #pragma omp for schedule(static)
        for (int c = 0; c < n; c++) {

            /* Gather column. */
            for (int r = 0; r < n; r++) {
                column[r] = x[(size_t)r * n + c];
            }

            /* Transform column. */
            fft1d(column, n, tw);

            /* Scatter column. */
            for (int r = 0; r < n; r++) {
                x[(size_t)r * n + c] = column[r];
            }
        }

        free(column);
    }

    return -1.0;
}