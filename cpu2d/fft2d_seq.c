/*
 * fft2d_seq.c
 * Sequential 2D FFT using row-column decomposition.
 *
 * 1. Perform a 1D FFT on every row.
 * 2. Gather each column into a contiguous buffer.
 * 3. Perform a 1D FFT on the column.
 * 4. Scatter the transformed column back.
 *
 * Author: Om Kumar Singh
 */

#include "fft_common.h"
#include "fft2d.h"

#include <stdlib.h>

/* Twiddle factors are cached between calls. */
static cpx *tw_cache = NULL;
static int tw_n = 0;

static const cpx *get_twiddles(int n) {
    if (n != tw_n) {
        free(tw_cache);

        tw_cache = malloc(sizeof(cpx) * (size_t)(n / 2));

        if (!tw_cache) {
            fprintf(stderr, "fft2d_seq: out of memory\n");
            exit(1);
        }

        make_twiddles(tw_cache, (size_t)n);
        tw_n = n;
    }

    return tw_cache;
}

/* Sequential radix-2 FFT on a contiguous 1D buffer. */
static void fft1d(cpx *x, int n, const cpx *tw) {
    const int logn = ilog2((size_t)n);

    /* Step 1: Bit-reversal permutation. */
    for (int i = 0; i < n; i++) {
        int j = (int)bit_reverse((uint32_t)i, logn);

        if (i < j) {
            cpx temp = x[i];
            x[i] = x[j];
            x[j] = temp;
        }
    }

    /* Step 2: Butterfly stages. */
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

double fft2d_seq(cpx *x, int n) {
    const cpx *tw = get_twiddles(n);

    cpx *column = malloc(sizeof(cpx) * (size_t)n);

    if (!column) {
        fprintf(stderr, "fft2d_seq: out of memory\n");
        exit(1);
    }

    /* Phase 1: FFT across rows. */
    for (int r = 0; r < n; r++) {
        fft1d(&x[(size_t)r * n], n, tw);
    }

    /* Phase 2: FFT across columns. */
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

    return -1.0;
}