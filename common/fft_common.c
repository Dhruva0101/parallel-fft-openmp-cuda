/*
 * fft_common.c: implementation of the shared utilities. See fft_common.h.
 */
#define _POSIX_C_SOURCE 199309L   /* for clock_gettime on Linux */
#include "fft_common.h"

#include <math.h>
#include <stdlib.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int is_pow2(size_t n) { return n > 0 && (n & (n - 1)) == 0; }

int ilog2(size_t n) {
    int k = 0;
    while (((size_t)1 << k) < n) k++;
    return k;
}

/* Reverse the lowest `bits` bits of x.
 * Example with bits = 3:  x = 1 (001) -> 4 (100),  x = 3 (011) -> 6 (110).
 * Simple loop version, chosen for clarity over speed. It is also O(N log N) total, which
 * is the same order as the FFT itself. GPU code uses __brev() instead.        */
uint32_t bit_reverse(uint32_t x, int bits) {
    uint32_t r = 0;
    for (int i = 0; i < bits; i++) {
        r = (r << 1) | (x & 1u);
        x >>= 1;
    }
    return r;
}

/* xorshift32: tiny, fast, fully deterministic PRNG. We don't use rand()
 * because its sequence differs between glibc (ARC) and macOS libc,
 * which would give different inputs on different machines. */
static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *state = x;
}

void gen_input(cpx *buf, size_t n, uint32_t seed) {
    uint32_t s = seed ? seed : 1u;          /* xorshift state must be non-zero */
    for (size_t i = 0; i < n; i++) {
        /* top 24 bits -> [0,1) exactly representable in float, then -> [-1,1) */
        float re = (float)(xorshift32(&s) >> 8) / 16777216.0f * 2.0f - 1.0f;
        float im = (float)(xorshift32(&s) >> 8) / 16777216.0f * 2.0f - 1.0f;
        buf[i] = cpx_make(re, im);
    }
}

void make_twiddles(cpx *tw, size_t n) {
    for (size_t k = 0; k < n / 2; k++) {
        double angle = -2.0 * M_PI * (double)k / (double)n;  /* computed in double */
        tw[k] = cpx_make((float)cos(angle), (float)sin(angle));
    }
}

/* Direct DFT with double-precision accumulation, so the reference is
 * strictly more accurate than any float FFT we compare against it. */
void naive_dft_1d(const cpx *in, cpx *out, size_t n) {
    for (size_t k = 0; k < n; k++) {
        double sr = 0.0, si = 0.0;
        for (size_t t = 0; t < n; t++) {
            /* (k*t) mod n keeps the angle small -> better precision for big n */
            double angle = -2.0 * M_PI * (double)((k * t) % n) / (double)n;
            double c = cos(angle), s = sin(angle);
            sr += in[t].re * c - in[t].im * s;
            si += in[t].re * s + in[t].im * c;
        }
        out[k] = cpx_make((float)sr, (float)si);
    }
}

void naive_dft_2d(const cpx *in, cpx *out, size_t rows, size_t cols) {
    /* Separability: 2D DFT = 1D DFT on every row, then on every column. */
    size_t big = rows > cols ? rows : cols;
    cpx *tmp = malloc(sizeof(cpx) * rows * cols);
    cpx *a = calloc(big, sizeof(cpx));   /* calloc: silences a gcc false-positive */
    cpx *b = malloc(sizeof(cpx) * big);
    if (!tmp || !a || !b) { fprintf(stderr, "naive_dft_2d: out of memory\n"); exit(1); }

    for (size_t r = 0; r < rows; r++)
        naive_dft_1d(&in[r * cols], &tmp[r * cols], cols);

    for (size_t c = 0; c < cols; c++) {
        for (size_t r = 0; r < rows; r++) a[r] = tmp[r * cols + c];   /* gather column  */
        naive_dft_1d(a, b, rows);
        for (size_t r = 0; r < rows; r++) out[r * cols + c] = b[r];   /* scatter back   */
    }
    free(tmp); free(a); free(b);
}

double max_norm_error(const cpx *ref, const cpx *got, size_t n) {
    double max_diff = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < n; i++) {
        double dr = (double)got[i].re - ref[i].re;
        double di = (double)got[i].im - ref[i].im;
        double d = sqrt(dr * dr + di * di);
        double m = sqrt((double)ref[i].re * ref[i].re + (double)ref[i].im * ref[i].im);
        if (d != d) return INFINITY;               /* NaN anywhere = failure */
        if (d > max_diff) max_diff = d;
        if (m > max_ref) max_ref = m;
    }
    return max_ref > 0.0 ? max_diff / max_ref : max_diff;
}

double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);   /* monotonic: immune to NTP clock jumps */
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}
