/*
 * fft_common.h: shared types and utilities used by all six implementations.
 *
 * Why a shared module:
 *   For speedup numbers to be meaningful, every implementation must
 *     (1) use the same data type,
 *     (2) receive bit-identical input,
 *     (3) be timed the same way, and
 *     (4) be judged correct by the same rule.
 *   Putting those four things in one place means no implementation can
 *   accidentally get an unfair advantage. See docs/DESIGN.md (D1, D2, D9, D10).
 */
#ifndef FFT_COMMON_H
#define FFT_COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {   /* lets the CUDA (.cu, compiled as C++) drivers link against this C code */
#endif

/* ------------------------------------------------------------------------- *
 * Complex type
 * ------------------------------------------------------------------------- *
 * A plain struct of two floats (8 bytes), laid out exactly like CUDA's
 * cuFloatComplex / float2. That means a host cpx* can be cudaMemcpy'd to the
 * GPU and reinterpreted as float2* with no conversion.
 *
 * We deliberately do NOT use C99 `float complex`: it does not exist in CUDA
 * device code, and we want identical arithmetic on CPU and GPU (D1).
 */
typedef struct {
    float re;
    float im;
} cpx;

/* Small inline helpers. `static inline` = each file gets its own copy, no link
 * conflicts, and the compiler can inline them into the butterfly loop. */
static inline cpx cpx_make(float re, float im) { cpx z = {re, im}; return z; }
static inline cpx cpx_add(cpx a, cpx b) { return cpx_make(a.re + b.re, a.im + b.im); }
static inline cpx cpx_sub(cpx a, cpx b) { return cpx_make(a.re - b.re, a.im - b.im); }
/* (a+bi)(c+di) = (ac - bd) + (ad + bc)i */
static inline cpx cpx_mul(cpx a, cpx b) {
    return cpx_make(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
static inline cpx cpx_conj(cpx a) { return cpx_make(a.re, -a.im); }

/* ------------------------------------------------------------------------- *
 * Integer helpers
 * ------------------------------------------------------------------------- */
int      is_pow2(size_t n);            /* 1 if n is a power of two (n > 0)        */
int      ilog2(size_t n);              /* log2(n) for a power of two              */
uint32_t bit_reverse(uint32_t x, int bits); /* reverse the low `bits` bits of x   */

/* ------------------------------------------------------------------------- *
 * Input generation (D2)
 * ------------------------------------------------------------------------- *
 * Fills buf[0..n) with pseudo-random complex values in [-1, 1).
 * Same (n, seed) -> same data on every machine, every run, every implementation.
 * For 2D, the same buffer is simply interpreted as a side x side row-major matrix.
 */
#define FFT_DEFAULT_SEED 548u
void gen_input(cpx *buf, size_t n, uint32_t seed);

/* ------------------------------------------------------------------------- *
 * Twiddle factors (D3)
 * ------------------------------------------------------------------------- *
 * tw[k] = exp(-2*pi*i*k / n) for k = 0 .. n/2 - 1, computed in double and
 * rounded to float ONCE. A butterfly in a stage with group size m uses
 *     W_m^j = exp(-2*pi*i*j/m) = tw[j * (n/m)]
 * so a single table of length n/2 serves every stage.
 * Caller allocates n/2 cpx.
 */
void make_twiddles(cpx *tw, size_t n);

/* ------------------------------------------------------------------------- *
 * Reference transforms (D10): slow, obviously correct, double precision
 * ------------------------------------------------------------------------- *
 * O(N^2) direct DFT straight from the definition:
 *     X[k] = sum_{t=0}^{N-1} x[t] * exp(-2*pi*i*k*t/N)
 * Only used for correctness checks on small sizes. Never timed.
 */
void naive_dft_1d(const cpx *in, cpx *out, size_t n);
/* 2D DFT of a rows x cols row-major matrix (computed as 1D DFT on rows then cols). */
void naive_dft_2d(const cpx *in, cpx *out, size_t rows, size_t cols);

/* ------------------------------------------------------------------------- *
 * Error metric (D10)
 * ------------------------------------------------------------------------- *
 * Normalised max error:   max_k |got[k] - ref[k]|  /  max_k |ref[k]|
 *
 * Why not per-element relative error? Some FFT outputs are legitimately ~0,
 * and dividing by ~0 makes a perfectly good result look wrong. Normalising
 * by the largest output magnitude measures error relative to the signal scale.
 */
double max_norm_error(const cpx *ref, const cpx *got, size_t n);

/* Float FFT error grows roughly like eps * log2(N); 1e-4 leaves a wide margin
 * even at N = 2^24, while still catching any real bug (bugs give errors ~1). */
#define FFT_TOLERANCE 1e-4

/* ------------------------------------------------------------------------- *
 * Timing (D9)
 * ------------------------------------------------------------------------- *
 * Wall-clock milliseconds from a monotonic clock. NOT clock(), which measures
 * CPU time summed over all threads and would make OpenMP look *slower*.
 */
double now_ms(void);

#ifdef __cplusplus
}
#endif

#endif /* FFT_COMMON_H */
