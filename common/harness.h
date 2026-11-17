/*
 * harness.h: one shared driver for timing and verifying every implementation.
 *
 * The idea: each implementation folder only writes the FFT itself plus a
 * 10-line main() that lists its implementations. Argument parsing, input
 * generation, verification, timing and CSV output all live here, so they are
 * guaranteed identical across all six versions (D9, D10, D11).
 */
#ifndef HARNESS_H
#define HARNESS_H

#include "fft_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Signature every implementation must follow:
 *
 *   data : in/out buffer on the HOST, transformed in place.
 *   n    : 1D -> sequence length N
 *          2D -> matrix side n (the buffer holds n*n elements, row-major)
 *
 *   return value: "kernel-only" milliseconds if the implementation measures
 *                 it itself (CUDA: time between cudaEvents around the kernels,
 *                 excluding host<->device copies), or -1.0 if not applicable (CPU).
 *
 * The harness ALWAYS measures the whole call with now_ms() -> "total_ms".
 * For CUDA, total_ms therefore includes the copies = end-to-end time.
 */
typedef double (*fft_fn)(cpx *data, int n);

typedef struct {
    const char *name;   /* e.g. "seq", "omp", "cuda"; written to the CSV */
    fft_fn      fn;
} fft_impl;

/*
 * Usage (from the generated binary):
 *   ./bin/cpu1d --impl omp --size 1048576 --reps 5        # benchmark -> CSV on stdout
 *   ./bin/cpu1d --impl seq --size 1024 --check            # correctness only
 *   ./bin/cpu1d --header                                  # print CSV header
 *
 * Exit code is non-zero if --check fails, so `make check` and CI can rely on it.
 */
int harness_main(int argc, char **argv, int dim, const fft_impl *impls, int nimpls);

#ifdef __cplusplus
}
#endif

#endif /* HARNESS_H */
