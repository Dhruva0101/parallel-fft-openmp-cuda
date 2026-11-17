/*
 * harness.c: shared benchmark and verification driver. See harness.h.
 */
#include "harness.h"

#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Above this many elements the O(N^2) naive DFT gets too slow, so we rely on
 * the round-trip check alone. 4096^2 = 16.7M inner iterations: ~0.1 s. */
#define NAIVE_CHECK_MAX 4096

static const char *CSV_HEADER =
    "impl,dim,size,elements,threads,run,total_ms,kernel_ms";

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [--impl NAME] [--size N] [--reps R] [--seed S] [--check] [--header]\n"
        "  --impl   which implementation to run (default: all in this binary)\n"
        "  --size   1D: length N, 2D: matrix side n (power of two, default 1024)\n"
        "  --reps   timed repetitions; run 0 is a warm-up (default 5)\n"
        "  --check  verify correctness instead of benchmarking\n"
        "  --header print the CSV header and exit\n", prog);
}

static int threads_in_use(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/*
 * Round-trip check using the conjugation identity
 *     IFFT(y) = conj( FFT( conj(y) ) ) / N
 * so we can test the inverse using only the forward implementation.
 * If FFT is correct, IFFT(FFT(x)) must give back x. Works at ANY size,
 * no O(N^2) reference needed. This is what lets us verify N = 2^24.
 */
static double round_trip_error(const fft_impl *impl, const cpx *x, size_t elems, int n) {
    cpx *y = malloc(sizeof(cpx) * elems);
    if (!y) { fprintf(stderr, "out of memory\n"); exit(1); }
    memcpy(y, x, sizeof(cpx) * elems);

    impl->fn(y, n);                                         /* y = FFT(x)        */
    for (size_t i = 0; i < elems; i++) y[i] = cpx_conj(y[i]);
    impl->fn(y, n);                                         /* y = FFT(conj(y))  */
    for (size_t i = 0; i < elems; i++) {                    /* conj and / N      */
        y[i] = cpx_conj(y[i]);
        y[i].re /= (float)elems;
        y[i].im /= (float)elems;
    }
    double err = max_norm_error(x, y, elems);
    free(y);
    return err;
}

static int run_check(const fft_impl *impl, int dim, int n, size_t elems, uint32_t seed) {
    cpx *x = malloc(sizeof(cpx) * elems);
    cpx *got = malloc(sizeof(cpx) * elems);
    if (!x || !got) { fprintf(stderr, "out of memory\n"); exit(1); }
    gen_input(x, elems, seed);
    int ok = 1;

    /* 1) Compare against the slow-but-obviously-correct DFT (small sizes only). */
    if (elems <= NAIVE_CHECK_MAX) {
        cpx *ref = malloc(sizeof(cpx) * elems);
        if (!ref) { fprintf(stderr, "out of memory\n"); exit(1); }
        if (dim == 1) naive_dft_1d(x, ref, elems);
        else          naive_dft_2d(x, ref, (size_t)n, (size_t)n);
        memcpy(got, x, sizeof(cpx) * elems);
        impl->fn(got, n);
        double e = max_norm_error(ref, got, elems);
        int pass = e < FFT_TOLERANCE;
        ok &= pass;
        printf("[%s] %-5s dim=%d size=%-8d vs naive DFT   err=%.3e  %s\n",
               pass ? "PASS" : "FAIL", impl->name, dim, n, e, pass ? "" : "<-- too large");
        free(ref);
    }

    /* 2) Round trip: IFFT(FFT(x)) == x (any size). */
    double e = round_trip_error(impl, x, elems, n);
    int pass = e < FFT_TOLERANCE;
    ok &= pass;
    printf("[%s] %-5s dim=%d size=%-8d round trip     err=%.3e  %s\n",
           pass ? "PASS" : "FAIL", impl->name, dim, n, e, pass ? "" : "<-- too large");

    free(x); free(got);
    return ok;
}

static void run_bench(const fft_impl *impl, int dim, int n, size_t elems,
                      int reps, uint32_t seed) {
    cpx *pristine = malloc(sizeof(cpx) * elems);
    cpx *work = malloc(sizeof(cpx) * elems);
    if (!pristine || !work) { fprintf(stderr, "out of memory\n"); exit(1); }
    gen_input(pristine, elems, seed);

    /* run 0 is a warm-up: page faults, OpenMP thread-pool creation, CUDA
     * context creation all land here. We still print it (marked run=0) so it's
     * visible, and the analysis script drops it. */
    for (int r = 0; r <= reps; r++) {
        memcpy(work, pristine, sizeof(cpx) * elems);   /* fresh input, NOT timed */
        double t0 = now_ms();
        double kernel_ms = impl->fn(work, n);
        double total_ms = now_ms() - t0;
        printf("%s,%d,%d,%zu,%d,%d,%.6f,%.6f\n",
               impl->name, dim, n, elems, threads_in_use(), r, total_ms, kernel_ms);
        fflush(stdout);
    }
    free(pristine); free(work);
}

int harness_main(int argc, char **argv, int dim, const fft_impl *impls, int nimpls) {
    const char *only = NULL;
    int n = 1024, reps = 5, check = 0;
    uint32_t seed = FFT_DEFAULT_SEED;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--impl")  && i + 1 < argc) only = argv[++i];
        else if (!strcmp(argv[i], "--size")  && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps")  && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed")  && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--check")) check = 1;
        else if (!strcmp(argv[i], "--header")) { puts(CSV_HEADER); return 0; }
        else { usage(argv[0]); return 2; }
    }
    if (n < 2 || !is_pow2((size_t)n)) {
        fprintf(stderr, "error: --size must be a power of two >= 2 (got %d)\n", n);
        return 2;
    }
    size_t elems = (dim == 1) ? (size_t)n : (size_t)n * (size_t)n;

    int found = 0, all_ok = 1;
    for (int k = 0; k < nimpls; k++) {
        if (only && strcmp(only, impls[k].name) != 0) continue;
        found = 1;
        if (check) all_ok &= run_check(&impls[k], dim, n, elems, seed);
        else       run_bench(&impls[k], dim, n, elems, reps, seed);
    }
    if (!found) { fprintf(stderr, "error: no implementation named '%s'\n", only); return 2; }
    return all_ok ? 0 : 1;
}
