/*
 * fft1d_cuda.cu
 * CUDA radix-2 FFT (1D).
 *
 * Bit-reversal kernel, then one kernel launch per stage with one thread per
 * butterfly. The boundary between launches acts as a grid-wide barrier (D5).
 * Returns the kernel-only time measured with CUDA events.
 *
 * Author: Om Kumar Singh
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "fft_common.h"

#define CUDA_CHECK(call) do {                                              \
    cudaError_t err_ = (call);                                             \
    if (err_ != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                cudaGetErrorString(err_), __FILE__, __LINE__);             \
        exit(1);                                                           \
    }                                                                      \
} while (0)

extern "C" double fft1d_cuda(cpx *x, int n) {
    (void)x; (void)n;
    /* TODO: implement */
    return -1.0;
}
