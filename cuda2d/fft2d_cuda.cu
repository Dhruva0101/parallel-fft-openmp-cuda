/*
 * fft2d_cuda.cu
 * CUDA 2D FFT: row FFT -> transpose -> row FFT -> transpose back.
 *
 * Row FFT kernel: one thread block per row, row held in shared memory (D6).
 * Transpose kernel: 32x32 tiles in shared memory, padded to 33 columns (D7).
 * Returns the kernel-only time measured with CUDA events.
 *
 * Author: Dhruva Kamble
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

#define TILE 32
#define TILE_ROWS 8

extern "C" double fft2d_cuda(cpx *x, int n) {
    (void)x; (void)n;
    /* TODO: implement */
    return -1.0;
}
