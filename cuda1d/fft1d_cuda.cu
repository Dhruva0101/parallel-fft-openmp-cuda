/*
 * fft1d_cuda.cu
 * CUDA radix-2 FFT (1D).
 *
 * 1. Bit-reversal permutation.
 * 2. One CUDA kernel launch per FFT stage.
 * 3. One GPU thread per butterfly.
 * 4. Kernel boundaries provide stage synchronization.
 *
 * Author: Om Kumar Singh
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "fft_common.h"

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err_ = (call);                                     \
    if (err_ != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(err_), __FILE__, __LINE__);     \
        exit(1);                                                   \
    }                                                              \
} while (0)

static_assert(sizeof(cpx) == sizeof(float2),
              "cpx must match float2 layout");

/* Complex arithmetic on GPU. */
__device__ __forceinline__
float2 c_add(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__
float2 c_sub(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__
float2 c_mul(float2 a, float2 b) {
    return make_float2(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

/*
 * Bit-reversal permutation.
 *
 * Each thread reads one input element and writes
 * it to its bit-reversed position.
 *
 * Separate input/output buffers avoid races.
 */
__global__
void bit_reverse_kernel(
    const float2 *input,
    float2 *output,
    int n,
    int logn
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        unsigned int j =
            __brev((unsigned int)i) >> (32 - logn);

        output[j] = input[i];
    }
}

/*
 * One butterfly operation per CUDA thread.
 */
__global__
void butterfly_kernel(
    float2 *data,
    const float2 *tw,
    int n,
    int stage
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= n / 2)
        return;

    int half = 1 << (stage - 1);
    int stride = n >> stage;

    int j = b & (half - 1);
    int i = ((b >> (stage - 1)) << stage) + j;

    float2 w = tw[j * stride];

    float2 u = data[i];
    float2 t = c_mul(w, data[i + half]);

    data[i] = c_add(u, t);
    data[i + half] = c_sub(u, t);
}

/*
 * Persistent device buffers.
 *
 * Reallocated only when FFT size changes.
 */
static float2 *d_input = NULL;
static float2 *d_data = NULL;
static float2 *d_tw = NULL;

static int current_n = 0;

static cudaEvent_t ev_start;
static cudaEvent_t ev_stop;
static int events_ready = 0;

static void prepare(int n) {
    if (!events_ready) {
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));
        events_ready = 1;
    }

    if (n == current_n)
        return;

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_tw));

    size_t bytes = (size_t)n * sizeof(float2);

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    CUDA_CHECK(cudaMalloc(
        &d_tw,
        (size_t)(n / 2) * sizeof(float2)
    ));

    /* Generate the same twiddles as CPU implementations. */
    cpx *tw = (cpx *)malloc(
        (size_t)(n / 2) * sizeof(cpx)
    );

    if (!tw) {
        fprintf(stderr, "fft1d_cuda: out of memory\n");
        exit(1);
    }

    make_twiddles(tw, (size_t)n);

    CUDA_CHECK(cudaMemcpy(
        d_tw,
        tw,
        (size_t)(n / 2) * sizeof(float2),
        cudaMemcpyHostToDevice
    ));

    free(tw);

    current_n = n;
}

extern "C"
double fft1d_cuda(cpx *x, int n) {
    prepare(n);

    const int logn = ilog2((size_t)n);

    const size_t bytes =
        (size_t)n * sizeof(float2);

    const int threads = 256;

    const int blocks_input =
        (n + threads - 1) / threads;

    const int blocks_butterfly =
        (n / 2 + threads - 1) / threads;

    /* Copy input from CPU to GPU. */
    CUDA_CHECK(cudaMemcpy(
        d_input,
        x,
        bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaEventRecord(ev_start));

    /* Step 1: Bit reversal. */
    bit_reverse_kernel<<<blocks_input, threads>>>(
        d_input,
        d_data,
        n,
        logn
    );

    CUDA_CHECK(cudaGetLastError());

    /*
     * Step 2: Butterfly stages.
     *
     * Separate kernel launches enforce dependencies
     * between consecutive FFT stages on the same stream.
     */
    for (int stage = 1; stage <= logn; stage++) {
        butterfly_kernel<<<blocks_butterfly, threads>>>(
            d_data,
            d_tw,
            n,
            stage
        );

        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float kernel_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &kernel_ms,
        ev_start,
        ev_stop
    ));

    /* Copy transformed output back to CPU. */
    CUDA_CHECK(cudaMemcpy(
        x,
        d_data,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    return (double)kernel_ms;
}