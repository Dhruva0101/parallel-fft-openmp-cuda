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
#define MAX_THREADS 1024

/* cpx and float2 have the same layout (see fft_common.h), so device buffers
 * are plain float2 and host data is copied in without conversion. */
static_assert(sizeof(cpx) == sizeof(float2), "cpx must match float2 layout");

__device__ __forceinline__ float2 c_add(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ float2 c_sub(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
__device__ __forceinline__ float2 c_mul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

/*
 * Batched row FFT (D6). ONE BLOCK PER ROW: the whole row sits in shared
 * memory, so __syncthreads() is a valid barrier between all log2(n) stages:
 * every thread that touches this row is in this block.
 * Every thread reaches every __syncthreads() (no early returns).
 */
__global__ void row_fft_kernel(float2 *data, const float2 *__restrict__ tw, int n, int logn) {
    extern __shared__ float2 s[];
    float2 *g = data + (size_t)blockIdx.x * n;

    /* Load with bit reversal folded in. The global read is coalesced. */
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        s[__brev((unsigned)i) >> (32 - logn)] = g[i];
    __syncthreads();

    for (int st = 1; st <= logn; st++) {
        const int half = 1 << (st - 1);
        const int stride = n >> st;
        for (int b = threadIdx.x; b < n / 2; b += blockDim.x) {   /* n/2 may exceed blockDim */
            int j = b & (half - 1);
            int i = ((b >> (st - 1)) << st) + j;
            float2 t = c_mul(tw[j * stride], s[i + half]);
            float2 u = s[i];
            s[i]        = c_add(u, t);
            s[i + half] = c_sub(u, t);
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < n; i += blockDim.x) g[i] = s[i];
}

/*
 * Tiled transpose (D7): out = in^T for an n x n matrix.
 * Reads a 32x32 tile with coalesced row reads, then writes it to the mirrored
 * position, reading the tile column-wise so the global write is coalesced too.
 * The +1 padding column puts each tile column in 32 different shared-memory
 * banks; without it the transposed read is a 32-way bank conflict.
 * Bounds checks cover n < 32 (test sizes 2..16).
 */
__global__ void transpose_kernel(float2 *__restrict__ out, const float2 *__restrict__ in, int n) {
    __shared__ float2 tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;   /* column in `in` */
    int y = blockIdx.y * TILE + threadIdx.y;   /* row in `in`    */
    for (int r = 0; r < TILE; r += TILE_ROWS)
        if (x < n && y + r < n)
            tile[threadIdx.y + r][threadIdx.x] = in[(size_t)(y + r) * n + x];
    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;       /* column in `out` */
    y = blockIdx.x * TILE + threadIdx.y;       /* row in `out`    */
    for (int r = 0; r < TILE; r += TILE_ROWS)
        if (x < n && y + r < n)
            out[(size_t)(y + r) * n + x] = tile[threadIdx.x][threadIdx.y + r];
}

/* Device buffers persist between calls and are rebuilt only when n changes, so
 * cudaMalloc and the twiddle upload land in the warm-up run (D3). */
static float2 *d_a = NULL, *d_b = NULL, *d_tw = NULL;
static int cur_n = 0;
static cudaEvent_t ev_start, ev_stop;
static int events_ready = 0;

static void prepare(int n) {
    if (!events_ready) {
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));
        events_ready = 1;
    }
    if (n == cur_n) return;

    size_t smem = (size_t)n * sizeof(float2);
    if (smem > 48 * 1024) {
        /* Rows above 6144 elements need the opt-in shared-memory limit. */
        int dev, max_smem;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        if (smem > (size_t)max_smem) {
            fprintf(stderr, "fft2d_cuda: n = %d needs %zu B shared memory per row, device allows %d B\n",
                    n, smem, max_smem);
            exit(1);
        }
        CUDA_CHECK(cudaFuncSetAttribute(row_fft_kernel,
                   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_tw));
    size_t bytes = (size_t)n * n * sizeof(float2);
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_tw, (size_t)(n / 2) * sizeof(float2)));

    /* Same twiddle table as the CPU versions (computed in double, rounded once). */
    cpx *tw = (cpx *)malloc((size_t)(n / 2) * sizeof(cpx));
    if (!tw) { fprintf(stderr, "fft2d_cuda: out of memory\n"); exit(1); }
    make_twiddles(tw, (size_t)n);
    CUDA_CHECK(cudaMemcpy(d_tw, tw, (size_t)(n / 2) * sizeof(float2), cudaMemcpyHostToDevice));
    free(tw);
    cur_n = n;
}

static void row_fft(float2 *buf, int n, int logn) {
    int threads = n / 2 < MAX_THREADS ? n / 2 : MAX_THREADS;
    size_t smem = (size_t)n * sizeof(float2);
    row_fft_kernel<<<n, threads, smem>>>(buf, d_tw, n, logn);
    CUDA_CHECK(cudaGetLastError());
}

static void transpose(float2 *out, const float2 *in, int n) {
    dim3 block(TILE, TILE_ROWS);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    transpose_kernel<<<grid, block>>>(out, in, n);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" double fft2d_cuda(cpx *x, int n) {
    prepare(n);
    const int logn = ilog2((size_t)n);
    const size_t bytes = (size_t)n * n * sizeof(float2);

    CUDA_CHECK(cudaMemcpy(d_a, x, bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(ev_start));
    row_fft(d_a, n, logn);        /* rows done                  */
    transpose(d_b, d_a, n);       /* columns are now rows       */
    row_fft(d_b, n, logn);        /* columns done               */
    transpose(d_a, d_b, n);       /* back to original layout    */
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop));

    CUDA_CHECK(cudaMemcpy(x, d_a, bytes, cudaMemcpyDeviceToHost));
    return (double)kernel_ms;
}
