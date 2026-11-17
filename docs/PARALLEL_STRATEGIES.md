# Parallel strategies

How each implementation maps the FFT onto hardware, and what we expect to see.
Background on the algorithm is in [FFT_BACKGROUND.md](FFT_BACKGROUND.md).

The one question behind every design here: **what is independent, and where
must everyone wait?**

| | Independent work | Must wait |
|---|---|---|
| 1D FFT | the `N/2` butterflies inside one stage | between every stage (`log₂N` times) |
| 2D FFT | every row FFT; then every column FFT | once, between rows and columns |

---

## 1. OpenMP 1D: fine-grained (`cpu1d/fft1d_omp.c`, Dhruva)

```
#pragma omp parallel                  ← threads created once
{
    omp for: bit-reversal swaps       ← barrier
    for each stage s:
        omp for: butterflies 0..N/2-1 ← barrier (stage boundary)
}
```

**Index mapping** (D4). Butterfly `b` in stage `s` (`half = 2^(s-1)`):

```
j = b & (half − 1)            position within its group
k = (b >> (s−1)) << s         first element of its group
i = k + j                     pair is (i, i + half), twiddle tw[j · N/2^s]
```

Example, N = 8, stage 2 (half = 2): b = 0,1,2,3 → i = 0,1,4,5 → pairs (0,2) (1,3) (4,6) (5,7). ✓

**Why not parallelise the group loop?** Groups per stage: `N/2, N/4, …, 1`.
With 16 threads, the last 4 stages have fewer groups than threads, and the
final stage (still `N/2` butterflies of work) runs on one core.

**Expectations.**
- Small N: slower than sequential. `log₂N` barriers × a few µs each > the work.
- Large N: speedup well below the core count, because the FFT is memory-bound
  (background §6), so threads end up waiting on DRAM bandwidth together.
- Late stages pair elements `N/2` apart → poor locality. A possible
  improvement is to run the early stages block-wise in cache.

## 2. OpenMP 2D: coarse-grained (`cpu2d/fft2d_omp.c`, Om)

```
#pragma omp parallel
{
    private column buffer
    omp for: FFT each row          ← one barrier
    omp for: gather column → FFT → scatter
}
```

Each thread runs complete sequential FFTs → only **one** barrier in the whole
transform, versus `log₂N` in 1D. Rows fit in cache (n = 4096 → 32 KB).

**Expectations.** Better speedup than OpenMP 1D at the same element count.
Column phase slower than row phase (strided gather/scatter). Possible false
sharing when neighbouring threads scatter into neighbouring columns.

## 3. CUDA 1D (`cuda1d/fft1d_cuda.cu`, Om)

```
H2D copy
bitrev_kernel       (out-of-place, 1 thread / element)
stage_kernel × log₂N   (1 thread / butterfly; kernel boundary = grid barrier)
D2H copy
```

**Why a launch per stage** (D5): a stage needs results from *anywhere* in the
array produced by the previous stage. Only a kernel boundary guarantees every
block has finished.

**Optimisation**: stages 1–11 only mix elements within 2048-element chunks →
one kernel, one block per chunk, all 11 stages in shared memory with
`__syncthreads()`. Remaining stages (12..log₂N) stay as separate launches.

**Expectations.**
- Small N: slower than CPU (launch overhead ~5 µs × log₂N + copy latency).
- Kernel-only: large speedups at large N (GPU memory bandwidth ≫ CPU).
- End-to-end: PCIe copies dominate; speedup much smaller than kernel-only.

## 4. CUDA 2D (`cuda2d/fft2d_cuda.cu`, Dhruva)

```
H2D copy
row_fft_kernel      grid = n blocks (one per row), row in shared memory
transpose_kernel    32×32 tiles, padded to 33
row_fft_kernel      (on the transposed matrix = original columns)
transpose_kernel    back to original layout
D2H copy
```

**One block per row** (D6) is what makes `__syncthreads()` correct: all
threads that touch a row live in the same block.

**Coalescing.** A warp = 32 threads executing together. If they read 32
consecutive floats2 (256 bytes), the hardware does ~2 memory transactions. If
they read 32 elements with stride `n`, it does 32. Rows → consecutive. Columns →
strided. The transpose converts one into the other.

**Tiled transpose** (D7):
```
read  tile[ty][tx] = in [row = by·32+ty][col = bx·32+tx]   coalesced along tx
sync
write out[row = bx·32+ty][col = by·32+tx] = tile[tx][ty]   coalesced along tx
```
`tile[32][33]`: shared memory has 32 banks. Reading a *column* of a 32-wide
tile hits one bank 32 times (serialised). Padding to 33 shifts each row by one
bank → conflict-free.

**Expectations.** Kernel-only speedup largest of all six versions (only 4
kernels, 4 passes over global memory, highly regular). End-to-end still limited
by PCIe, but less than 1D CUDA because the GPU work per byte copied is larger.

---

## 5. Hypotheses to test in the report

Stated before benchmarking; the results section confirms or refutes each one.

1. Both parallel CPU versions are slower than sequential for small inputs.
2. OpenMP 2D scales better than OpenMP 1D at equal element counts (fewer barriers).
3. OpenMP speedup plateaus below the core count (memory bandwidth).
4. CUDA beats CPU only above a crossover size; that crossover is larger end-to-end than kernel-only.
5. For CUDA, end-to-end time at large N is dominated by host↔device copies.
6. The shared-memory fused stages reduce CUDA 1D kernel time noticeably.
