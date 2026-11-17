# FFT background

The algorithm behind every implementation in this project, from the definition
of the DFT to the iterative radix-2 FFT and its 2D extension.

---

## 1. What the DFT computes

Given `N` complex samples `x[0..N-1]`, the Discrete Fourier Transform produces
`N` complex outputs:

```
X[k] = Σ_{t=0}^{N-1}  x[t] · W_N^(k·t)          where  W_N = e^(−2πi/N)
```

`W_N` is the "N-th root of unity": a point on the unit circle, `1/N` of a turn
clockwise. `X[k]` measures how much of the frequency "k cycles per N samples"
is present in the signal.

Computed directly: `N` outputs × `N` terms each = **O(N²)**. For `N = 2²⁰`
that's 10¹² complex multiply-adds. `common/fft_common.c: naive_dft_1d` does
exactly this; we use it only to check answers on small inputs.

---

## 2. Cooley–Tukey: divide and conquer

Split the sum into even-indexed and odd-indexed samples:

```
X[k] = Σ x[2t]·W_N^(2tk)  +  W_N^k · Σ x[2t+1]·W_N^(2tk)
     =      E[k]          +  W_N^k ·      O[k]
```

Because `W_N^2 = W_{N/2}`, `E` and `O` are themselves DFTs of size `N/2`.
And because `W_N^(k+N/2) = −W_N^k`, the second half of the outputs comes free:

```
X[k]       = E[k] + W_N^k · O[k]
X[k + N/2] = E[k] − W_N^k · O[k]          for k = 0 .. N/2 − 1
```

That pair of equations is the **butterfly**: 1 complex multiply, 1 add, 1 subtract.

```
   E[k] ──────●────────────►  E + W·O   = X[k]
               ╲   ╱
                ╳
               ╱   ╲
   O[k] ──(×W)─●────────────►  E − W·O   = X[k+N/2]
```

Recurse until size 1 (a 1-point DFT is the identity). There are `log₂N` levels,
each does `N/2` butterflies → **O(N log N)**. For `N = 2²⁰` that's ~10⁷ instead
of 10¹². This is why `N` must be a power of two (radix-2): we halve at every level.

`W_N^k` values are called **twiddle factors**.

---

## 3. From recursion to iteration: bit reversal

Recursion is awkward to parallelise (and slow on a GPU). The iterative form
runs the recursion *bottom-up*. Trace where each input ends up after splitting
even/odd repeatedly, for `N = 8`:

| index | binary | after even/odd splits, lands at | binary |
|---|---|---|---|
| 0 | 000 | 0 | 000 |
| 1 | 001 | 4 | 100 |
| 2 | 010 | 2 | 010 |
| 3 | 011 | 6 | 110 |
| 4 | 100 | 1 | 001 |
| 5 | 101 | 5 | 101 |
| 6 | 110 | 3 | 011 |
| 7 | 111 | 7 | 111 |

The destination is the **index with its bits reversed**. So: permute once up
front with `bit_reverse`, and from then on every stage only combines
*neighbouring* blocks, in place, with no extra memory.

---

## 4. The iterative stages

After bit reversal, stage `s = 1 .. log₂N` combines pairs of size-`m/2` DFTs into
size-`m` DFTs, where `m = 2^s`:

```
N = 8 after bit-reversal:   x0 x4 x2 x6 x1 x5 x3 x7

stage 1 (m=2, half=1):  [x0 x4] [x2 x6] [x1 x5] [x3 x7]     4 groups × 1 butterfly
stage 2 (m=4, half=2):  [x0 x4 x2 x6]   [x1 x5 x3 x7]       2 groups × 2 butterflies
stage 3 (m=8, half=4):  [x0 x4 x2 x6 x1 x5 x3 x7]           1 group  × 4 butterflies
```

Inside a group starting at `k`, butterfly `j` pairs element `k+j` with
`k+j+half`, using twiddle `W_m^j`.

**Every stage has exactly N/2 butterflies, but the number of groups drops
from N/2 to 1.** This drives the OpenMP design (see
[PARALLEL_STRATEGIES.md §1](PARALLEL_STRATEGIES.md)).

**Twiddle table trick.** `W_m^j = W_N^(j·N/m)`, so one table
`tw[k] = W_N^k, k < N/2` serves every stage: index it with `j * (N/m)`.

---

## 5. 2D FFT: separability

For an `n × n` matrix:

```
X[u][v] = Σ_r Σ_c x[r][c] · W_n^(u·r) · W_n^(v·c)
        = Σ_r W_n^(u·r) · ( Σ_c x[r][c] · W_n^(v·c) )
```

The inner sum is a 1D FFT of row `r`. So: **1D FFT every row, then 1D FFT every
column.** The 2D FFT is therefore `2n` 1D FFTs in two independent batches.

The catch is memory layout. C stores matrices row-major: a row is contiguous,
a column is spread out with stride `n`. Columns are therefore slow to access
(cache misses on CPU, uncoalesced loads on GPU). Two fixes:
- **Gather/scatter**: copy a column into a small contiguous buffer, FFT, copy back (CPU).
- **Transpose**: turn columns into rows, reuse the row FFT, transpose back (GPU).

---

## 6. Why the FFT is hard to speed up: it's memory-bound

Per stage: `N/2` butterflies × 10 flops = `5N` flops.
Per stage memory: read and write `N` complex floats = `16N` bytes.

**Arithmetic intensity ≈ 5N / 16N ≈ 0.3 flops/byte.** Modern CPUs and GPUs
can do 10–50+ flops per byte they load from DRAM. So the FFT spends most of its
time *waiting for memory*, not computing. Consequences we expect to see:

- Adding CPU threads stops helping once memory bandwidth is saturated.
- On the GPU, what matters is **how many times data crosses global memory**,
  which motivates doing several stages per load in shared memory.
- For 1D CUDA, the PCIe copy (≈ 16N bytes over ~25 GB/s) can cost more than
  the whole transform on the GPU. Hence: report kernel-only AND end-to-end.

---

## 7. Precision

Floats have ~7 significant digits (`ε ≈ 1.2·10⁻⁷`). FFT round-off grows roughly
like `ε · log₂N`, so at `N = 2²⁴` we expect errors around 10⁻⁶, well below our
tolerance of 10⁻⁴, while a real bug produces errors near 1.
