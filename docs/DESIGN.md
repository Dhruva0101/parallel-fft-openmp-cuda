# Design decisions

The main design choices in this project, each written as a short record:
context, decision, alternatives, consequences. Source files refer to these by
number (for example "D5").

---

### D0: Split the work by ownership

**Context.** Two authors and six implementations. We wanted to work in parallel
without blocking each other or causing merge conflicts.

**Decision.** Each author owns one sequential, one OpenMP and one CUDA version,
crossing dimensions:

| | Sequential | OpenMP | CUDA |
|---|---|---|---|
| Dhruva | 1D | 1D | 2D |
| Om | 2D | 2D | 1D |

Each implementation folder has a single owner. The shared code in `common/`
was written jointly up front.

**Alternatives.** Splitting by dimension (one author would never compare CUDA
1D vs 2D trade-offs) or by backend (the CUDA author would depend on the CPU
author's baseline).

**Consequences.** We both work on CUDA and on both dimensions. The 2D code
contains its own short 1D FFT routine, which keeps `cpu2d/` independent of `cpu1d/`.

### D1: `float` everywhere, with a `double` reference

**Context.** GPUs are much faster in single precision. Comparing a `double`
CPU code with a `float` GPU code would not be a fair speedup.

**Decision.** All six timed implementations use `cpx = {float re, im}`. The
naive DFT used for checking accumulates in `double`.

**Alternatives.** `double` everywhere. This is fair too, but FP64 throughput
varies widely between GPU models and would dominate the comparison.

**Consequences.** Same bytes moved and same arithmetic on CPU and GPU. `cpx`
has the same layout as CUDA's `float2`, so buffers copy to the device as-is.

### D2: Generated input instead of data files

**Context.** Text input files at 2²⁴ elements are hundreds of MB and slow to
parse, and they are easy to truncate or corrupt without noticing.

**Decision.** `gen_input(buf, n, seed)` uses a deterministic xorshift generator.
The same `(n, seed)` gives identical data on every machine and in every implementation.

**Alternatives.** Binary files (still large, still need generating); `rand()`
(its sequence differs between macOS and Linux C libraries).

**Consequences.** No I/O, any size on demand, reproducible on ARC and on laptops.

### D3: Precomputed twiddle table, cached per size

**Context.** The textbook loop updates the twiddle factor with `w *= wm`. Each
multiply adds some round-off, and the error accumulates within a group.

**Decision.** `make_twiddles` computes `tw[k] = e^(−2πik/N)` once in `double`
and rounds to `float`. Each implementation caches its table (and, on the GPU,
its device buffers) and rebuilds only when `N` changes.

**Alternatives.** The recurrence (less accurate, and a serial dependency);
calling `sincos` per butterfly (accurate, but expensive in the inner loop).

**Consequences.** One table serves every stage through `tw[j·N/m]`. Setup cost
falls in the warm-up run for every implementation equally.

### D4: OpenMP 1D parallelises butterflies, not groups

**Context.** Stage `s` has `N/2^s` groups. Parallelising over groups gives
plenty of work in early stages but only one group in the last stage, so the
largest stages would run on a single thread.

**Decision.** Each stage is flattened into a loop over its `N/2` butterflies,
and butterfly `b` is mapped to its group and offset with bit operations. One
`#pragma omp parallel` region contains all stages; each stage is an `omp for`
whose implicit barrier separates it from the next.

**Alternatives.** `collapse(2)` over groups and offsets (awkward because the
inner bound changes per stage); recursive tasks (more overhead).

**Consequences.** Every stage is evenly balanced, and the thread team is
created once rather than `log₂N` times.

### D5: CUDA 1D uses one kernel launch per stage

**Context.** `__syncthreads()` synchronises only the threads of one block.
For large `N` a stage spans many blocks, so block-level sync cannot order stages.

**Decision.** One kernel launch per stage. Launches on the same stream run in
order, so a kernel boundary acts as a grid-wide barrier. Bit reversal is done
out of place to avoid read/write races. As an optimisation, the first stages
(which only mix data inside 2048-element chunks) are fused into a single
shared-memory kernel.

**Alternatives.** Cooperative groups `grid.sync()` (requires all blocks to be
resident at once, which limits `N`); a Stockham FFT (avoids bit reversal but is
harder to compare directly with the CPU code).

**Consequences.** Correct at every size. The cost is `log₂N` launches of a few
microseconds each, which the fused kernel reduces.

### D6: CUDA 2D uses one thread block per row, held in shared memory

**Context.** Each row FFT needs a barrier between its stages. If a row were
spread across blocks, `__syncthreads()` would not be enough (same issue as D5).

**Decision.** One block per row. The row is loaded into shared memory, with the
bit-reversal applied during the load, and all stages synchronise with
`__syncthreads()`. Threads loop over butterflies, so rows with more than 2048
elements still work.

**Alternatives.** Per-stage launches as in D5 (correct, but `2·log₂n` launches
and global-memory traffic on every stage).

**Consequences.** Each row is read from and written to global memory once.
`n = 4096` needs 32 KB of shared memory per block, within the 48 KB default,
so one kernel covers the whole size range.

### D7: Transpose between the row and column passes on the GPU

**Context.** Columns have stride `n`, so a warp reading a column touches 32
separate memory segments (uncoalesced).

**Decision.** Row FFT, tiled transpose, row FFT, transpose back. The transpose
uses 32×32 shared-memory tiles padded to 33 columns.

**Alternatives.** A column kernel reading strided memory directly (simpler,
much slower); leaving the output transposed (faster, but different from the
CPU results).

**Consequences.** Reads and writes are coalesced in both transposes, and the
padding avoids shared-memory bank conflicts.

### D8: OpenMP 2D processes rows, then columns

**Decision.** Row FFTs in parallel, one barrier, then column FFTs in parallel.
Each thread gathers a column into its own buffer, transforms it and scatters it back.

**Consequences.** This gives a direct comparison between coarse-grained
parallelism (2D: whole FFTs per thread, one barrier) and fine-grained
parallelism (1D: butterflies per thread, `log₂N` barriers).

### D9: Wall-clock timing, and two numbers for CUDA

**Context.** `clock()` measures CPU time summed over all threads, so a parallel
run appears slower the more threads it uses.

**Decision.** `now_ms()` uses `CLOCK_MONOTONIC`. The harness times every call
the same way (`total_ms`). CUDA versions also return `kernel_ms`, measured with
`cudaEvent`s around the kernels only.

**Consequences.** CUDA `total_ms` is end-to-end (including PCIe copies) and is
directly comparable with CPU times. `kernel_ms` shows the GPU's raw speed.

### D10: Correctness via naive DFT, round trip and cross-check

**Decision.** Three levels of checking:
1. Small sizes: compare with a `double` naive DFT.
2. Any size: round trip using `IFFT(y) = conj(FFT(conj(y)))/N`, which needs
   only the forward transform.
3. Final: CUDA results against the sequential results on large inputs.

The error metric is `max|got − ref| / max|ref|`, since per-element relative
error is unstable for outputs close to zero. Tolerance is `1e-4`.

**Consequences.** Each implementation can be verified independently, at every size.

### D11: Shared harness and CSV output

**Decision.** `common/harness.c` handles argument parsing, input generation,
verification, timing and output for all six versions. Output is one CSV row per run:
`impl,dim,size,elements,threads,run,total_ms,kernel_ms`. Run 0 is a warm-up.

**Consequences.** Implementation files contain only FFT code, and a single
script (`scripts/analyze.py`) processes every result file.

### D12: Same compiler flags for every version

**Decision.** `-O3` for all C code and `-O3 -arch=<gpu>` for CUDA, set once in
`common/config.mk`. No `-ffast-math` (it changes numerical results) and no
`-march=native` (binaries built on an ARC login node may run on a different CPU).

**Consequences.** Measured speedups reflect parallelism, not compiler settings.
