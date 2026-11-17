# Benchmarking methodology

## What is measured

| Column | Meaning |
|---|---|
| `impl` | `seq`, `omp`, or `cuda` |
| `dim` | 1 or 2 |
| `size` | 1D: N. 2D: matrix side n |
| `elements` | N, or n²; the x-axis for comparing 1D and 2D |
| `threads` | `OMP_NUM_THREADS` in effect (1 for non-OpenMP binaries) |
| `run` | 0 = warm-up (dropped in analysis), 1..R = timed |
| `total_ms` | wall-clock time of the whole call (CUDA: includes H2D + D2H copies) |
| `kernel_ms` | CUDA only: time between events around the kernels; −1 otherwise |

**Speedup** `S = T_seq / T_impl`, where `T_seq` is the sequential version of
the same dimension and size, and each `T` is the mean of runs 1..R.

## Sizes

Matched by element count so 1D and 2D land on the same x-axis:

| elements | 2¹⁰ | 2¹² | 2¹⁴ | 2¹⁶ | 2¹⁸ | 2²⁰ | 2²² | 2²⁴ |
|---|---|---|---|---|---|---|---|---|
| 1D N | 1024 | 4096 | 16384 | 65536 | 262144 | 1048576 | 4194304 | 16777216 |
| 2D n×n | 32² | 64² | 128² | 256² | 512² | 1024² | 2048² | 4096² |

2²⁴ complex floats = 128 MB per buffer, which fits easily in node and GPU memory.

## Controls (why the numbers can be trusted)

- **Warm-up run discarded**: absorbs page faults, thread-pool creation, CUDA
  context creation, twiddle/device-buffer setup.
- **Fresh input every run**, copied outside the timed region.
- **Exclusive node** for CPU runs (`#SBATCH --exclusive`): no other jobs
  competing for cores or memory bandwidth.
- **Thread pinning**: `OMP_PROC_BIND=close`, `OMP_PLACES=cores`; threads =
  physical cores (no hyperthreads).
- **Fixed configuration** across all sizes: one thread count, block size 256,
  one node type, one GPU type (proposal §3).
- **Same compiler flags** for every version (D12).
- **Mean ± std** reported; a std above ~5 % of the mean means re-run.

## Running

```bash
# local quick look
make && make check
REPS=3 ./scripts/bench.sh cpu1d

# ARC, from the repo root
sbatch scripts/slurm/bench_cpu.slurm
sbatch scripts/slurm/bench_gpu.slurm
# then
python3 scripts/analyze.py      # -> results/summary.md, plots/*.png
```

## Machine (fill in from the Slurm job output)

| | CPU runs | GPU runs |
|---|---|---|
| ARC node / partition | | |
| CPU model, cores | | |
| Memory | | |
| GPU model, memory | — | |
| Compiler / version | gcc | nvcc |
| CUDA / driver | — | |
| `OMP_NUM_THREADS` | | — |
| `CUDA_ARCH` | — | |
