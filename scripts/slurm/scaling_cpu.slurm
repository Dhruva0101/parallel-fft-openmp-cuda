#!/bin/bash
# scaling_cpu.slurm: strong-scaling sweep for the OpenMP versions.
# Fixed problem sizes, thread counts 1..16, on one whole ARC node.
# Submit from the repo root:   sbatch scripts/slurm/scaling_cpu.slurm
#
# Output: results/scaling.csv (same columns as the other CSVs; `threads`
# says how many were used). Sequential runs once per size as the baseline.
#SBATCH --job-name=fft-scaling
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=results/slurm-scaling-%j.out
#SBATCH --partition=rome

set -eu
cd "$SLURM_SUBMIT_DIR"

echo "host: $(hostname)"; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Socket|L3' || true

THREADS=${THREADS:-"1 2 4 8 16"}
REPS=${REPS:-5}
# Same element counts for 1D and 2D: 2^16, 2^20, 2^24 (2D sides 256, 1024, 4096).
SIZES_1D=${SIZES_1D:-"65536 1048576 16777216"}
SIZES_2D=${SIZES_2D:-"256 1024 4096"}

# Same pinning as bench.sh: threads packed onto neighbouring physical cores.
export OMP_PROC_BIND=close OMP_PLACES=cores

make cpu
./scripts/check.sh cpu1d cpu2d

# Write to a per-job file and rename at the end, so two jobs submitted by
# accident can never interleave rows in the same CSV.
final=results/scaling.csv
out=results/scaling.${SLURM_JOB_ID:-local}.partial.csv
./bin/cpu1d --header > "$out"
for n in $SIZES_1D; do
    echo ">> cpu1d seq size=$n"
    OMP_NUM_THREADS=1 ./bin/cpu1d --impl seq --size "$n" --reps "$REPS" >> "$out"
    for t in $THREADS; do
        echo ">> cpu1d omp size=$n threads=$t"
        OMP_NUM_THREADS=$t ./bin/cpu1d --impl omp --size "$n" --reps "$REPS" >> "$out"
    done
done
for n in $SIZES_2D; do
    echo ">> cpu2d seq size=$n"
    OMP_NUM_THREADS=1 ./bin/cpu2d --impl seq --size "$n" --reps "$REPS" >> "$out"
    for t in $THREADS; do
        echo ">> cpu2d omp size=$n threads=$t"
        OMP_NUM_THREADS=$t ./bin/cpu2d --impl omp --size "$n" --reps "$REPS" >> "$out"
    done
done
mv "$out" "$final"
echo "wrote $final"
