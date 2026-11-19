#!/bin/bash
# bench_cpu.slurm: CPU benchmarks on one whole ARC node (exclusive, so no other jobs share it).
# Submit from the repo root:   sbatch scripts/slurm/bench_cpu.slurm
#
# The CPU model is printed below and recorded in docs/BENCHMARKING.md ("Machine" table).
# Only some binaries:   TARGETS=cpu1d sbatch scripts/slurm/bench_cpu.slurm
#SBATCH --job-name=fft-cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm-cpu-%j.out
#SBATCH --partition=rome

set -eu
cd "$SLURM_SUBMIT_DIR"

echo "host: $(hostname)"; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Socket' || true

# One thread per PHYSICAL core (hyperthreads share FP units and would blur results).
CORES=$(lscpu -p=CORE | grep -v '^#' | sort -u | wc -l)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$CORES}
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

TARGETS=${TARGETS:-"cpu1d cpu2d"}
make cpu
./scripts/check.sh $TARGETS
./scripts/bench.sh $TARGETS
