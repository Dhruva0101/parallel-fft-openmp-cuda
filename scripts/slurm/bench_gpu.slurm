#!/bin/bash
# bench_gpu.slurm: CUDA benchmarks on one ARC GPU node.
# Submit from the repo root:   sbatch scripts/slurm/bench_gpu.slurm
#
# Partition rtx4060ti16g = NVIDIA RTX 4060 Ti 16 GB, compute capability 8.9 -> sm_89.
# GPU model, driver and CUDA version are printed below and recorded in docs/BENCHMARKING.md.
# Only some binaries:   TARGETS=cuda2d sbatch scripts/slurm/bench_gpu.slurm
#SBATCH --job-name=fft-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm-gpu-%j.out
#SBATCH --partition=rtx4060ti16g

set -eu
cd "$SLURM_SUBMIT_DIR"

module load cuda
echo "host: $(hostname)"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
nvcc --version | tail -1

TARGETS=${TARGETS:-"cuda1d cuda2d"}
make gpu CUDA_ARCH=${CUDA_ARCH:-sm_89}
./scripts/check.sh $TARGETS
./scripts/bench.sh $TARGETS
