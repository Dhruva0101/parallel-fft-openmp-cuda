#!/bin/bash
# bench_gpu.slurm: CUDA benchmarks on one ARC GPU node.
# Submit from the repo root:   sbatch scripts/slurm/bench_gpu.slurm
#
# TODO: set the ARC GPU partition and CUDA_ARCH to match the GPU
# (see common/config.mk). GPU model, driver and CUDA version are printed
# below and recorded in docs/BENCHMARKING.md.
#SBATCH --job-name=fft-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm-gpu-%j.out
##SBATCH --partition=<gpu-partition>

set -eu
cd "$SLURM_SUBMIT_DIR"

# module load cuda        # TODO: exact module name on ARC
echo "host: $(hostname)"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
nvcc --version | tail -1

make gpu CUDA_ARCH=${CUDA_ARCH:-sm_80}
./scripts/check.sh
./scripts/bench.sh cuda1d cuda2d
