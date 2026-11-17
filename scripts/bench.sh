#!/usr/bin/env bash
# bench.sh: benchmark sweep. Writes one CSV per binary into results/.
#
#   ./scripts/bench.sh                 # every built binary
#   ./scripts/bench.sh cpu1d cpu2d     # only these
#
# Size ranges are matched by ELEMENT COUNT so 1D and 2D can be compared:
#   1D: N = 2^10 .. 2^24            2D: n = 2^5 .. 2^12  (n*n = 2^10 .. 2^24)
# See docs/BENCHMARKING.md for the full methodology.
set -eu
cd "$(dirname "$0")/.."

REPS=${REPS:-5}
SIZES_1D=${SIZES_1D:-"1024 4096 16384 65536 262144 1048576 4194304 16777216"}
SIZES_2D=${SIZES_2D:-"32 64 128 256 512 1024 2048 4096"}

# Pin OpenMP threads to cores so the OS doesn't migrate them mid-run (less noise).
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}

targets=("$@")
[[ ${#targets[@]} -eq 0 ]] && targets=(cpu1d cpu2d cuda1d cuda2d)

mkdir -p results
for bin in "${targets[@]}"; do
    [[ -x bin/$bin ]] || { echo "skip: bin/$bin not built"; continue; }
    case $bin in *1d) sizes=$SIZES_1D ;; *) sizes=$SIZES_2D ;; esac
    out=results/$bin.csv
    ./bin/"$bin" --header > "$out"
    for n in $sizes; do
        echo ">> $bin size=$n"
        ./bin/"$bin" --size "$n" --reps "$REPS" >> "$out"
    done
    echo "wrote $out"
done
