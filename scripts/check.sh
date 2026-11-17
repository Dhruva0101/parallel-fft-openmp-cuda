#!/usr/bin/env bash
# check.sh: run correctness tests for every binary that has been built.
# Small sizes are checked against the naive DFT AND by round trip;
# large sizes by round trip only (see docs/DESIGN.md, D10).
# Exit code != 0 if anything fails, so CI goes red.
set -u
cd "$(dirname "$0")/.."

SIZES_1D="2 4 8 16 256 1024 4096 65536 1048576"
SIZES_2D="2 4 8 16 32 64 256 1024"

fail=0
run() {   # run <binary> <sizes...>
    local bin=$1; shift
    if [[ ! -x bin/$bin ]]; then echo "skip: bin/$bin not built"; return; fi
    for n in "$@"; do
        ./bin/"$bin" --size "$n" --check || fail=1
    done
}

run cpu1d  $SIZES_1D
run cuda1d $SIZES_1D
run cpu2d  $SIZES_2D
run cuda2d $SIZES_2D

if (( fail )); then echo; echo "SOME CHECKS FAILED"; exit 1; fi
echo; echo "ALL CHECKS PASSED"
