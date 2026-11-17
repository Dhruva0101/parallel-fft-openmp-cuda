# Top-level Makefile: delegates to each implementation folder.
#
#   make            build everything that can be built here (CPU always, GPU if nvcc exists)
#   make cpu        build bin/cpu1d bin/cpu2d
#   make gpu        build bin/cuda1d bin/cuda2d     (needs nvcc)
#   make check      correctness tests for every built binary
#   make bench      full benchmark sweep -> results/*.csv
#   make analyze    summary table + plots from results/*.csv
#   make clean

HAVE_NVCC := $(shell command -v nvcc 2>/dev/null)

all: cpu $(if $(HAVE_NVCC),gpu)

cpu:
	$(MAKE) -C cpu1d
	$(MAKE) -C cpu2d

gpu:
	$(MAKE) -C cuda1d
	$(MAKE) -C cuda2d

check: all
	./scripts/check.sh

bench: all
	./scripts/bench.sh

analyze:
	python3 scripts/analyze.py

clean:
	$(MAKE) -C cpu1d clean
	$(MAKE) -C cpu2d clean
	$(MAKE) -C cuda1d clean
	$(MAKE) -C cuda2d clean

help:
	@sed -n '3,10p' Makefile

.PHONY: all cpu gpu check bench analyze clean help
