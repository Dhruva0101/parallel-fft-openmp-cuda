# config.mk: shared build settings, included by every folder Makefile.
# One place for compiler flags keeps them identical across all versions (D12).

# Relative paths on purpose: make cannot handle spaces in file names, so an
# absolute path would break the build if the repo sits in a folder with a space.
# Every folder Makefile lives exactly one level below the repo root, so ".." is the root.
ROOT    := ..
COMMON  := $(ROOT)/common
BINDIR  := $(ROOT)/bin
COMMON_SRC := $(COMMON)/fft_common.c $(COMMON)/harness.c
COMMON_HDR := $(COMMON)/fft_common.h $(COMMON)/harness.h

# ---- C compiler --------------------------------------------------------------
# Apple's `gcc` is really clang and has no OpenMP. On macOS, pick the newest
# Homebrew gcc-NN automatically. On ARC (Linux) plain gcc is used.
# Override any time:  make CC=gcc-14
ifeq ($(origin CC),default)
  ifeq ($(shell uname -s),Darwin)
    CC := $(or $(shell ls /opt/homebrew/bin/gcc-[0-9]* /usr/local/bin/gcc-[0-9]* 2>/dev/null | sort -t- -k2 -n | tail -1),gcc)
  else
    CC := gcc
  endif
endif

# -O3 for both baseline and parallel versions (D12: same flags everywhere).
# No -march=native: binaries built on an ARC login node may run on a compute
# node with a different CPU, and -ffast-math would change numerical results.
CFLAGS   ?= -O3 -std=c11 -Wall -Wextra
OMPFLAGS ?= -fopenmp

# ---- CUDA --------------------------------------------------------------------
# Set CUDA_ARCH to match the ARC GPU you benchmark on (see docs/BENCHMARKING.md):
#   sm_70 = V100, sm_75 = T4/RTX 20xx, sm_80 = A100, sm_86 = A30/RTX 30xx, sm_89 = L40/RTX 40xx
NVCC      ?= nvcc
CUDA_ARCH ?= sm_80
NVFLAGS   ?= -O3 -arch=$(CUDA_ARCH) -lineinfo
