# cuda1d - Author: Om Kumar Singh
include ../common/config.mk

BIN := $(BINDIR)/cuda1d
SRC := main.c fft1d_cuda.cu $(COMMON_SRC)

# nvcc sends .cu files through the CUDA compiler and .c files to the host
# C compiler, then links everything together.
$(BIN): $(SRC) $(COMMON_HDR)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -I$(COMMON) -o $@ $(SRC) -lm

clean:
	rm -f $(BIN)

.PHONY: clean
