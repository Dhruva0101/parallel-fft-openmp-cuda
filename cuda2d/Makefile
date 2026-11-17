# cuda2d - Author: Dhruva Kamble
include ../common/config.mk

BIN := $(BINDIR)/cuda2d
SRC := main.c fft2d_cuda.cu $(COMMON_SRC)

$(BIN): $(SRC) $(COMMON_HDR)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -I$(COMMON) -o $@ $(SRC) -lm

clean:
	rm -f $(BIN)

.PHONY: clean
