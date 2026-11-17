# cpu1d - Author: Dhruva Kamble
include ../common/config.mk

BIN := $(BINDIR)/cpu1d
SRC := main.c fft1d_seq.c fft1d_omp.c $(COMMON_SRC)

$(BIN): $(SRC) fft1d.h $(COMMON_HDR)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(OMPFLAGS) -I$(COMMON) -o $@ $(SRC) -lm

clean:
	rm -f $(BIN)

.PHONY: clean
