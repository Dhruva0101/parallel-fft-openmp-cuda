# cpu2d - Author: Om Kumar Singh
include ../common/config.mk

BIN := $(BINDIR)/cpu2d
SRC := main.c fft2d_seq.c fft2d_omp.c $(COMMON_SRC)

$(BIN): $(SRC) fft2d.h $(COMMON_HDR)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(OMPFLAGS) -I$(COMMON) -o $@ $(SRC) -lm

clean:
	rm -f $(BIN)

.PHONY: clean
