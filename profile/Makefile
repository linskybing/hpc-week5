NVCC := nvcc
CFLAGS := -O2 -arch=sm_70

TARGETS := transpose transpose_shared

all: $(TARGETS)

transpose: transpose.cu
	$(NVCC) $(CFLAGS) -o $@ $<

transpose_shared: transpose_shared.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
