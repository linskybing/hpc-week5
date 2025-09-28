NVCC = nvcc
CFLAGS = -O2 -arch=sm_70 -std=c++17

TARGETS = mul_v1 mul_v2 mul_v3

SRCS = mul_v1.cu mul_v2.cu mul_v3.cu

all: $(TARGETS)

%: %.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean
