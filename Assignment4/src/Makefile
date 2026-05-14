
CPP_CODE := $(wildcard *.cpp)
CPP_EXE  := $(patsubst %.cpp,%,$(CPP_CODE))

CUDA_CODE := $(wildcard *.cu)
CUDA_EXE  := $(patsubst %.cu,%,$(CUDA_CODE))
# NVCC_FLAGS ?= -gencode arch=compute_50,code=sm_50

all: cpp cuda

cpp: $(CPP_EXE)

cuda: $(CUDA_EXE)

clean:
	rm -f $(CPP_EXE) $(CUDA_EXE) 

%: %.c *.h
	g++ $< -o $@

%: %.cu *.h
	nvcc $(NVCC_FLAGS) $< -o $@


zip:
	zip assignment4_solutions.zip *.cu *.h *.cpp Makefile