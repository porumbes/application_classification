include Makefile.inc

ARCH=-gencode arch=compute_70,code=compute_70 -gencode arch=compute_70,code=sm_70

all : new

new: new.cu ac.cu
	$(NVCC) -ccbin=${CXX} ${NVCCFLAGS} --compiler-options "${CXXFLAGS}" -o new new.cu ac.cu -I cub/cub

clean:
	rm -f main new