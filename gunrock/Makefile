ARCH=\
  -gencode arch=compute_60,code=compute_60 \
  -gencode arch=compute_60,code=sm_60


all : main

main: main.cu readTable.cc initialize.cu iterate.cc kernels.cu kernels.cuh
	nvcc $(ARCH) -o main \
		--compiler-options -Wall \
		main.cu readTable.cc initialize.cu iterate.cc kernels.cu \
		-I cub/cub

clean:
	rm -f main