ARCH=\
  -gencode arch=compute_60,code=compute_60 \
  -gencode arch=compute_60,code=sm_60


all : main

main: main.cu readTable.cc constructGraph.cc initialize.cu iterate.cc kernels.cu
	nvcc $(ARCH) -o main main.cu readTable.cc constructGraph.cc initialize.cu iterate.cc \
		-I cub/cub

clean:
	rm -f main