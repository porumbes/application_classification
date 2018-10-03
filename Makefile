# application_classification/Makefile

ARCH=\
  -gencode arch=compute_60,code=compute_60 \
  -gencode arch=compute_60,code=sm_60

all : main

main: main.cu ac.cu
	nvcc $(ARCH) -o main \
		--compiler-options -Wall \
		main.cu ac.cu \
		-I cub/cub

clean:
	rm -f main