ARCH=\
  -gencode arch=compute_60,code=compute_60 \
  -gencode arch=compute_60,code=sm_60


all : main test

main: main.cc readTable.cc constructGraph.cc initialize.cc iterate.cc timer.cc
	g++ -w -lm -fopenmp -O3 -o main main.cc readTable.cc constructGraph.cc initialize.cc iterate.cc timer.cc

test: test.cu
	nvcc -g $(ARCH) -o test \
		test.cu 
	    # -I$(GRAPHBLAS_PATH)/ext/moderngpu/include \
	    # -I$(GRAPHBLAS_PATH)/ext/cub/cub \
	    # -I$(GRAPHBLAS_PATH)/ \
	    # -I/usr/local/cuda/samples/common/inc/ \
	    # -lboost_program_options \
	    # -lcublas \
	    # -lcusparse

clean:
	rm -f main