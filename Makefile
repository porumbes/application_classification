ARCH=\
  -gencode arch=compute_60,code=compute_60 \
  -gencode arch=compute_60,code=sm_60


all : main

main: main.cc readTable.cc constructGraph.cc initialize.cc iterate.cc timer.cc
	g++ -w -lm -fopenmp -O3 -o main main.cc readTable.cc constructGraph.cc initialize.cc iterate.cc timer.cc

clean:
	rm -f main