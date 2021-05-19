include Makefile.inc

all : main

main: main.cu ac.hxx
	$(NVCC) -ccbin=${CXX} ${NVCCFLAGS} --compiler-options "${CXXFLAGS}" -o main main.cu

clean:
	rm -f main