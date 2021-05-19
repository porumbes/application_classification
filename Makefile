include Makefile.inc

all : main

main: src/main.cu src/ac.hxx
	$(NVCC)                            \
		-ccbin=${CXX}                    \
		${NVCCFLAGS}                     \
		--compiler-options "${CXXFLAGS}" \
		-o main src/main.cu

clean:
	rm -f main