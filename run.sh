#!/bin/bash

# test.sh

make clean
make -j12
mkdir -p results

time ./main data/data.bin data/pattern.bin > results/cuda_result

# cat results/cuda_result_old | openssl md5
cat results/cuda_result     | openssl md5

# python validate.py
