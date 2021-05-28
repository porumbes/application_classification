#!/bin/bash

# test.sh

make clean
make -j12
mkdir -p results

CUDA_VISIBLE_DEVICES=0       ./main data/data.bin data/pattern.bin > results/cuda_result
CUDA_VISIBLE_DEVICES=0,1     ./main data/data.bin data/pattern.bin > results/cuda_result
CUDA_VISIBLE_DEVICES=0,1,2   ./main data/data.bin data/pattern.bin > results/cuda_result
CUDA_VISIBLE_DEVICES=0,1,2,3 ./main data/data.bin data/pattern.bin > results/cuda_result

# cat results/cuda_result_old | openssl md5
cat results/cuda_result     | openssl md5
# Should be: 084432fb47655341ab992a0f35fcd655

# python validate.py
