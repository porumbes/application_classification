#!/bin/bash

# test.sh

make clean
make -j12

# time ./main                      \
#   data/rmat18.Vertex.csv         \
#   data/rmat18.Edges.csv          \
#   data/georgiyPattern.Vertex.csv \
#   data/georgiyPattern.Edges.csv  > cuda_result_old

time ./new data.bin pattern.bin > cuda_result

cat cuda_result_old | openssl md5
cat cuda_result     | openssl md5

# cp cuda_result_old cuda_result
# python test.py
python validate.py

