#!/bin/bash

# run.sh

make clean
make
rm -f orig_result
time ./main 2 \
    data/georgiyData.Vertex.csv \
    data/georgiyData.Edges.csv \
    data/georgiyPattern.Vertex.csv \
    data/georgiyPattern.Edges.csv > orig_result

# tail orig_result | openssl md5

rm -f python_result
python test.py
python validate.py