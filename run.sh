# !/bin/bash

# run.sh

# make clean && make
# rm -f cuda_result
# ./main \
#     ./data/georgiyData.Vertex.csv \
#     ./data/georgiyData.Edges.csv \
#     ./data/georgiyPattern.Vertex.csv \
#     ./data/georgiyPattern.Edges.csv > cuda_result

# cat cuda_result | openssl md5
# echo "(stdin)= bd57a5126d5f943ad5c15408d410790d"

# rm -f python_result
# python test.py
# python validate.py


# --
# Test rmat

make clean
make
rm -f cuda_result
time ./main \
    data/rmat18.Vertex.csv \
    data/rmat18.Edges.csv \
    data/georgiyPattern.Vertex.csv \
    data/georgiyPattern.Edges.csv > cuda_result

cat cuda_result | openssl md5
