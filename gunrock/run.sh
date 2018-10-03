# !/bin/bash

# run.sh

DATA_PATH="/home/bjohnson/scratch/ApplicationClassification/data"
make clean
make
rm -f orig_result
time ./main \
    $DATA_PATH/rmat18.Vertex.csv \
    $DATA_PATH/rmat18.Edges.csv \
    $DATA_PATH/georgiyPattern.Vertex.csv \
    $DATA_PATH/georgiyPattern.Edges.csv > orig_result

# make clean
# make
# rm -f orig_result
# ./main \
#     ../data/georgiyData.Vertex.csv \
#     ../data/georgiyData.Edges.csv \
#     ../data/georgiyPattern.Vertex.csv \
#     ../data/georgiyPattern.Edges.csv > orig_result

# rm -f python_result
# python test.py
# python validate.py
