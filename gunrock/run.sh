# !/bin/bash

# run.sh

make clean && make
rm -f orig_result
./main \
    ../data/georgiyData.Vertex.csv \
    ../data/georgiyData.Edges.csv \
    ../data/georgiyPattern.Vertex.csv \
    ../data/georgiyPattern.Edges.csv > orig_result

cat orig_result | openssl md5
echo "(stdin)= bd57a5126d5f943ad5c15408d410790d"

rm -f python_result
python test.py
python validate.py


# --
# Test rmat

# DATA_PATH="/home/bjohnson/scratch/ApplicationClassification/data"
# make clean
# make
# rm -f orig_result
# time ./main \
#     $DATA_PATH/rmat18.Vertex.csv \
#     $DATA_PATH/rmat18.Edges.csv \
#     $DATA_PATH/georgiyPattern.Vertex.csv \
#     $DATA_PATH/georgiyPattern.Edges.csv > orig_result

# cat orig_result | openssl md5
