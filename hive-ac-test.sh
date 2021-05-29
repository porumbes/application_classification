#!/bin/bash

APP_NAME="main"

BIN_PREFIX="./"
DATA_PREFIX="./data"

OUTPUT_DIR=${1:-"eval_mgpu"}
NUM_GPUS=${2:-"1"}
JSON_FILE=""

#for file_name in "${DATA1[@]}"
#do
     # prepare output json file name with number of gpus for this run
     JSON_FILE="ac__rmat18__GPU${NUM_GPUS}"

     #echo \
     $BIN_PREFIX$APP_NAME \
     $DATA_PREFIX/data.bin \
     $DATA_PREFIX/pattern.bin \
     "$OUTPUT_DIR/$JSON_FILE.json" \
     > "$OUTPUT_DIR/$JSON_FILE.output.txt"
#done
