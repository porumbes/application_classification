#!/bin/bash

APP_NAME="main"

BIN_PREFIX="./"
DATA_PREFIX="./data" # assuming hive-gen-data.sh placed data here

declare -A DATA_PATTERN
DATA_PATTERN["rmat18"]="georgiyPattern"
DATA_PATTERN+=(["JohnsHopkins"]="JohnsHopkins")

OUTPUT_DIR=${1:-"eval_mgpu"}
NUM_GPUS=${2:-"1"}
JSON_FILE=""

for key in ${!DATA_PATTERN[@]}; do
     # prepare output json file name with number of gpus for this run
     JSON_FILE="ac__${key}__GPU${NUM_GPUS}"

     #echo \
     $BIN_PREFIX$APP_NAME \
     $DATA_PREFIX/${key}_data.bin \
     $DATA_PREFIX/${DATA_PATTERN[${key}]}_pattern.bin \
     "$OUTPUT_DIR/$JSON_FILE.json" \
     > "$OUTPUT_DIR/$JSON_FILE.output.txt"
done

#for file_name in "${DATA1[@]}"
#do
     # prepare output json file name with number of gpus for this run
#     JSON_FILE="ac__rmat18__GPU${NUM_GPUS}"

     #echo \
#     $BIN_PREFIX$APP_NAME \
#     $DATA_PREFIX/_data.bin \
#     $DATA_PREFIX/_pattern.bin \
#     "$OUTPUT_DIR/$JSON_FILE.json" \
#     > "$OUTPUT_DIR/$JSON_FILE.output.txt"
#done
