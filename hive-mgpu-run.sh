#!/bin/bash

NUM_GPUS=${1:-16}

PARTITION_NAME=${2:-"dgx2"}
#PARTITION_NAME="batch"

APP_SCRIPT="./hive-ac-test.sh"

OUTPUT_DIR="ac_eval_mgpu/$PARTITION_NAME"
mkdir -p $OUTPUT_DIR

for (( i=1; i<=$NUM_GPUS; i++))
do
    # prepare and run SLURM command
    #SLURM_CMD="srun --cpus-per-gpu 1 -G $i -p $PARTITION_NAME -w $NODE_NAME"

    SLURM_CMD="srun --cpus-per-gpu 1 -G $i -p $PARTITION_NAME -N 1"
    $SLURM_CMD $APP_SCRIPT $OUTPUT_DIR $i &

    #echo "$SLURM_CMD $APP_SCRIPT $OUTPUT_DIR $i"
    #sleep 1
done
