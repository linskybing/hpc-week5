#! /bin/bash

DIR=$(pwd)/ncu_reports

mkdir -p $DIR

export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK

ncu \
    --clock-control none \
    --set=full \
    -f \
    -o $DIR/rank_$OMPI_COMM_WORLD_RANK \
    "$@"