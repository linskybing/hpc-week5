#! /bin/bash

DIR=$(pwd)/nsys_reports

mkdir -p $DIR

export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK

nsys profile \
    -o "$DIR/rank_$OMPI_COMM_WORLD_RANK.nsys-rep" \
    --mpi-impl openmpi \
    --force-overwrite true \
    --trace cuda,mpi,cublas,ucx,osrt \
    $@
