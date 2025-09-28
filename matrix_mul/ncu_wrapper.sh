#! /bin/bash

DIR=$(pwd)/ncu_reports

mkdir -p $DIR

ncu \
    --clock-control none \
    --set=full \
    -f \
    -o $DIR/report \
    "$@"