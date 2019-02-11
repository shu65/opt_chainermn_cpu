#!/usr/bin/env bash

RESULT_DIR=$1

#Chainer MultiprocessIterator
NUM_ITERATOR_WORKERS=$2

# OpenMP
export OMP_NUM_THREADS=$3


# Run
python train_mnist_with_benchmark.py -g --benchmark -o $RESULT_DIR -j $NUM_ITERATOR_WORKERS