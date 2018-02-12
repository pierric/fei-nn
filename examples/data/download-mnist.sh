#! /bin/bash

DATASET="train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte"

for fn in $DATASET; do
    curl http://yann.lecun.com/exdb/mnist/$fn.gz | gunzip - > $fn
done
