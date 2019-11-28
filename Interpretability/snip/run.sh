#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python Lenet_SNIP.py --datasource mnist --path_data data --arch lenet5 --target_sparsity 0.99 --batch_size 100 --train_iterations 10000 --optimizer adam  2>&1 | tee Lenet_SNIP.txt

