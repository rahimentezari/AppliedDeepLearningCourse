#!/bin/bash

# train test
#CUDA_VISIBLE_DEVICES=0 python Lenet_SNIP.py --datasource mnist --path_data data --arch lenet5 --target_sparsity 0.001 --batch_size 100 --train_iterations 10000 --optimizer sgd  2>&1 | tee Lenet_SNIP001.txt

# plot
#CUDA_VISIBLE_DEVICES=0 python Lenet_SNIP_Lime.py --datasource mnist --path_data data --arch lenet5 --target_sparsity 0.001 --batch_size 100 --train_iterations 10000 --optimizer sgd  2>&1 | tee Lenet_SNIP_lime001.txt

# Diff
#CUDA_VISIBLE_DEVICES=0 python Lenet_SNIP.py --datasource mnist --path_data data --arch lenet5 --target_sparsity 0.999 --batch_size 100 --train_iterations 10000 --optimizer sgd  2>&1 | tee Lenet_SNIP999.txt

#CUDA_VISIBLE_DEVICES=0 python Lenet_SNIP.py --datasource mnist --path_data data --arch lenet5 --target_sparsity 0.992 --batch_size 100 --train_iterations 10000 --optimizer sgd  2>&1 | tee Lenet_SNIP992.txt

#CUDA_VISIBLE_DEVICES=0 python Lenet_SNIP.py --datasource mnist --path_data data --arch lenet5 --target_sparsity 0.99 --batch_size 100 --train_iterations 10000 --optimizer sgd  2>&1 | tee Lenet_SNIP99.txt
#CUDA_VISIBLE_DEVICES=0 python Lenet_SNIP_Lime.py --datasource mnist --path_data data --arch lenet5 --target_sparsity 0.99 --batch_size 100 --train_iterations 10000 --optimizer sgd  2>&1 | tee Lenet_SNIP_lime99.txt

#CUDA_VISIBLE_DEVICES=0 python Lenet_SNIP.py --datasource mnist --path_data data --arch lenet5 --target_sparsity 0.50 --batch_size 100 --train_iterations 10000 --optimizer sgd  2>&1 | tee Lenet_SNIP50.txt
#CUDA_VISIBLE_DEVICES=0 python Lenet_SNIP_Lime.py --datasource mnist --path_data data --arch lenet5 --target_sparsity 0.50 --batch_size 100 --train_iterations 10000 --optimizer sgd  2>&1 | tee Lenet_SNIP_lime50.txt

#CUDA_VISIBLE_DEVICES=0 python Lenet_SNIP.py --datasource mnist --path_data data --arch lenet5 --target_sparsity 0.10 --batch_size 100 --train_iterations 10000 --optimizer sgd  2>&1 | tee Lenet_SNIP10.txt
#CUDA_VISIBLE_DEVICES=0 python Lenet_SNIP_Lime.py --datasource mnist --path_data data --arch lenet5 --target_sparsity 0.50 --batch_size 100 --train_iterations 10000 --optimizer sgd  2>&1 | tee Lenet_SNIP_lime10.txt

#CUDA_VISIBLE_DEVICES=0 python Lenet_SNIP_Lime_Diff.py --datasource mnist --path_data data --arch lenet5 --target_sparsity 0.992 --batch_size 100 --train_iterations 10000 --optimizer sgd  2>&1 | tee Lenet_SNIP_lime_diff.txt


CUDA_VISIBLE_DEVICES=0 python Lenet_SNIP_Lime_Diff.py --datasource mnist --path_data data --arch lenet5 --target_sparsity 0.991 --batch_size 100 --train_iterations 10000 --optimizer sgd  2>&1 | tee Lenet_SNIP_lime_diff.txt