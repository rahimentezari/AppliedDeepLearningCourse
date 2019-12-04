#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 python Lenet_SNIP.py --datasource mnist --path_data data --arch lenet5 --target_sparsity 0.99 --batch_size 100 --train_iterations 10000 --optimizer sgd  2>&1 | tee Lenet_SNIP.txt
#CUDA_VISIBLE_DEVICES=0 python Lenet_SNIP_Lime.py --datasource mnist --path_data data --arch lenet5 --target_sparsity 0.99 --batch_size 100 --train_iterations 10000 --optimizer sgd  2>&1 | tee Lenet_SNIP_lime.txt

#CUDA_VISIBLE_DEVICES=0 python Lenet_SNIP.py --datasource mnist --path_data data --arch lenet5 --target_sparsity 0.50 --batch_size 100 --train_iterations 10000 --optimizer sgd  2>&1 | tee Lenet_SNIP50.txt
#CUDA_VISIBLE_DEVICES=0 python Lenet_SNIP_Lime.py --datasource mnist --path_data data --arch lenet5 --target_sparsity 0.50 --batch_size 100 --train_iterations 10000 --optimizer sgd  2>&1 | tee Lenet_SNIP_lime50.txt

CUDA_VISIBLE_DEVICES=0 python Lenet_SNIP.py --datasource mnist --path_data data --arch lenet5 --target_sparsity 0.10 --batch_size 100 --train_iterations 10000 --optimizer sgd  2>&1 | tee Lenet_SNIP10.txt
#CUDA_VISIBLE_DEVICES=0 python Lenet_SNIP_Lime.py --datasource mnist --path_data data --arch lenet5 --target_sparsity 0.10 --batch_size 100 --train_iterations 10000 --optimizer sgd  2>&1 | tee Lenet_SNIP_lime10.txt