#!/bin/bash

cd /home/hyliu/code/SuperPoint-Piang
source activate super 
CUDA_VISIBLE_DEVICES=5 python train.py > /home/hyliu/code/log/train1.log


# nohup bash ./run_train.sh &

