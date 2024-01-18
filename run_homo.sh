#!/bin/bash

cd /home/hyliu/code/SuperPoint-Piang
source activate super 
CUDA_VISIBLE_DEVICES=5 python homo.py > output/homo4.log


# nohup bash ./run_homo.sh &
