import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "5" 

print(torch.cuda.is_available())
print(torch.cuda.current_device())

# CUDA_VISIBLE_DEVICES=5 python 1.py
