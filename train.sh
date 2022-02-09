#!/bin/bash


python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 20001 train_both.py

