#!/bin/bash
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export MASTER_ADDR='localhost'
export MASTER_PORT='12355'
export WORLD_SIZE=2

torchrun --nproc_per_node=2 train_masked_blocks.py
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=2 train_masked_blocks.py
