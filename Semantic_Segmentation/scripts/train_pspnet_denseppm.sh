#!/usr/bin/env bash


export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
export PYTHONPATH=$PYTHONPATH:`pwd`

config_path='baseline.pspnet_denseppm'
model_dir='./log/normal_baseline/pspnet_denseppm'

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9990 train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    train.eval_interval_epoch 5