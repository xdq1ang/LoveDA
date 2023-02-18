#!/usr/bin/env bash


export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
export PYTHONPATH=$PYTHONPATH:`pwd`



config_path='baseline.unet'
model_dir='./log/train_in_isprs/unet'
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9990 train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    train.eval_interval_epoch 5

config_path='baseline.cenet'
model_dir='./log/train_in_isprs/cenet'
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9990 train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    train.eval_interval_epoch 5

config_path='baseline.farsegv1'
model_dir='./log/train_in_isprs/farsegv1'
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9990 train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    train.eval_interval_epoch 5

config_path='baseline.pan'
model_dir='./log/train_in_isprs/pan'
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9990 train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    train.eval_interval_epoch 5

config_path='baseline.pspnet'
model_dir='./log/train_in_isprs/pspnet'
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9990 train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    train.eval_interval_epoch 5

config_path='baseline.lanet'
model_dir='./log/train_in_isprs/lanet'
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9990 train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    train.eval_interval_epoch 5

config_path='baseline.deeplabv3'
model_dir='./log/train_in_isprs/deeplabv3'
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9990 train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    train.eval_interval_epoch 5

config_path='baseline.manet'
model_dir='./log/train_in_isprs/manet'
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9990 train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    train.eval_interval_epoch 5

config_path='baseline.factseg'
model_dir='./log/train_in_isprs/factseg'
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9990 train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    train.eval_interval_epoch 5

# config_path='baseline.pspnet_denseppm'
# model_dir='./log/train_in_isprs/pspnet_denseppm'
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9990 train.py \
#     --config_path=${config_path} \
#     --model_dir=${model_dir} \
#     train.eval_interval_epoch 5