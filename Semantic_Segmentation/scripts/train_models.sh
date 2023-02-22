#!/usr/bin/env bash


export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
export PYTHONPATH=$PYTHONPATH:`pwd`



# config_path='baseline.unet'
# model_dir='./log/train_in_city_osm/unet'
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9990 train.py \
#     --config_path=${config_path} \
#     --model_dir=${model_dir} \
#     train.eval_interval_epoch 5

# config_path='baseline.cenet'
# model_dir='./log/train_in_city_osm/cenet'
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9990 train.py \
#     --config_path=${config_path} \
#     --model_dir=${model_dir} \
#     train.eval_interval_epoch 5

# config_path='baseline.farsegv1'
# model_dir='./log/train_in_city_osm/farsegv1'
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9990 train.py \
#     --config_path=${config_path} \
#     --model_dir=${model_dir} \
#     train.eval_interval_epoch 5

# config_path='baseline.pan'
# model_dir='./log/train_in_city_osm/pan'
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9990 train.py \
#     --config_path=${config_path} \
#     --model_dir=${model_dir} \
#     train.eval_interval_epoch 5

# config_path='baseline.pspnet'
# model_dir='./log/train_in_city_osm/pspnet'
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9990 train.py \
#     --config_path=${config_path} \
#     --model_dir=${model_dir} \
#     train.eval_interval_epoch 5

# config_path='baseline.lanet'
# model_dir='./log/train_in_city_osm/lanet'
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9990 train.py \
#     --config_path=${config_path} \
#     --model_dir=${model_dir} \
#     train.eval_interval_epoch 5

# config_path='baseline.deeplabv3'
# model_dir='./log/train_in_city_osm/deeplabv3'
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9990 train.py \
#     --config_path=${config_path} \
#     --model_dir=${model_dir} \
#     train.eval_interval_epoch 5

# config_path='baseline.manet'
# model_dir='./log/train_in_city_osm/manet'
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9990 train.py \
#     --config_path=${config_path} \
#     --model_dir=${model_dir} \
#     train.eval_interval_epoch 5

# config_path='baseline.factseg'
# model_dir='./log/train_in_city_osm/factseg'
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9990 train.py \
#     --config_path=${config_path} \
#     --model_dir=${model_dir} \
#     train.eval_interval_epoch 5

# config_path='baseline.pspnet_denseppm'
# model_dir='./log/train_in_city_osm/pspnet_denseppm'
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9990 train.py \
#     --config_path=${config_path} \
#     --model_dir=${model_dir} \
#     train.eval_interval_epoch 5