#!/bin/bash

# 实验名称 (建议带上 proxy 标识)
name=igev_edge_proxy_scratch

# 路径设置
logdir=./checkpoints/$name
# 设为 None 表示从头训练 (验证收敛速度)
restore_ckpt=None

# ================= 核心加速参数 (Proxy Settings) =================
# 显存够的话 Batch 可以开大，但为了稳定建议配合梯度累加
batch_size=4
# 迭代次数减半 (加速核心)
train_iters=12
# 分辨率缩小 (加速核心)
image_size="256 512"
# 只跑 2.5 万步看趋势
num_steps=25000
# 学习率保持标准 (从头训练)
lr=0.0002

# 保持不变的几何参数
max_disp=192

# 显卡设置
export CUDA_VISIBLE_DEVICES=0,1

python train_stereo_edge_weighted_loss.py \
    --name $name \
    --logdir $logdir \
    --restore_ckpt $restore_ckpt \
    --batch_size $batch_size \
    --train_iters $train_iters \
    --image_size $image_size \
    --max_disp $max_disp \
    --num_steps $num_steps \
    --lr $lr \
    --train_datasets sceneflow \
    --edge_scale_mode schedule  # 建议开启 warmup