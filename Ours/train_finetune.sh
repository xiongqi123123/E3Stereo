#!/bin/bash

# 实验名称
name=edge_cpt

# 路径设置
logdir=/root/autodl-tmp/stereo/logs/$name
# 【重要】必须指向官方下载的 IGEV 预训练权重
#restore_ckpt=/root/autodl-tmp/stereo/model_cache/sceneflow.pth
restore_ckpt=/root/autodl-tmp/stereo/model_cache/sceneflow.pth

# ================= 全量 SOTA 参数 (Full Settings) =================
# 显存允许的话，建议 Batch=4 (配合梯度累加 2次 = 等效BS 8)
batch_size=4
gradient_accumulation_steps=2
# 恢复标准迭代次数
train_iters=22
# 恢复标准训练分辨率
image_size="320 736"
# 微调只需要跑 2万步左右
num_steps=200000
# 【关键】极小的学习率，防止破坏预训练权重
lr=0.00005

# 保持不变的几何参数
#max_disp=192

# 显卡设置
#export CUDA_VISIBLE_DEVICES=0,1

python train_stereo_edge_weighted_loss.py \
    --name $name \
    --logdir $logdir \
    --restore_ckpt $restore_ckpt \
    --batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps\
    --train_iters $train_iters \
    --image_size $image_size \
    --num_steps $num_steps \
    --lr $lr \
    --edge_scale_mode schedule