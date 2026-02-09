#!/bin/bash

# ============================================================
# Ours_3: Depth-Aware Edge 训练脚本
# 核心改进：RGB Sobel 边缘 + 视差 Sobel 边缘，按训练进度动态融合
# ============================================================

name=depth_aware_edge

# ================= 训练参数 =================
batch_size=4
gradient_accumulation_steps=2
train_iters=22
image_size="320 736"
num_steps=500000

# ================= Depth-Aware Edge 参数 =================
# 融合权重 schedule：
#   初期 (0~10k步)：RGB 0.9 / Disp 0.1  (视差不准，主要信任RGB)
#   线性过渡到...
#   后期 (10k~200k步)：RGB 0.5 / Disp 0.5  (视差变准，充分利用)

python train3.py \
    --name $name \
    --batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --train_iters $train_iters \
    --image_size $image_size \
    --num_steps $num_steps \
    --edge_scale_mode schedule \
    --depth_aware_edge \
    --da_weight_mode schedule \
    --rgb_edge_weight 0.9 \
    --rgb_edge_weight_final 0.5 \
    --disp_edge_weight 0.1 \
    --disp_edge_weight_final 0.5 \
    --da_warmup_steps 10000
