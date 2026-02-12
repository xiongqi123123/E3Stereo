#!/bin/bash

# ============================================================
# Ours_3: GT-Depth-Aware Edge 训练脚本
# 适配硬件: RTX 5090 (32GB)
# 配置: Batch Size 6 * Acc 2 = Effective Batch Size 12
# ============================================================

name=gt_lr0002

# ================= 训练参数修改 =================
# 32GB 显存允许开到 6 (320x736分辨率下)
batch_size=6

# 保持 2，这样有效 Batch Size = 12。
# 大 Batch Size 有助于更稳定的收敛，利用好你的显存优势。
gradient_accumulation_steps=2

train_iters=22
image_size="320 736"

# 由于有效 Batch Size 变大(8->12)，每一步效率更高。
# 30万步相当于之前的 45万步的训练量，非常充足。
num_steps=200000

# ================= 验证参数 =================
valid_freq=5000
val_samples=0

# ================= 学习率参数 =================
lr=0.0002           # 降低学习率到0.0002 (之前0.001太大导致后期崩溃)
warmup_pct=0.05     # warmup前5%步数 (200000*0.05=10000步)

# ================= Depth-Aware Edge 参数 =================
# 保持之前的 GT 策略配置
python train3.py \
    --name $name \
    --batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --train_iters $train_iters \
    --image_size $image_size \
    --num_steps $num_steps \
    --valid_freq $valid_freq \
    --val_samples $val_samples \
    --lr $lr \
    --warmup_pct $warmup_pct \
    --edge_scale_mode fixed \
    --depth_aware_edge \
    --da_weight_mode fixed \
    --rgb_edge_weight 0.5 \
    --disp_edge_weight 0.1