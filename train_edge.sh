#!/bin/bash
# 几何边缘分支独立训练：复用 IGEV Feature + EdgeHead，在 SceneFlow 上学习几何边缘
# 数据：/home/qi.xiong/StereoMatching/IGEV-Improve/data/sceneflow（需已运行 gtedge.py 生成 gtedge）

name=geo-edge-sceneflow-1-Spt-it3
restore_ckpt= 
logdir=./checkpoints/$name
batch_size=16
lr=0.0001
num_steps=50000

export CUDA_VISIBLE_DEVICES=0

python train_edge.py \
    --name $name \
    --logdir $logdir \
    --data_root /home/qi.xiong/StereoMatching/IGEV-Improve/data/sceneflow \
    --batch_size $batch_size \
    --lr $lr \
    --num_steps $num_steps \
    --pos_weight 2.0 \
    --mixed_precision \
    --eval_freq 5000 \
    --eval_samples 500 \
    --ods_dist_frac 0.0075 \
    --save_freq 10000 \
    --refine_iters 3
