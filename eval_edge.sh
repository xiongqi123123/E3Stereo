#!/bin/bash
# 几何边缘模型评估与可视化
# 用法：bash eval_edge.sh

CKPT=/home/qi.xiong/StereoMatching/IGEV-Improve/EStereo/checkpoints/geo-edge-sceneflow-1.5/geo-edge-sceneflow-1.5_best.pth
DATA_ROOT=/home/qi.xiong/StereoMatching/IGEV-Improve/data/sceneflow
OUTPUT_DIR=/home/qi.xiong/StereoMatching/IGEV-Improve/EStereo/eval_edge_vis
NUM_VIS=100
SPLIT=TEST

CUDA_VISIBLE_DEVICES=0 python eval_edge.py \
    --ckpt $CKPT \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --split $SPLIT \
    --num_vis $NUM_VIS \
    --thresh 0.5 \
    --save_metrics 
