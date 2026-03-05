#!/bin/bash
# 推理 SceneFlow TEST 全集，按原文件架构保存到 igev/disparity
# 输出结构: igev/disparity/A/0000/0006.png, igev/disparity/B/0000/0006.png, ...

SCENEFLOW_ROOT="/home/qi.xiong/StereoMatching/IGEV-Improve/data/sceneflow"
INPUT_FOLDER="${SCENEFLOW_ROOT}/frames_finalpass/TEST"
OUTPUT_DIR="${SCENEFLOW_ROOT}/igev/disparity"
CKPT="${1:-/home/qi.xiong/StereoMatching/IGEV-Improve/EStereo/model/sceneflow.pth}"

export CUDA_VISIBLE_DEVICES=0

cd "$(dirname "$0")"
echo "Input:  $INPUT_FOLDER"
echo "Output: $OUTPUT_DIR"
echo "Model:  $CKPT"
python3 demo_imgs.py \
    --folder "$INPUT_FOLDER" \
    --output_directory "$OUTPUT_DIR" \
    --restore_ckpt "$CKPT"

echo "Done. Results saved to $OUTPUT_DIR"
