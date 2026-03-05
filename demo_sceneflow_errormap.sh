#!/bin/bash
# 生成 SceneFlow TEST 全集的 Error Map，按原文件架构保存到 igev/errormap
# 输出结构: igev/errormap/A/0000/0006_errormap.png, ...

SCENEFLOW_ROOT="/home/qi.xiong/StereoMatching/IGEV-Improve/data/sceneflow"
INPUT_FOLDER="${SCENEFLOW_ROOT}/frames_finalpass/TEST"
OUTPUT_DIR="${SCENEFLOW_ROOT}/igev/errormap"
CKPT="${1:-/home/qi.xiong/StereoMatching/IGEV-Improve/EStereo/model/sceneflow.pth}"
# 若有 demo_imgs --save_numpy 的 .npy，可指定以跳过推理
PRED_DIR="${2:-}"

export CUDA_VISIBLE_DEVICES=4

cd "$(dirname "$0")"
echo "Input:  $INPUT_FOLDER"
echo "Output: $OUTPUT_DIR"
echo "Model:  $CKPT"
if [ -n "$PRED_DIR" ]; then
    echo "Pred:   $PRED_DIR (.npy)"
    python3 demo_errormap.py --mode folder \
        --folder "$INPUT_FOLDER" \
        --output_directory "$OUTPUT_DIR" \
        --restore_ckpt "$CKPT" \
        --data_root "$SCENEFLOW_ROOT" \
        --pred_dir "$PRED_DIR"
else
    python3 demo_errormap.py --mode folder \
        --folder "$INPUT_FOLDER" \
        --output_directory "$OUTPUT_DIR" \
        --restore_ckpt "$CKPT" \
        --data_root "$SCENEFLOW_ROOT"
fi

echo "Done. Results saved to $OUTPUT_DIR"
