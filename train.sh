name=estero-guided-upsample-gated
restore_ckpt=None
logdir=./checkpoints/$name
batch_size=4
train_datasets=sceneflow
lr=0.0002
num_steps=20000

export CUDA_VISIBLE_DEVICES=5

python train_stereo.py \
    --name $name \
    --logdir $logdir \
    --batch_size $batch_size \
    --train_datasets $train_datasets \
    --lr $lr \
    --num_steps $num_steps \
    --edge_guided_upsample --edge_upsample_fusion_mode gated

# --edge_context_fusion --edge_fusion_mode film       # Edge 融合到 Context
# --edge_guided_upsample --edge_upsample_fusion_mode film  # Edge 引导上采样
# --edge_guided_disp_head --edge_disp_fusion_mode film     # Edge 引导 Disp Head