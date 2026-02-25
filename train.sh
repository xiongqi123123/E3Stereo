name=estero-20W-shared-edgeaware-0.5-2.0-0.3-refinement0.1
restore_ckpt=None
logdir=./checkpoints/$name
batch_size=4
train_datasets=sceneflow
lr=0.0002
num_steps=200000

export CUDA_VISIBLE_DEVICES=3

echo "================================================"
echo "Training $name"
echo "================================================"

python train_stereo.py \
    --name $name \
    --logdir $logdir \
    --batch_size $batch_size \
    --train_datasets $train_datasets \
    --lr $lr \
    --num_steps $num_steps \
    --edge_source shared --edge_model /home/qi.xiong/StereoMatching/IGEV-Improve/EStereo/checkpoints/geo-edge-sceneflow-1-Spt-it3/geo-edge-sceneflow-1-Spt-it3_best.pth \
    --edge_loss_weight 0.5 --edge_weight_epe_weight 2.0 --edge_aware_smoothness_weight 0.3 \
    --edge_guided_refinement --refinement_loss_weight 0.1

    

# --edge_use_scale
# --edge_floor 0.1 --edge_context_film_gamma_min 0.0  # Edge 下限和 FiLM 下限
# --edge_source gt --edge_model None                # Edge 来源: 'gt' 使用外部读取的 gt edge (如 gtedge.py 预生成)
# --edge_context_fusion --edge_fusion_mode film       # Edge 融合到 Context
# --edge_guided_upsample --edge_upsample_fusion_mode film  # Edge 引导上采样
# --edge_guided_disp_head --edge_disp_fusion_mode film     # Edge 引导 Disp Head
# --edge_guided_cost_agg --edge_cost_agg_fusion_mode film # Edge 注入 Cost Agg (Hourglass)，优化 init_disp
# --edge_guided_gwc --edge_gwc_fusion_mode film         # Edge 注入 GWC corr_feature_att，边界感知的初始代价
# --edge_motion_encoder --edge_motion_fusion_mode film   # Edge 注入 Motion Encoder，边界感知的 motion 特征
# --edge_guided_refinement    # Edge 引导视差 refinement 后处理，边界锐化
# --edge_geo_radius_aware --edge_geo_radius_shrink 0.5  # Edge 感知 Geo Encoding Volume 采样半径
# --feature_edge_x4_film --feature_edge_x4_film_strength 1.0  # Edge 感知 x4 特征 FiLM 调制强度