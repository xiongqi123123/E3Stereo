name=estero-gt-edge-guided-disp-head-film
restore_ckpt=None
logdir=./checkpoints/$name
batch_size=4
train_datasets=sceneflow
lr=0.0002
num_steps=20000

export CUDA_VISIBLE_DEVICES=3

python train_stereo.py \
    --name $name \
    --logdir $logdir \
    --batch_size $batch_size \
    --train_datasets $train_datasets \
    --lr $lr \
    --num_steps $num_steps \
    --edge_source gt --edge_model None \
    --edge_guided_disp_head --edge_disp_fusion_mode film 

# --edge_source gt --edge_model None                # Edge 来源: 'gt' 使用外部读取的 gt edge (如 gtedge.py 预生成)
# --edge_context_fusion --edge_fusion_mode film       # Edge 融合到 Context
# --edge_guided_upsample --edge_upsample_fusion_mode film  # Edge 引导上采样
# --edge_guided_disp_head --edge_disp_fusion_mode film     # Edge 引导 Disp Head
# --edge_guided_cost_agg --edge_cost_agg_fusion_mode film # Edge 注入 Cost Agg (Hourglass)，优化 init_disp
# --edge_guided_gwc --edge_gwc_fusion_mode film         # Edge 注入 GWC corr_feature_att，边界感知的初始代价
# --edge_motion_encoder --edge_motion_fusion_mode film   # Edge 注入 Motion Encoder，边界感知的 motion 特征
# --edge_guided_refinement --edge_refinement_fusion_mode film  # Edge 引导视差 refinement 后处理，边界锐化
# --boundary_only_refinement --edge_refinement_fusion_mode film  # 仅在边界区域 refinement，平坦区不施加
# --edge_aware_smoothness --edge_smoothness_weight 0.05          # Edge-aware smoothness loss，边界处允许视差突变
# --edge_weight_epe_weight 0.1                                   # Edge-weighted EPE，边界处提高误差权重