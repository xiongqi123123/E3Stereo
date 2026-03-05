name=ETH3D-ZeroShot
restore_ckpt=/home/qi.xiong/StereoMatching/IGEV-Improve/EStereo/checkpoints/estero-20K-eth3d-ftedge-shared/150_estero-20K-eth3d-ftedge-shared.pth
dataset=eth3d
valid_iters=32

export CUDA_VISIBLE_DEVICES=5

python3 evaluate_stereo.py \
    --name $name \
    --restore_ckpt $restore_ckpt \
    --dataset $dataset \
    --valid_iters $valid_iters \
    --edge_source shared --edge_guided_refinement