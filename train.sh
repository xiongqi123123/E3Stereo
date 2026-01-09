name=igev-edgecontext-wfusion
restore_ckpt=None
logdir=./checkpoints/$name
batch_size=6
train_datasets=sceneflow
lr=0.0002
num_steps=20000

export CUDA_VISIBLE_DEVICES=0

python train_stereo.py \
    --name $name \
    --logdir $logdir \
    --batch_size $batch_size \
    --train_datasets $train_datasets \
    --lr $lr \
    --num_steps $num_steps