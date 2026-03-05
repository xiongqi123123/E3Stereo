import os
import argparse
import logging
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from core.igev_stereo import IGEVStereo
from evaluate_stereo import *
import core.stereo_datasets as datasets
import torch.nn.functional as F
import datetime
try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


def edge_aware_smoothness_loss(disp, edge, valid_mask=None):
    """
    Edge-aware smoothness: penalize disparity gradients in flat regions,
    allow discontinuities at edges.

    L = |∂d/∂x| * exp(-α * |e_x|) + |∂d/∂y| * exp(-α * |e_y|)

    α=10 使 edge>0.3 区域的惩罚权重 <0.05，几乎不施加平滑；
    而 edge≈0 的平坦区域权重≈1，完整施加平滑约束。
    """
    disp = disp.float()
    edge = edge.detach().float()
    if edge.shape[-2:] != disp.shape[-2:]:
        edge = F.interpolate(edge, size=disp.shape[-2:], mode='bilinear', align_corners=False)
    edge = edge.clamp(0, 1)

    grad_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    # 取相邻像素 edge 的最大值：只要一侧是边，就抑制平滑惩罚
    edge_x = torch.max(edge[:, :, :, :-1], edge[:, :, :, 1:])
    edge_y = torch.max(edge[:, :, :-1, :], edge[:, :, 1:, :])

    weight_x = torch.exp(-10.0 * edge_x)
    weight_y = torch.exp(-10.0 * edge_y)

    if valid_mask is not None:
        vm = valid_mask.bool()
        valid_x = vm[:, :, :, :-1] & vm[:, :, :, 1:]
        valid_y = vm[:, :, :-1, :] & vm[:, :, 1:, :]
        loss_x = (grad_x * weight_x)[valid_x].mean() if valid_x.any() else grad_x.new_tensor(0.0)
        loss_y = (grad_y * weight_y)[valid_y].mean() if valid_y.any() else grad_y.new_tensor(0.0)
    else:
        loss_x = (grad_x * weight_x).mean()
        loss_y = (grad_y * weight_y).mean()

    return loss_x + loss_y


def sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, loss_gamma=0.9, max_disp=192,
                  edge=None, edge_weight_epe=0.0, edge_smoothness_weight=0.0,
                  refinement_loss_weight=0.0):
    """ Loss function defined over sequence of disp predictions """

    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    mag = torch.sum(disp_gt**2, dim=1).sqrt()
    valid_mask = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert valid_mask.shape == disp_gt.shape, [valid_mask.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid_mask.bool()]).any()

    # Edge-aware pixel weighting: upweight edge regions in L1 loss
    # detach edge to prevent loss gradients from shrinking edge_log_scale
    pixel_weight = None
    if edge is not None and edge_weight_epe > 0:
        edge_w = edge.detach().float()
        if edge_w.shape[-2:] != disp_gt.shape[-2:]:
            edge_w = F.interpolate(edge_w, size=disp_gt.shape[-2:], mode='bilinear', align_corners=False)
        pixel_weight = 1.0 + edge_weight_epe * edge_w.clamp(0, 1)

    init_loss = F.smooth_l1_loss(disp_init_pred, disp_gt, reduction='none')
    if pixel_weight is not None:
        init_loss = init_loss * pixel_weight
    disp_loss += 1.0 * init_loss[valid_mask.bool()].mean()

    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        assert i_loss.shape == valid_mask.shape, [i_loss.shape, valid_mask.shape, disp_gt.shape, disp_preds[i].shape]
        if pixel_weight is not None:
            i_loss = i_loss * pixel_weight
        disp_loss += i_weight * i_loss[valid_mask.bool()].mean()

    # Edge-aware smoothness loss (only on final prediction)
    if edge is not None and edge_smoothness_weight > 0:
        smooth_loss = edge_aware_smoothness_loss(disp_preds[-1], edge, valid_mask)
        disp_loss += edge_smoothness_weight * smooth_loss

    # Refinement auxiliary loss: dedicated extra loss on the final output only.
    # Purpose: counteract gradient dilution (final pred receives only ~7% of sequence
    # loss gradient), giving the refinement network a much stronger training signal.
    # Uses a stronger edge weight (edge_weight_epe * 2) to specifically push boundary quality.
    if refinement_loss_weight > 0:
        ref_loss = (disp_preds[-1] - disp_gt).abs()
        if edge is not None:
            edge_r = edge.detach().float()
            if edge_r.shape[-2:] != disp_gt.shape[-2:]:
                edge_r = F.interpolate(edge_r, size=disp_gt.shape[-2:], mode='bilinear', align_corners=False)
            # Double edge weight vs sequence loss: pull refinement net toward boundary accuracy
            ref_pixel_weight = 1.0 + (edge_weight_epe * 2) * edge_r.clamp(0, 1)
            ref_loss = ref_loss * ref_pixel_weight
        ref_loss_val = ref_loss[valid_mask.bool()].mean()
        disp_loss += refinement_loss_weight * ref_loss_val
        metrics_ref_loss = ref_loss_val.item()
    else:
        metrics_ref_loss = None

    epe_map = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt().unsqueeze(1)  # [B, 1, H, W]
    epe_flat_all = epe_map.view(-1)[valid_mask.view(-1)]

    metrics = {
        'epe': epe_flat_all.mean().item(),
        '1px': (epe_flat_all < 1).float().mean().item(),
        '3px': (epe_flat_all < 3).float().mean().item(),
        '5px': (epe_flat_all < 5).float().mean().item(),
        'epe_edge': 0.0,
        'epe_flat': 0.0,
    }
    if edge is not None and edge_smoothness_weight > 0:
        metrics['smooth_loss'] = smooth_loss.item()
    if metrics_ref_loss is not None:
        metrics['ref_loss'] = metrics_ref_loss

    if edge is not None:
        edge_for_split = edge.detach().float()
        if edge_for_split.shape[-2:] != valid_mask.shape[-2:]:
            edge_for_split = F.interpolate(edge_for_split, size=valid_mask.shape[-2:], mode='bilinear', align_corners=False)

        edge_mask = (edge_for_split > 0.5) & valid_mask
        flat_mask = (~(edge_for_split > 0.5)) & valid_mask

        if edge_mask.any():
            metrics['epe_edge'] = epe_map[edge_mask].view(-1).mean().item()
        if flat_mask.any():
            metrics['epe_flat'] = epe_map[flat_mask].view(-1).mean().item()

    return disp_loss, metrics

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler

class Logger:
    SUM_FREQ = 100
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=args.logdir)

    def _print_training_status(self):
        keys = sorted(self.running_loss.keys())
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in keys]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = " ".join(f"{k}={v:.4f}" for k, v in zip(keys, metrics_data))
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str}{metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=args.logdir)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=args.logdir)

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def _get_model(model):
    """DataParallel 时返回 model.module，否则返回 model"""
    return model.module if hasattr(model, 'module') else model


def train(args):

    accumulate_steps = getattr(args, 'accumulate_steps', 1)
    n_gpus = torch.cuda.device_count()
    # 单卡 + 梯度累积时不用 DataParallel，避免 NCCL 问题
    if n_gpus <= 1 or accumulate_steps > 1:
        model = IGEVStereo(args)
        if accumulate_steps > 1:
            logging.info(f"Gradient accumulation: batch_size={args.batch_size} x {accumulate_steps} = effective batch {args.batch_size * accumulate_steps}")
    else:
        model = nn.DataParallel(IGEVStereo(args))

    train_loader = datasets.fetch_dataloader(args)

    # 冻结 backbone + Stereo 分支，仅微调 edge_head + edge_refine
    if getattr(args, 'freeze_for_edge_finetune', False):
        _get_model(model).freeze_for_edge_finetune()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        logging.info("Freeze backbone+stereo for edge finetune: trainable=%d, frozen=%d", trainable, frozen)

    print("Parameter Count: %d" % count_parameters(model))
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    logger = Logger(model, scheduler)

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt, map_location='cpu', weights_only=False)
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        _get_model(model).load_state_dict(state, strict=True)
        logging.info(f"Done loading checkpoint")
    if getattr(args, 'edge_init_from_geo', None) and getattr(args, 'edge_source', None) == 'shared':
        logging.info("Loading edge_head + edge_refine from GeoEdgeNet...")
        ckpt = torch.load(args.edge_init_from_geo, map_location='cpu')
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        state_to_load = {}
        for k, v in state.items():
            if k.startswith("backbone."):
                continue
            if k.startswith("edge_head."):
                state_to_load[k] = v
            elif k.startswith("refine."):
                state_to_load["edge_refine." + k[7:]] = v
        _get_model(model).load_state_dict(state_to_load, strict=False)
        logging.info(f"Loaded {len(state_to_load)} keys from GeoEdgeNet")
    model.cuda()
    model.train()
    _get_model(model).freeze_bn()  # We keep BatchNorm frozen

    validation_frequency = 5000

    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    global_batch_num = 0
    accum_metrics = {}
    total_batches_seen = 0
    while should_keep_training:

        for i_batch, (meta, *data_blob) in enumerate(tqdm(train_loader)):
            total_batches_seen += 1
            if (i_batch % accumulate_steps) == 0:
                optimizer.zero_grad(set_to_none=True)
            # data_blob: image1, image2, disp_gt, valid；若有 GT edge 则多一维
            if len(data_blob) == 5:
                image1, image2, disp_gt, valid, gt_edge = [x.cuda() if x is not None else None for x in data_blob]
            else:
                image1, image2, disp_gt, valid = [x.cuda() for x in data_blob]
                gt_edge = None

            # 小数据集（如 KITTI ~100 batch/epoch）每 epoch 清缓存，避免显存碎片化 OOM；SceneFlow 每 epoch 也清一次无害
            if i_batch == 0 and total_batches_seen > len(train_loader):
                torch.cuda.empty_cache()
            # 模型输入：仅 edge_source=='gt' 时把数据集 edge 喂给模型；geo/rcf/shared 时用模型内部预测
            left_edge_for_model = gt_edge if (getattr(args, 'edge_source', None) == 'gt' and gt_edge is not None) else None
            assert model.training
            try:
                out = model(image1, image2, iters=args.train_iters, left_edge=left_edge_for_model)
            except torch.OutOfMemoryError:
                logging.error(f"OOM at global_batch={global_batch_num}, i_batch={i_batch}. Skip this batch and clear CUDA cache.")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            if isinstance(out, tuple) and len(out) == 4:
                disp_init_pred, disp_preds, left_edge_out, edge_logits_out = out
            else:
                disp_init_pred, disp_preds, left_edge_out = out
                edge_logits_out = None
            assert model.training
            # metrics 始终用 GT edge 做 epe_edge/epe_flat 的 mask（无 GT 时退化为模型输出的 edge）
            edge_for_metrics = gt_edge if gt_edge is not None else left_edge_out

            loss, metrics = sequence_loss(
                disp_preds, disp_init_pred, disp_gt, valid, max_disp=args.max_disp,
                edge=edge_for_metrics,
                edge_weight_epe=getattr(args, 'edge_weight_epe_weight', 0.0),
                edge_smoothness_weight=getattr(args, 'edge_aware_smoothness_weight', 0.0),
                refinement_loss_weight=getattr(args, 'refinement_loss_weight', 0.0),
            )
            edge_loss_weight = getattr(args, 'edge_loss_weight', 0.0)
            if edge_logits_out is not None and gt_edge is not None and edge_loss_weight > 0:
                gt_e = gt_edge.float()
                if gt_e.dim() == 3:
                    gt_e = gt_e.unsqueeze(1)
                if gt_e.shape[-2:] != edge_logits_out.shape[-2:]:
                    gt_e = F.interpolate(gt_e, size=edge_logits_out.shape[-2:], mode='bilinear', align_corners=False)
                # 数值稳定：float32 + clamp 防止 mixed precision 下 BCE 溢出
                logits_safe = edge_logits_out.float().clamp(-50.0, 50.0)
                if getattr(args, 'edge_loss_valid_mask', False) and valid is not None:
                    vm = valid.unsqueeze(1).float()
                    if vm.shape[-2:] != logits_safe.shape[-2:]:
                        vm = F.interpolate(vm, size=logits_safe.shape[-2:], mode='nearest')
                    loss_per_pixel = F.binary_cross_entropy_with_logits(logits_safe, gt_e.clamp(0, 1), reduction='none')
                    n_valid = vm.sum().clamp(min=1.0)
                    edge_loss = (loss_per_pixel * vm).sum() / n_valid
                else:
                    edge_loss = F.binary_cross_entropy_with_logits(logits_safe, gt_e.clamp(0, 1), reduction='mean')
                loss = loss + edge_loss_weight * edge_loss
                metrics['edge_loss'] = edge_loss.item()
            # NaN 检测：跳过坏 batch，避免梯度污染导致模型崩溃
            loss_val = loss.item()
            if not math.isfinite(loss_val):
                logging.warning(f"Step ~{global_batch_num * accumulate_steps}: NaN/Inf loss (={loss_val}), skipping batch to avoid corruption")
                optimizer.zero_grad(set_to_none=True)
                # 必须显式释放大 tensor 并清缓存，否则下一 batch 会因显存未释放而 OOM
                del out, disp_init_pred, disp_preds, left_edge_out, edge_for_metrics
                if edge_logits_out is not None:
                    del edge_logits_out
                del image1, image2, disp_gt, valid
                if gt_edge is not None:
                    del gt_edge
                torch.cuda.empty_cache()
                continue
            # 梯度累积：loss 按步数缩放，使梯度正确累加
            loss = loss / accumulate_steps
            try:
                scaler.scale(loss).backward()
            except torch.OutOfMemoryError:
                logging.error(f"OOM during backward at global_batch={global_batch_num}, i_batch={i_batch}. Skip this batch and clear CUDA cache.")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            for k, v in metrics.items():
                accum_metrics[k] = accum_metrics.get(k, 0.0) + v
            accum_metrics['_loss'] = accum_metrics.get('_loss', 0.0) + loss_val

            # 每个 micro-batch 都尽早释放大 tensor，避免高水位显存长期驻留导致后续随机 OOM
            del out, disp_init_pred, disp_preds, left_edge_out, edge_for_metrics
            if edge_logits_out is not None:
                del edge_logits_out
            del image1, image2, disp_gt, valid
            if gt_edge is not None:
                del gt_edge

            if (i_batch + 1) % accumulate_steps == 0:
                scaler.unscale_(optimizer)
                params_to_clip = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                if total_steps % 100 == 0:
                    torch.cuda.empty_cache()
                avg_loss = accum_metrics.pop('_loss', 0.0) / accumulate_steps
                avg_metrics = {k: v / accumulate_steps for k, v in accum_metrics.items()}
                logger.writer.add_scalar("live_loss", avg_loss, global_batch_num)
                logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
                global_batch_num += 1
                logger.push(avg_metrics)
                accum_metrics = {}
                total_steps += 1

                if total_steps % validation_frequency == validation_frequency - 1:
                    save_path = Path(args.logdir + '/%d_%s.pth' % (total_steps + 1, args.name))
                    logging.info(f"Saving file {save_path.absolute()}")
                    torch.save(model.state_dict(), save_path)
                    if 'sceneflow' in args.train_datasets:
                        results = validate_sceneflow(_get_model(model), iters=args.valid_iters, args=args)
                    elif 'kitti' in args.train_datasets:
                        results = validate_kitti(_get_model(model), iters=args.valid_iters)
                    elif 'eth3d' in args.train_datasets:
                        results = validate_eth3d(_get_model(model), iters=args.valid_iters, mixed_prec=args.mixed_precision, args=args)
                    else:
                        # 未知数据集：用训练集做验证（取第一个）
                        ds = args.train_datasets[0]
                        if 'sceneflow' in ds:
                            results = validate_sceneflow(_get_model(model), iters=args.valid_iters, args=args)
                        elif 'kitti' in ds:
                            results = validate_kitti(_get_model(model), iters=args.valid_iters)
                        elif 'eth3d' in ds:
                            results = validate_eth3d(_get_model(model), iters=args.valid_iters, mixed_prec=args.mixed_precision, args=args)
                        else:
                            logging.warning(f"Unknown validation dataset '{ds}', skip validation")
                            results = {}
                    logger.write_dict(results)
                    model.train()
                    _get_model(model).freeze_bn()
                    torch.cuda.empty_cache()  # 缓解长时间训练后的显存碎片化 OOM

            if total_steps >= args.num_steps:
                should_keep_training = False
                break

    print("FINISHED TRAINING")
    logger.close()
    PATH = args.logdir + '/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='igev-stereo', help="name your experiment")
    parser.add_argument('--restore_ckpt', default=None, help="load the weights from a specific checkpoint")
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float16', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--logdir', default='./checkpoints/sceneflow', help='the directory to save logs and checkpoints')
    parser.add_argument('--device', type=str, default='cuda:4', help='device to use')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help="batch size used during training.")
    parser.add_argument('--accumulate_steps', type=int, default=1, help="gradient accumulation steps (effective_batch = batch_size * accumulate_steps).")
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=200000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 736], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=22, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32, help='number of disp-field updates during validation forward pass')

    # Architecure choices
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4], help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.2, 0.4], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    
    # Edge augmentation (需配合 edge_model 使用)
    parser.add_argument('--edge_use_scale', action='store_true', help='use edge to scale the edge strength')
    parser.add_argument('--edge_source', type=str, default=None, choices=['rcf', 'gt', 'geo', 'shared'],
                        help="edge source: 'rcf' RCF, 'gt' gtedge, 'geo' GeoEdgeNet(frozen), 'shared' shared backbone+EdgeHead(joint train)")
    parser.add_argument('--edge_model', type=str, default=None,
                        help='path to the edge model (当 edge_source=rcf/geo 且开启任意 edge_* 时必需)')
    parser.add_argument('--edge_init_from_geo', type=str, default=None,
                        help='当 edge_source=shared 时，从此 GeoEdgeNet 加载 edge_head+edge_refine 权重（backbone 不加载，用 stereo 的）')
    parser.add_argument('--edge_loss_weight', type=float, default=0.0,
                        help='当 edge_source=shared 时，edge BCE loss 权重（需数据集提供 gt_edge，0=off）')
    parser.add_argument('--edge_context_fusion', action='store_true',
                        help='fuse edge into context features for GRU input')
    parser.add_argument('--edge_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated'],
                        help='edge-context fusion: concat(hard), film(soft), gated(soft)')
    parser.add_argument('--edge_floor', type=float, default=0.0,
                        help='全局 edge 下限，仅在有边像素上生效，所有 edge 模块共用（0=不限制）')
    parser.add_argument('--edge_context_film_gamma_min', type=float, default=0.0,
                        help='Context FiLM 的 γ 在有边像素上的下限（0=不限制）')
    parser.add_argument('--edge_guided_upsample', action='store_true',
                        help='use edge to guide disparity upsampling for sharper boundaries')
    parser.add_argument('--edge_upsample_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated', 'mlp'],
                        help='edge-guided upsampling fusion: concat/film/gated/mlp')
    parser.add_argument('--edge_guided_disp_head', action='store_true',
                        help='use edge to guide delta_disp prediction in GRU update')
    parser.add_argument('--edge_disp_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated', 'mlp'],
                        help='edge-disp fusion: concat/film/gated/mlp')
    parser.add_argument('--edge_guided_cost_agg', action='store_true',
                        help='inject edge into cost_agg (Hourglass) for better init_disp at boundaries')
    parser.add_argument('--edge_cost_agg_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated'],
                        help='edge-cost_agg fusion: concat/film/gated')
    parser.add_argument('--edge_guided_gwc', action='store_true',
                        help='inject edge into GWC corr_feature_att for boundary-aware initial cost')
    parser.add_argument('--edge_gwc_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated'],
                        help='edge-GWC fusion: concat/film/gated')
    parser.add_argument('--edge_motion_encoder', action='store_true',
                        help='inject edge into Motion Encoder for boundary-aware motion features')
    parser.add_argument('--edge_motion_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated'],
                        help='edge-motion encoder fusion: concat/film/gated')
    parser.add_argument('--edge_guided_refinement', action='store_true',
                        help='edge+RGB guided disparity refinement (dilated conv, ~38K params, only on final output)')
    parser.add_argument('--refinement_loss_weight', type=float, default=0.0,
                        help='dedicated aux loss weight on final output (0=off); '
                             'compensates gradient dilution from sequence loss (~7%% signal). '
                             'Recommended 1.0 when edge_guided_refinement is on.')
    # Edge-aware geo encoding volume (Combined_Geo_Encoding_Volume)
    parser.add_argument('--edge_geo_radius_aware', action='store_true',
                        help='use edge to adaptively shrink sampling radius in geo encoding volume')
    parser.add_argument('--edge_geo_radius_shrink', type=float, default=0.5,
                        help='lambda for shrinking geo sampling radius near edges (0=off, 0.5=moderate)')
    # Feature backbone: Edge-FiLM on x4 feature (left branch)
    parser.add_argument('--feature_edge_x4_film', action='store_true',
                        help='use edge-conditioned FiLM on x4 feature (left image only)')
    parser.add_argument('--feature_edge_x4_film_strength', type=float, default=1.0,
                        help='strength of FiLM modulation on x4 feature')
    # Edge-Aware Loss: 仅在 loss 层面利用 edge，零额外模型参数，零推理开销
    parser.add_argument('--edge_weight_epe_weight', type=float, default=0.0,
                        help='edge-weighted EPE: upweight edge pixels in L1 loss (0=off, recommended 2~5)')
    parser.add_argument('--edge_aware_smoothness_weight', type=float, default=0.0,
                        help='edge-aware smoothness loss weight (0=off, recommended 0.1~1.0)')
    parser.add_argument('--freeze_for_edge_finetune', action='store_true',
                        help='冻结 backbone 和 Stereo 分支，仅微调 edge_head + edge_refine（需 edge_source=shared）')
    args = parser.parse_args()

    torch.manual_seed(666)
    np.random.seed(666)

    Path(args.logdir).mkdir(exist_ok=True, parents=True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        filename=f"{args.logdir}/train{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    train(args)
