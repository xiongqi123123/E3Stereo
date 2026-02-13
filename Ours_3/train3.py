import os
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from igev_stereo import IGEVStereo
from evaluate_stereo import *
import stereo_datasets as datasets
import torch.nn.functional as F

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


def _sobel_normalize(edge_strength):
    """
    快速 batch-wise 归一化边缘强度到 [0, 1]
    Args:
        edge_strength: [B, 1, H, W] 未归一化的梯度幅值
    Returns:
        edge_normalized: [B, 1, H, W] 范围 [0, 1]
    """
    B = edge_strength.shape[0]
    edge_flat = edge_strength.view(B, -1)
    # 保护措施：如果全图是平的（如纯黑GT），max可能为0
    max_val = edge_flat.max(dim=1, keepdim=True)[0]
    upper = max_val * 0.98  # [B, 1]
    upper = upper.view(B, 1, 1, 1)
    edge_clamped = torch.clamp(edge_strength, min=0.0)
    edge_normalized = torch.minimum(edge_clamped, upper) / (upper + 1e-8)
    return edge_normalized


def extract_edge_from_image(image, method='sobel'):
    """
    从 RGB 图像提取边缘强度图
    Args:
        image: [B, 3, H, W] RGB 图像，范围 [0, 255] 或已归一化
        method: 'sobel' 或 'canny'
    Returns:
        edge_map: [B, 1, H, W] 边缘强度图，范围 [0, 1]
    """
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    if gray.max() > 1.0:
        gray = gray / 255.0

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=image.dtype, device=image.device).view(1, 1, 3, 3)

    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    edge_strength = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    return _sobel_normalize(edge_strength)


def extract_disp_edge(disp_map):
    """
    从视差图提取视差梯度边缘
    修改说明：现在主要用于处理 GT，不再需要 detach (虽然加上也没事)

    Args:
        disp_map: [B, 1, H, W] 或 [B, H, W] 视差图 (GT 或 Pred)
    Returns:
        disp_edge: [B, 1, H, W] 视差边缘强度图，范围 [0, 1]
    """
    # 确保维度正确
    if disp_map.dim() == 3:
        disp_map = disp_map.unsqueeze(1)

    # 如果是预测值，为了安全还是 detach；如果是 GT，不需要但也没影响
    disp = disp_map.detach()

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=disp.dtype, device=disp.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=disp.dtype, device=disp.device).view(1, 1, 3, 3)

    grad_x = F.conv2d(disp, sobel_x, padding=1)
    grad_y = F.conv2d(disp, sobel_y, padding=1)
    disp_edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    return _sobel_normalize(disp_edge)


def extract_depth_aware_edge(rgb_edge, disp_gt_edge, rgb_weight=0.7, disp_weight=0.3):
    """
    深度感知边缘：融合 RGB 梯度边缘 + GT 视差梯度边缘
    """
    joint_edge = rgb_weight * rgb_edge + disp_weight * disp_gt_edge
    return joint_edge


def compute_adaptive_weight(edge_map, base_weight=1.0, edge_scale=2.0, error_map=None):
    """
    计算自适应的像素权重
    """
    # 基于边缘强度的连续权重：weight = base + scale * edge_strength
    weight_map = base_weight + edge_scale * edge_map

    # 如果提供了误差图，可以进一步加权（难例挖掘）
    if error_map is not None:
        error_normalized = error_map / (error_map.mean() + 1e-8)
        error_weight = torch.clamp(error_normalized, 0.5, 2.0)
        weight_map = weight_map * error_weight

    return weight_map


class DepthAwareEdgeScheduler:
    """
    控制 rgb_edge_weight / disp_edge_weight 融合比例的调度器
    即使改用 GT，Schedule 依然有用：
    训练初期 CNN 容易学到纹理对齐 (RGB)，后期需要强制它关注几何边缘 (Disp GT)。
    """

    def __init__(self, mode='schedule', rgb_w_init=0.9, rgb_w_final=0.5,
                 disp_w_init=0.1, disp_w_final=0.5, total_steps=200000,
                 warmup_steps=10000):
        self.mode = mode
        self.rgb_w_init = rgb_w_init
        self.rgb_w_final = rgb_w_final
        self.disp_w_init = disp_w_init
        self.disp_w_final = disp_w_final
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def get_weights(self, step):
        if self.mode == 'fixed':
            return self.rgb_w_init, self.disp_w_init

        if step < self.warmup_steps:
            return self.rgb_w_init, self.disp_w_init

        remaining = self.total_steps - self.warmup_steps
        progress = min((step - self.warmup_steps) / max(remaining, 1), 1.0)

        rgb_w = self.rgb_w_init + progress * (self.rgb_w_final - self.rgb_w_init)
        disp_w = self.disp_w_init + progress * (self.disp_w_final - self.disp_w_init)
        return rgb_w, disp_w


class AdaptiveEdgeScale:
    """
    自适应边缘权重缩放因子
    """

    def __init__(self, mode='fixed', init_scale=2.0, min_scale=0.5, max_scale=5.0,
                 warmup_steps=5000, target_ratio=1.5, adjustment_rate=0.01):
        self.mode = mode
        self.current_scale = init_scale
        self.init_scale = init_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.warmup_steps = warmup_steps
        self.target_ratio = target_ratio
        self.adjustment_rate = adjustment_rate

        self.ema_edge_epe = None
        self.ema_flat_epe = None
        self.ema_decay = 0.99

    def get_scale(self, step=None, edge_epe=None, flat_epe=None):
        if self.mode == 'fixed':
            return self.current_scale

        elif self.mode == 'schedule':
            if step < self.warmup_steps:
                progress = step / self.warmup_steps
                self.current_scale = self.min_scale + progress * (self.max_scale - self.min_scale)
            else:
                self.current_scale = self.max_scale
            return self.current_scale

        elif self.mode == 'adaptive':
            if edge_epe is None or flat_epe is None:
                return self.current_scale

            if self.ema_edge_epe is None:
                self.ema_edge_epe = edge_epe
                self.ema_flat_epe = flat_epe
            else:
                self.ema_edge_epe = self.ema_decay * self.ema_edge_epe + (1 - self.ema_decay) * edge_epe
                self.ema_flat_epe = self.ema_decay * self.ema_flat_epe + (1 - self.ema_decay) * flat_epe

            current_ratio = self.ema_edge_epe / (self.ema_flat_epe + 1e-8)

            if current_ratio > self.target_ratio:
                adjustment = self.adjustment_rate * (current_ratio - self.target_ratio)
                self.current_scale = min(self.max_scale, self.current_scale + adjustment)
            else:
                adjustment = self.adjustment_rate * (self.target_ratio - current_ratio)
                self.current_scale = max(self.min_scale, self.current_scale - adjustment)

            return self.current_scale

        return self.current_scale

    def state_dict(self):
        return {
            'current_scale': self.current_scale,
            'ema_edge_epe': self.ema_edge_epe,
            'ema_flat_epe': self.ema_flat_epe,
        }

    def load_state_dict(self, state):
        self.current_scale = state['current_scale']
        self.ema_edge_epe = state['ema_edge_epe']
        self.ema_flat_epe = state['ema_flat_epe']


def sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, rgb_edge_map,
                  loss_gamma=0.9, max_disp=192, edge_scale=2.0, compute_metrics=False,
                  depth_aware=False, rgb_edge_weight=0.7, disp_edge_weight=0.3):
    """
    带深度感知边缘加权的序列损失函数
    【重要修改】现在使用 GT 视差来计算边缘权重，而非使用预测视差。
    """
    n_predictions = len(disp_preds)

    def _get_pixel_weights(edge_map):
        pw = compute_adaptive_weight(edge_map, base_weight=1.0, edge_scale=edge_scale)
        if pw.dim() == 4:
            pw = pw.squeeze(1)
        return pw

    # ========== 核心修改：预计算 GT Edge Map ==========
    # 如果启用 depth_aware，我们使用 disp_gt 计算边缘。
    # 这是一个客观存在的边缘，不会随模型预测变差而消失。
    final_edge_map = rgb_edge_map  # 默认只是 RGB

    if depth_aware:
        # 1. 提取 GT 边缘 (disp_gt is [B, 1, H, W] at this point)
        disp_gt_edge = extract_disp_edge(disp_gt)
        # 2. 融合 RGB 和 GT 边缘 (使用当前步数的权重)
        # 注意：这里计算出的 joint_edge 对本次 batch 的所有 iterations 都是通用的
        final_edge_map = extract_depth_aware_edge(
            rgb_edge_map, disp_gt_edge,
            rgb_weight=rgb_edge_weight, disp_weight=disp_edge_weight
        )

    # 预计算权重 (所有 iteration 共享同一个客观的权重图)
    target_weights = _get_pixel_weights(final_edge_map)

    # Squeeze channel dimension for valid mask computation: [B, 1, H, W] -> [B, H, W]
    disp_gt_2d = disp_gt.squeeze(1)
    valid_2d = valid.squeeze(1)
    mag = disp_gt_2d.abs()  # For single-channel disparity
    # 有效性 Mask：GT 存在且在最大视差范围内
    valid_mask = (valid_2d >= 0.5) & (mag < max_disp)

    # 辅助函数：计算加权 Loss
    def weighted_l1(pred, gt, weights, mask):
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if gt.dim() == 4:
            gt = gt.squeeze(1)
        loss = F.smooth_l1_loss(pred, gt, reduction='none')
        # 在 mask 范围内计算加权 loss
        weighted_loss = (loss * weights)[mask]
        weight_sum = weights[mask].sum()
        return weighted_loss.sum() / (weight_sum + 1e-8)

    disp_loss = 0.0

    # ========== 初始视差 Loss ==========
    # 使用计算好的 target_weights
    disp_loss += 1.0 * weighted_l1(disp_init_pred, disp_gt, target_weights, valid_mask)

    # ========== 迭代 Loss ==========
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)

        pred_i = disp_preds[i]

        # 即使是迭代预测，衡量的标准（权重）也应该是基于 GT 的
        i_loss_val = weighted_l1(pred_i, disp_gt, target_weights, valid_mask)
        disp_loss += i_weight * i_loss_val

    # ========== Metrics Calculation ==========
    if compute_metrics:
        final_pred = disp_preds[-1]
        if final_pred.dim() == 4:
            final_pred = final_pred.squeeze(1)
        gt = disp_gt.squeeze(1) if disp_gt.dim() == 4 else disp_gt

        epe = (final_pred - gt).abs()

        # 使用 final_edge_map (含 GT 信息) 来划分边缘和平坦区域
        edge_threshold = 0.3
        edge_mask_flat = final_edge_map.squeeze(1) if final_edge_map.dim() == 4 else final_edge_map

        is_edge = (edge_mask_flat > edge_threshold) & valid_mask
        is_flat = (edge_mask_flat <= edge_threshold) & valid_mask

        epe_all = epe[valid_mask]
        epe_edge = epe[is_edge] if is_edge.sum() > 0 else epe_all
        epe_flat = epe[is_flat] if is_flat.sum() > 0 else epe_all

        # Pixel-level metrics
        d1 = (epe_all > 3).float().mean().item()
        d3 = (epe_all > 1).float().mean().item()
        d5 = (epe_all > 0.5).float().mean().item()

        # Image-level EPE (mean EPE per image)
        batch_size = final_pred.shape[0]
        image_epe_list = []
        image_epe_edge_list = []
        image_epe_flat_list = []

        for b in range(batch_size):
            b_valid = valid_mask[b]
            if b_valid.sum() > 0:
                image_epe_list.append(epe[b][b_valid].mean().item())

                b_edge = is_edge[b]
                if b_edge.sum() > 0:
                    image_epe_edge_list.append(epe[b][b_edge].mean().item())

                b_flat = is_flat[b]
                if b_flat.sum() > 0:
                    image_epe_flat_list.append(epe[b][b_flat].mean().item())

        import numpy as np
        image_epe = np.mean(image_epe_list) if image_epe_list else 0.0
        image_epe_edge = np.mean(image_epe_edge_list) if image_epe_edge_list else 0.0
        image_epe_flat = np.mean(image_epe_flat_list) if image_epe_flat_list else 0.0

        metrics = {
            # Pixel-level EPE
            'epe': epe_all.mean().item(),
            'epe_edge': epe_edge.mean().item(),
            'epe_flat': epe_flat.mean().item(),
            # Image-level EPE
            'image_epe': image_epe,
            'image_epe_edge': image_epe_edge,
            'image_epe_flat': image_epe_flat,
            # Accuracy metrics
            '1px': (epe_all < 1).float().mean().item(),
            '3px': (epe_all < 3).float().mean().item(),
            '5px': (epe_all < 5).float().mean().item(),
            # D-metrics (error rate)
            'd1': d1,
            'd3': d3,
            'd5': d5,
        }
        return disp_loss, metrics
    else:
        return disp_loss, None


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    if hasattr(args, 'use_constant_lr') and args.use_constant_lr:
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=args.num_steps)
        logging.info("Using ConstantLR")
    else:
        # 获取warmup比例参数，默认0.05 (5%)
        pct_start = getattr(args, 'warmup_pct', 0.05)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            args.lr,
            args.num_steps + 100,
            pct_start=pct_start,
            div_factor=10.0,
            final_div_factor=20.0,
            cycle_momentum=False,
            anneal_strategy='linear'
        )
        warmup_steps = int(args.num_steps * pct_start)
        logging.info(f"Using OneCycleLR: max_lr={args.lr}, pct_start={pct_start} ({warmup_steps} steps warmup)")
    return optimizer, scheduler


class Logger:
    SUM_FREQ = 100

    def __init__(self, model, scheduler, logdir):
        self.model = model
        self.scheduler = scheduler
        self.logdir = logdir
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=logdir)
        self.last_train_metrics = {}

    def _print_training_status(self, edge_scale=None):
        avg_metrics = {k: self.running_loss[k] / Logger.SUM_FREQ for k in self.running_loss}
        lr = self.scheduler.get_last_lr()[0]

        print("\n" + "=" * 80)
        scale_str = f" | EdgeScale: {edge_scale:.2f}" if edge_scale else ""
        print(f"[Ours_3 GT-Edge] Step: {self.total_steps + 1:>6d} | LR: {lr:.2e}{scale_str}")
        print("-" * 80)
        print(
            f"  Pixel EPE - All: {avg_metrics.get('epe', 0):.4f} | Edge: {avg_metrics.get('epe_edge', 0):.4f} | Flat: {avg_metrics.get('epe_flat', 0):.4f}")
        print(
            f"  Image EPE - All: {avg_metrics.get('image_epe', 0):.4f} | Edge: {avg_metrics.get('image_epe_edge', 0):.4f} | Flat: {avg_metrics.get('image_epe_flat', 0):.4f}")
        print(
            f"  Accuracy - 1px: {avg_metrics.get('1px', 0) * 100:.2f}% | 3px: {avg_metrics.get('3px', 0) * 100:.2f}% | 5px: {avg_metrics.get('5px', 0) * 100:.2f}%")
        print(
            f"  D-Metric - D1: {avg_metrics.get('d1', 0) * 100:.2f}% | D3: {avg_metrics.get('d3', 0) * 100:.2f}% | D5: {avg_metrics.get('d5', 0) * 100:.2f}%")
        print("=" * 80)

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.logdir)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics, edge_scale=None):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        self.last_train_metrics = metrics.copy()
        self.last_edge_scale = edge_scale

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ - 1:
            self._print_training_status(edge_scale)
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.logdir)

        for key in results:
            self.writer.add_scalar(f"val/{key}", results[key], self.total_steps)

    def print_train_val_comparison(self, val_results):
        print("\n" + "=" * 80)
        edge_scale_str = f" | EdgeScale: {getattr(self, 'last_edge_scale', 0):.2f}" if hasattr(self,
                                                                                               'last_edge_scale') and self.last_edge_scale else ""
        print(f"[Ours_3 GT-Edge] Step: {self.total_steps:>6d} - Train vs Val Comparison{edge_scale_str}")
        print("=" * 80)

        train_loss = self.last_train_metrics.get('loss', 0)
        train_epe = self.last_train_metrics.get('epe', 0)
        train_d1 = self.last_train_metrics.get('d1', 0) * 100

        # Handle different validation datasets with new key naming
        val_epe = (val_results.get('sceneflow-pix-epe-all') or
                   val_results.get('eth3d-pix-epe-all-all') or
                   val_results.get('middleburyH-pix-epe-all-all') or
                   val_results.get('kitti-epe') or 0)

        val_d1 = (val_results.get('sceneflow-pix-d1') or
                  val_results.get('eth3d-pix-bad1-all') or
                  val_results.get('middleburyH-pix-bad2-all') or
                  val_results.get('kitti-d1') or 0)

        print(f"{'Metric':<15} {'Train':>12} {'Val':>12}")
        print("-" * 40)
        print(f"{'Loss':<15} {train_loss:>12.4f} {'N/A':>12}")
        print(f"{'EPE':<15} {train_epe:>12.4f} {val_epe:>12.4f}")
        print(f"{'Error Rate':<15} {train_d1:>11.2f}% {val_d1:>11.2f}%")
        print("=" * 80 + "\n")

    def close(self):
        self.writer.close()


def train(args):
    model = nn.DataParallel(IGEVStereo(args))
    print("Parameter Count: %d" % count_parameters(model))

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    logger = Logger(model, scheduler, args.logdir)

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")
    model.to(args.device)
    model.train()
    model.module.freeze_bn()

    scaler = GradScaler(enabled=args.mixed_precision)

    edge_scaler = AdaptiveEdgeScale(
        mode=args.edge_scale_mode,
        init_scale=args.edge_scale,
        min_scale=args.edge_scale_min,
        max_scale=args.edge_scale_max,
        warmup_steps=args.edge_warmup_steps,
        target_ratio=args.edge_target_ratio,
        adjustment_rate=args.edge_adjustment_rate
    )

    da_scheduler = DepthAwareEdgeScheduler(
        mode=args.da_weight_mode,
        rgb_w_init=args.rgb_edge_weight, rgb_w_final=args.rgb_edge_weight_final,
        disp_w_init=args.disp_edge_weight, disp_w_final=args.disp_edge_weight_final,
        total_steps=args.num_steps,
        warmup_steps=args.da_warmup_steps
    )
    logging.info(f"Depth-aware edge with GT: {args.depth_aware_edge}, mode: {args.da_weight_mode}")

    total_steps = 0
    global_batch_num = 0
    best_epe = float('inf')

    ckpt_output_dir = Path(args.logdir).parent / 'ckpt_output'
    ckpt_output_dir.mkdir(exist_ok=True, parents=True)

    accumulation_counter = 0
    pbar = tqdm(total=args.num_steps, desc="Training", dynamic_ncols=True)

    while total_steps < args.num_steps:
        # 直接获取 batch_data，不要用 (_, *...)
        for i_batch, batch_data in enumerate(train_loader):
            if total_steps >= args.num_steps:
                break

            # Unpack: first element is file paths (tuple), rest are tensors
            file_paths, image1, image2, disp_gt, valid = batch_data
            # Move only tensors to GPU
            image1, image2, disp_gt, valid = image1.to(args.device), image2.to(args.device), disp_gt.to(args.device), valid.to(args.device)

            # RGB Edge
            edge_map = extract_edge_from_image(image1)

            assert model.training
            disp_init_pred, disp_preds = model(image1, image2, iters=args.train_iters)
            assert model.training

            current_edge_scale = edge_scaler.get_scale(step=total_steps)
            current_rgb_w, current_disp_w = da_scheduler.get_weights(total_steps)

            # 每 LOG_FREQ 步计算一次 metrics
            should_log = (total_steps + 1) % Logger.SUM_FREQ == 0

            # 核心修正：传入 disp_gt, 而不是 disp_preds
            loss, metrics = sequence_loss(
                disp_preds, disp_init_pred, disp_gt, valid, edge_map,
                max_disp=args.max_disp, edge_scale=current_edge_scale,
                compute_metrics=should_log,
                depth_aware=args.depth_aware_edge,
                rgb_edge_weight=current_rgb_w,
                disp_edge_weight=current_disp_w
            )

            if metrics is not None and args.edge_scale_mode == 'adaptive':
                edge_scaler.get_scale(
                    step=total_steps,
                    edge_epe=metrics['epe_edge'],
                    flat_epe=metrics['epe_flat']
                )

            loss = loss / args.gradient_accumulation_steps
            global_batch_num += 1
            scaler.scale(loss).backward()

            accumulation_counter += 1

            if accumulation_counter >= args.gradient_accumulation_steps:
                scaler.unscale_(optimizer)
                if args.train_datasets == 'sceneflow':
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                optimizer.zero_grad()
                accumulation_counter = 0

            pbar.update(1)
            pbar.set_postfix({
                'Loss': f"{loss.item() * args.gradient_accumulation_steps:.3f}",
                'Scale': f"{current_edge_scale:.2f}",
                'RGB/GT': f"{current_rgb_w:.2f}/{current_disp_w:.2f}",
            })

            if metrics is not None:
                metrics['loss'] = loss.item() * args.gradient_accumulation_steps
                logger.push(metrics, current_edge_scale)

            if total_steps > 0 and total_steps % args.valid_freq == 0:
                save_path = Path(args.logdir + '/%d_%s.pth' % (total_steps, args.name))
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save(model.state_dict(), save_path)

                # ================= [FIX START] 多数据集验证逻辑 =================
                results = {}

                # 遍历用户指定的所有验证集
                for val_set in args.eval_datasets:
                    logging.info(f"Validating on {val_set}...")

                    if val_set == 'eth3d':
                        # 确保 eth3d_root 已传入
                        res = validate_eth3d(model.module, iters=args.valid_iters, root=args.eth3d_root,
                                             val_samples=args.val_samples)
                    elif val_set == 'middlebury':
                        # 默认验证 Half 分辨率 (H)，速度较快且能反映泛化性
                        res = validate_middlebury(model.module, iters=args.valid_iters, root=args.middlebury_root,
                                                  split='H', val_samples=args.val_samples)
                    elif val_set == 'kitti':
                        res = validate_kitti(model.module, iters=args.valid_iters, root=args.kitti_root,
                                             val_samples=args.val_samples)
                    elif val_set == 'sceneflow':
                        res = validate_sceneflow(model.module, iters=args.valid_iters, root=args.sceneflow_root,
                                                 val_samples=args.val_samples)

                    # 合并结果字典
                    results.update(res)

                # 记录日志
                logger.write_dict(results)
                logger.print_train_val_comparison(results)

                # ================= 最佳模型保存逻辑 =================
                # 优先级策略：ETH3D Bad1.0 > Middlebury Bad2.0 > KITTI D1 > SceneFlow EPE
                target_metric = float('inf')
                metric_name = "metric"

                if 'eth3d-pix-bad1-all' in results:
                    target_metric = results['eth3d-pix-bad1-all']
                    metric_name = "eth3d_bad1"
                elif 'middleburyH-pix-bad2-all' in results:
                    target_metric = results['middleburyH-pix-bad2-all']
                    metric_name = "midd_bad2"
                elif 'kitti-d1' in results:
                    target_metric = results['kitti-d1']
                    metric_name = "kitti_d1"
                elif 'sceneflow-pix-epe-all' in results:
                    target_metric = results['sceneflow-pix-epe-all']
                    metric_name = "sf_epe"

                if target_metric < best_epe:
                    best_epe = target_metric
                    # 文件名体现是哪个指标最佳
                    best_ckpt_path = ckpt_output_dir / f'{args.name}_best_{metric_name}_{best_epe:.4f}_step{total_steps}.pth'
                    torch.save(model.state_dict(), best_ckpt_path)
                    logging.info(f"✓ New best {metric_name}: {best_epe:.4f}, saved to {best_ckpt_path}")

                model.train()
                model.module.freeze_bn()
                # ================= [FIX END] =================

            total_steps += 1

    pbar.close()
    print("FINISHED TRAINING")
    logger.close()
    PATH = args.logdir + '/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='gt_depth_aware_edge', help="name your experiment")
    parser.add_argument('--restore_ckpt', default=None, help="load the weights from a specific checkpoint")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--warmup_pct', type=float, default=0.05, help="warmup percentage for OneCycleLR (0.05 = 5%)")
    parser.add_argument('--da_weight_mode', type=str, default='fixed', choices=['fixed', 'schedule'])
    parser.add_argument('--use_constant_lr', action='store_true')
    parser.add_argument('--num_steps', type=int, default=200000, help="length of training schedule.")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size used during training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="number of gradient accumulation steps")
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 512],
                        help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=12,
                        help="number of updates to the disparity field in each forward pass.")

    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='bfloat16', choices=['float16', 'bfloat16', 'float32'],
                        help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--logdir', default='/root/autodl-tmp/stereo/logs/our3_211',
                        help='the directory to save logs and checkpoints')

    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # Validation parameters
    parser.add_argument('--valid_freq', type=int, default=1000, help='validation frequency (steps)')
    parser.add_argument('--valid_iters', type=int, default=32,
                        help='number of disp-field updates during validation forward pass')
    # 修改：默认为 0，代表验证全部数据
    parser.add_argument('--val_samples', type=int, default=0, help='number of samples to validate (0 for all)')

    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                        help="hidden state and context dimensions")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4], help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'],
                        help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.2, 0.4],
                        help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')

    parser.add_argument('--edge_scale', type=float, default=1.0, help='Initial edge weight scale factor')
    parser.add_argument('--edge_scale_mode', type=str, default='fixed', choices=['fixed', 'schedule', 'adaptive'])
    parser.add_argument('--edge_scale_min', type=float, default=0.5)
    parser.add_argument('--edge_scale_max', type=float, default=2.0)
    parser.add_argument('--edge_warmup_steps', type=int, default=5000)
    parser.add_argument('--edge_target_ratio', type=float, default=1.5)
    parser.add_argument('--edge_adjustment_rate', type=float, default=0.01)

    parser.add_argument('--depth_aware_edge', action='store_true', default=True, help='Enable GT depth-aware edge')
    parser.add_argument('--no_depth_aware_edge', dest='depth_aware_edge', action='store_false')

    parser.add_argument('--rgb_edge_weight', type=float, default=0.5)
    parser.add_argument('--rgb_edge_weight_final', type=float, default=0.5)
    parser.add_argument('--disp_edge_weight', type=float, default=0.1)
    parser.add_argument('--disp_edge_weight_final', type=float, default=0.1)
    parser.add_argument('--da_warmup_steps', type=int, default=10000)

    parser.add_argument('--dataset_root', type=str, default='/root/autodl-tmp/stereo/dataset_cache',
                        help='root directory for all datasets')
    parser.add_argument('--sceneflow_root', type=str, default=None, help='path to SceneFlow dataset')
    parser.add_argument('--kitti_root', type=str, default=None, help='path to KITTI dataset')

    # [NEW] 添加对独立 KITTI 2012 / 2015 路径的支持
    parser.add_argument('--kitti12_root', type=str, default=None, help='path to KITTI 2012 dataset')
    parser.add_argument('--kitti15_root', type=str, default=None, help='path to KITTI 2015 dataset')

    parser.add_argument('--middlebury_root', type=str, default=None, help='path to Middlebury dataset')
    parser.add_argument('--eth3d_root', type=str, default=None, help='path to ETH3D dataset')

    # [NEW] 添加评估数据集参数，支持多个 (nargs='+')
    parser.add_argument('--eval_datasets', nargs='+', default=['eth3d'],
                        choices=['sceneflow', 'kitti', 'eth3d', 'middlebury'],
                        help='datasets for validation during training')

    args = parser.parse_args()

    if args.sceneflow_root is None:
        args.sceneflow_root = os.path.join(args.dataset_root, 'SceneFlow')
    if args.kitti_root is None:
        args.kitti_root = os.path.join(args.dataset_root, 'KITTI')

    # [NEW] 设置 KITTI 12/15 的默认路径
    if args.kitti12_root is None:
        args.kitti12_root = os.path.join(args.dataset_root, 'KITTI/KITTI_2012')
    if args.kitti15_root is None:
        args.kitti15_root = os.path.join(args.dataset_root, 'KITTI/KITTI_2015')

    if args.middlebury_root is None:
        args.middlebury_root = os.path.join(args.dataset_root, 'Middlebury/MiddEval3')
    if args.eth3d_root is None:
        args.eth3d_root = os.path.join(args.dataset_root, 'ETH3D')

    torch.manual_seed(666)
    np.random.seed(666)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path(args.logdir).mkdir(exist_ok=True, parents=True)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {args.device}")

    train(args)