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
    upper = edge_flat.max(dim=1, keepdim=True)[0] * 0.98  # [B, 1]
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


def extract_disp_edge(disp_pred):
    """
    从视差预测提取视差梯度边缘（Depth-Aware Edge 的核心）

    原理：在深度不连续处（如物体边界、遮挡边界），视差值会有剧烈跳变，
    Sobel 算子能检测到这些跳变，形成"视差边缘"。这些边缘恰恰是立体匹配
    中最困难的区域——也是 RGB Sobel 可能遗漏的区域（例如纹理相似但深度不同）。

    关键：使用 .detach() 切断梯度！视差边缘只用于计算 loss 权重，
    不应该让梯度通过权重路径反传（否则模型会学到"让视差梯度变小"的错误信号）。

    Args:
        disp_pred: [B, 1, H, W] 或 [B, H, W] 视差预测
    Returns:
        disp_edge: [B, 1, H, W] 视差边缘强度图，范围 [0, 1]
    """
    disp = disp_pred.detach()
    if disp.dim() == 3:
        disp = disp.unsqueeze(1)

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=disp.dtype, device=disp.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=disp.dtype, device=disp.device).view(1, 1, 3, 3)

    grad_x = F.conv2d(disp, sobel_x, padding=1)
    grad_y = F.conv2d(disp, sobel_y, padding=1)
    disp_edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    return _sobel_normalize(disp_edge)


def extract_depth_aware_edge(rgb_edge, disp_pred, rgb_weight=0.7, disp_weight=0.3):
    """
    深度感知边缘：融合 RGB 梯度边缘 + 视差梯度边缘

    设计理念：
    - RGB 边缘提供稳定的纹理/颜色边界先验（训练全程稳定）
    - 视差边缘捕获真正的深度不连续（随模型训练逐步改善）
    - 二者互补：
      * RGB边缘能检测但非深度边缘的位置（如纹理边界）→ 适度加权
      * 视差边缘能检测但RGB边缘遗漏的位置（如同色不同深度）→ 关键补充
      * 两者都强的位置 → 这是真正的困难边缘，给予最高权重

    Args:
        rgb_edge: [B, 1, H, W] 预计算的RGB边缘图（range [0,1]）
        disp_pred: [B, 1, H, W] 或 [B, H, W] 视差预测
        rgb_weight: RGB边缘的融合权重
        disp_weight: 视差边缘的融合权重
    Returns:
        joint_edge: [B, 1, H, W] 深度感知联合边缘图，范围 [0, 1]
    """
    disp_edge = extract_disp_edge(disp_pred)
    joint_edge = rgb_weight * rgb_edge + disp_weight * disp_edge
    return joint_edge


def compute_adaptive_weight(edge_map, base_weight=1.0, edge_scale=2.0, error_map=None):
    """
    计算自适应的像素权重
    Args:
        edge_map: [B, 1, H, W] 边缘强度图，范围 [0, 1]
        base_weight: 基础权重
        edge_scale: 边缘区域的额外权重缩放
        error_map: [B, 1, H, W] 可选的误差图，用于误差驱动的权重调整
    Returns:
        weight_map: [B, 1, H, W] 像素权重图
    """
    # 基于边缘强度的连续权重：weight = base + scale * edge_strength
    weight_map = base_weight + edge_scale * edge_map

    # 如果提供了误差图，可以进一步加权（难例挖掘）
    if error_map is not None:
        # 误差大的区域给予更高权重（上限为 2 倍）
        error_normalized = error_map / (error_map.mean() + 1e-8)
        error_weight = torch.clamp(error_normalized, 0.5, 2.0)
        weight_map = weight_map * error_weight

    return weight_map


class DepthAwareEdgeScheduler:
    """
    控制 rgb_edge_weight / disp_edge_weight 融合比例的调度器

    为什么不做成可学习参数（nn.Parameter）？
    因为这两个 weight 是 loss 的权重。如果让优化器学习它们，
    优化器会把 disp_edge_weight 学到 0（因为去掉视差边缘后 loss 更低），
    完全违背我们加强边缘监督的目标。
    loss 权重不能被它所加权的同一个 loss 来优化——这是根本性矛盾。

    正确做法：基于训练进度的 schedule。
    直觉：训练初期视差预测很差 → disp_edge 不可靠 → 少用；
          训练后期视差预测变好 → disp_edge 可靠 → 多用。
    """
    def __init__(self, mode='schedule', rgb_w_init=0.9, rgb_w_final=0.5,
                 disp_w_init=0.1, disp_w_final=0.5, total_steps=200000,
                 warmup_steps=10000):
        """
        Args:
            mode: 'fixed' 或 'schedule'
            rgb_w_init / rgb_w_final: RGB 边缘权重的起止值
            disp_w_init / disp_w_final: 视差边缘权重的起止值
            total_steps: 总训练步数（用于计算 schedule 进度）
            warmup_steps: 前 N 步纯 RGB（disp_weight=disp_w_init），之后线性过渡
        """
        self.mode = mode
        self.rgb_w_init = rgb_w_init
        self.rgb_w_final = rgb_w_final
        self.disp_w_init = disp_w_init
        self.disp_w_final = disp_w_final
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def get_weights(self, step):
        """
        返回当前步的 (rgb_weight, disp_weight)
        """
        if self.mode == 'fixed':
            return self.rgb_w_init, self.disp_w_init

        # schedule 模式：warmup 阶段保持初始值，之后线性过渡到 final
        if step < self.warmup_steps:
            return self.rgb_w_init, self.disp_w_init

        # warmup 结束后，线性过渡
        remaining = self.total_steps - self.warmup_steps
        progress = min((step - self.warmup_steps) / max(remaining, 1), 1.0)

        rgb_w = self.rgb_w_init + progress * (self.rgb_w_final - self.rgb_w_init)
        disp_w = self.disp_w_init + progress * (self.disp_w_final - self.disp_w_init)
        return rgb_w, disp_w


class AdaptiveEdgeScale:
    """
    自适应边缘权重缩放因子
    支持三种模式：
    1. fixed: 固定值
    2. schedule: 随训练步数按 schedule 变化
    3. adaptive: 根据 edge_EPE / flat_EPE 比值动态调整
    """
    def __init__(self, mode='fixed', init_scale=2.0, min_scale=0.5, max_scale=5.0,
                 warmup_steps=5000, target_ratio=1.5, adjustment_rate=0.01):
        """
        Args:
            mode: 'fixed', 'schedule', 'adaptive'
            init_scale: 初始 edge_scale
            min_scale: 最小 edge_scale
            max_scale: 最大 edge_scale
            warmup_steps: warmup 步数（schedule 模式）
            target_ratio: 目标 edge_EPE/flat_EPE 比值（adaptive 模式）
            adjustment_rate: 调整速率（adaptive 模式）
        """
        self.mode = mode
        self.current_scale = init_scale
        self.init_scale = init_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.warmup_steps = warmup_steps
        self.target_ratio = target_ratio
        self.adjustment_rate = adjustment_rate

        # 用于 adaptive 模式的滑动平均
        self.ema_edge_epe = None
        self.ema_flat_epe = None
        self.ema_decay = 0.99

    def get_scale(self, step=None, edge_epe=None, flat_epe=None):
        """
        获取当前的 edge_scale
        Args:
            step: 当前训练步数（schedule 模式需要）
            edge_epe: 当前 batch 的边缘 EPE（adaptive 模式需要）
            flat_epe: 当前 batch 的平坦区域 EPE（adaptive 模式需要）
        """
        if self.mode == 'fixed':
            return self.current_scale

        elif self.mode == 'schedule':
            # Warmup: 从 min_scale 线性增加到 max_scale
            if step < self.warmup_steps:
                progress = step / self.warmup_steps
                self.current_scale = self.min_scale + progress * (self.max_scale - self.min_scale)
            else:
                self.current_scale = self.max_scale
            return self.current_scale

        elif self.mode == 'adaptive':
            if edge_epe is None or flat_epe is None:
                return self.current_scale

            # 更新滑动平均
            if self.ema_edge_epe is None:
                self.ema_edge_epe = edge_epe
                self.ema_flat_epe = flat_epe
            else:
                self.ema_edge_epe = self.ema_decay * self.ema_edge_epe + (1 - self.ema_decay) * edge_epe
                self.ema_flat_epe = self.ema_decay * self.ema_flat_epe + (1 - self.ema_decay) * flat_epe

            # 计算当前比值
            current_ratio = self.ema_edge_epe / (self.ema_flat_epe + 1e-8)

            # 根据比值调整 edge_scale
            # 如果 edge_EPE/flat_EPE > target_ratio，边缘需要更多关注，增加 scale
            # 如果 edge_EPE/flat_EPE < target_ratio，边缘已经足够好，减少 scale
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

    相比 Ours（纯 RGB Sobel 边缘）的核心改进：
    当 depth_aware=True 时，每个 GRU 迭代都使用"当前迭代的视差预测"计算视差边缘，
    与 RGB 边缘融合后作为该迭代的 loss 权重。这形成一个自增强回路：
      模型预测越准确 → 视差边缘越精确 → 加权越精准 → 训练信号越有效

    Args:
        disp_preds: 迭代预测的视差列表
        disp_init_pred: 初始视差预测
        disp_gt: GT 视差 [B, 1, H, W]
        valid: 有效像素掩码 [B, H, W]
        rgb_edge_map: RGB 边缘强度图 [B, 1, H, W]，范围 [0, 1]（预计算，所有迭代共享）
        loss_gamma: 损失衰减系数
        max_disp: 最大视差
        edge_scale: 边缘权重缩放因子
        compute_metrics: 是否计算详细metrics
        depth_aware: 是否启用深度感知边缘（Ours_3 核心创新）
        rgb_edge_weight: RGB 边缘在联合边缘中的权重
        disp_edge_weight: 视差边缘在联合边缘中的权重
    """
    n_predictions = len(disp_preds)
    mag = torch.sum(disp_gt ** 2, dim=1).sqrt()
    valid_mask = (valid >= 0.5) & (mag < max_disp)

    def _get_pixel_weights(edge_map):
        """从 edge_map 计算 pixel weights，并 squeeze 到 [B, H, W]"""
        pw = compute_adaptive_weight(edge_map, base_weight=1.0, edge_scale=edge_scale)
        if pw.dim() == 4:
            pw = pw.squeeze(1)
        return pw

    def _get_edge_for_disp(disp_pred):
        """根据是否启用 depth_aware 返回对应的 edge_map"""
        if depth_aware:
            return extract_depth_aware_edge(
                rgb_edge_map, disp_pred,
                rgb_weight=rgb_edge_weight, disp_weight=disp_edge_weight
            )
        else:
            return rgb_edge_map

    # 辅助函数：计算加权 Loss
    def weighted_l1(pred, gt, weights, mask):
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if gt.dim() == 4:
            gt = gt.squeeze(1)
        loss = F.smooth_l1_loss(pred, gt, reduction='none')
        weighted_loss = (loss * weights)[mask]
        weight_sum = weights[mask].sum()
        return weighted_loss.sum() / (weight_sum + 1e-8)

    disp_loss = 0.0

    # ========== 初始视差 Loss ==========
    # 使用 init_disp 的视差边缘 + RGB 边缘
    init_edge = _get_edge_for_disp(disp_init_pred)
    init_weights = _get_pixel_weights(init_edge)
    disp_loss += 1.0 * weighted_l1(disp_init_pred, disp_gt, init_weights, valid_mask)

    # ========== 迭代 Loss ==========
    # 核心改进：每个迭代使用该迭代自己的 depth-aware edge
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)

        pred_i = disp_preds[i]
        if pred_i.dim() == 4:
            pred_i = pred_i.squeeze(1)
        gt = disp_gt.squeeze(1) if disp_gt.dim() == 4 else disp_gt

        # 为当前迭代计算 depth-aware edge map
        iter_edge = _get_edge_for_disp(disp_preds[i])
        iter_weights = _get_pixel_weights(iter_edge)

        i_loss_val = (pred_i - gt).abs()
        weighted_loss = (i_loss_val * iter_weights)[valid_mask]
        weight_sum = iter_weights[valid_mask].sum()
        weighted_loss_val = weighted_loss.sum() / (weight_sum + 1e-8)

        disp_loss += i_weight * weighted_loss_val

    # 只在需要时计算详细 metrics
    if compute_metrics:
        final_pred = disp_preds[-1]
        if final_pred.dim() == 4:
            final_pred = final_pred.squeeze(1)
        gt = disp_gt.squeeze(1) if disp_gt.dim() == 4 else disp_gt

        epe = (final_pred - gt).abs()

        # 使用最终迭代的 depth-aware edge 进行边缘/平坦分类
        final_edge = _get_edge_for_disp(disp_preds[-1])
        edge_threshold = 0.3
        edge_mask_flat = final_edge.squeeze(1) if final_edge.dim() == 4 else final_edge
        is_edge = (edge_mask_flat > edge_threshold) & valid_mask
        is_flat = (edge_mask_flat <= edge_threshold) & valid_mask

        epe_all = epe[valid_mask]
        epe_edge = epe[is_edge] if is_edge.sum() > 0 else epe_all
        epe_flat = epe[is_flat] if is_flat.sum() > 0 else epe_all

        d1 = (epe_all > 3).float().mean().item()
        metrics = {
            'epe': epe_all.mean().item(),
            'epe_edge': epe_edge.mean().item(),
            'epe_flat': epe_flat.mean().item(),
            '1px': (epe_all < 1).float().mean().item(),
            '3px': (epe_all < 3).float().mean().item(),
            '5px': (epe_all < 5).float().mean().item(),
            'd1': d1,
        }
        return disp_loss, metrics
    else:
        return disp_loss, None


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    # 快速验证模式：使用常数LR（适合从预训练权重finetune）
    if hasattr(args, 'use_constant_lr') and args.use_constant_lr:
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=args.num_steps)
        logging.info("Using ConstantLR for quick verification from pretrained weights")
    else:
        # 标准训练模式：使用OneCycleLR
        # 优化学习率调度以避免后期震荡：
        # 1. pct_start=0.05: 前 5% 步数 warmup，剩余 95% 缓慢下降
        # 2. div_factor=10: 初始 lr = max_lr/10 = 2e-5 (更温和的起点)
        # 3. final_div_factor=20: 最终 lr = max_lr/200 = 1e-6 (避免过低)
        # 对于 200k 步训练，学习率在 150k 步时约为 max_lr * 0.5，而非原来的 0.25
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            args.lr,
            args.num_steps+100,
            pct_start=0.05,
            div_factor=10.0,
            final_div_factor=20.0,
            cycle_momentum=False,
            anneal_strategy='linear'
        )
        logging.info(f"Using OneCycleLR: max_lr={args.lr}, pct_start=0.05, "
                    f"initial_lr={args.lr/10:.6f}, final_lr={args.lr/200:.6f}")
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
        self.last_train_metrics = {}  # 保存最近的train metrics用于validation时打印

    def _print_training_status(self, edge_scale=None):
        avg_metrics = {k: self.running_loss[k]/Logger.SUM_FREQ for k in self.running_loss}
        lr = self.scheduler.get_last_lr()[0]

        # 清晰的格式化输出
        print("\n" + "="*80)
        scale_str = f" | EdgeScale: {edge_scale:.2f}" if edge_scale else ""
        print(f"[Ours_3 DepthAwareEdge] Step: {self.total_steps+1:>6d} | LR: {lr:.2e}{scale_str}")
        print("-"*80)
        print(f"  EPE: {avg_metrics.get('epe', 0):.4f} | "
              f"Edge EPE: {avg_metrics.get('epe_edge', 0):.4f} | "
              f"Flat EPE: {avg_metrics.get('epe_flat', 0):.4f}")
        print(f"  1px: {avg_metrics.get('1px', 0)*100:.2f}% | "
              f"3px: {avg_metrics.get('3px', 0)*100:.2f}% | "
              f"5px: {avg_metrics.get('5px', 0)*100:.2f}%")
        if avg_metrics.get('epe_flat', 0) > 0:
            ratio = avg_metrics.get('epe_edge', 0) / avg_metrics.get('epe_flat', 1)
            print(f"  Edge/Flat Ratio: {ratio:.2f}x")
        print("="*80)

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.logdir)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics, edge_scale=None):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        # 保存最近的train metrics用于validation时打印
        self.last_train_metrics = metrics.copy()
        self.last_edge_scale = edge_scale

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status(edge_scale)
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.logdir)

        # 为validation结果添加 "val/" 前缀，便于TensorBoard区分
        for key in results:
            self.writer.add_scalar(f"val/{key}", results[key], self.total_steps)

    def print_train_val_comparison(self, val_results):
        """打印Train和Val的对比结果"""
        print("\n" + "="*80)
        edge_scale_str = f" | EdgeScale: {getattr(self, 'last_edge_scale', 0):.2f}" if hasattr(self, 'last_edge_scale') and self.last_edge_scale else ""
        print(f"[Ours_3 DepthAwareEdge] Step: {self.total_steps:>6d} - Train vs Val Comparison{edge_scale_str}")
        print("="*80)

        # 获取train metrics
        train_loss = self.last_train_metrics.get('loss', 0)
        train_epe = self.last_train_metrics.get('epe', 0)
        train_d1 = self.last_train_metrics.get('d1', 0) * 100  # 转为百分比

        # 获取val metrics (根据dataset不同key不同)
        val_epe = val_results.get('scene-disp-epe', val_results.get('kitti-epe', 0))
        val_d1 = val_results.get('scene-disp-d1', val_results.get('kitti-d1', 0))

        print(f"{'Metric':<15} {'Train':>12} {'Val':>12}")
        print("-"*40)
        print(f"{'Loss':<15} {train_loss:>12.4f} {'N/A':>12}")
        print(f"{'EPE':<15} {train_epe:>12.4f} {val_epe:>12.4f}")
        print(f"{'D1 (>3px)':<15} {train_d1:>11.2f}% {val_d1:>11.2f}%")

        # 如果有edge/flat EPE也打印出来
        if 'epe_edge' in self.last_train_metrics:
            train_epe_edge = self.last_train_metrics.get('epe_edge', 0)
            train_epe_flat = self.last_train_metrics.get('epe_flat', 0)
            val_epe_edge = val_results.get('scene-disp-epe-edge', 0)
            val_epe_flat = val_results.get('scene-disp-epe-flat', 0)
            print(f"{'EPE_edge':<15} {train_epe_edge:>12.4f} {val_epe_edge:>12.4f}")
            print(f"{'EPE_flat':<15} {train_epe_flat:>12.4f} {val_epe_flat:>12.4f}")

        print("="*80 + "\n")

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
    model.module.freeze_bn() # We keep BatchNorm frozen

    scaler = GradScaler(enabled=args.mixed_precision)

    # 创建自适应边缘权重缩放器
    edge_scaler = AdaptiveEdgeScale(
        mode=args.edge_scale_mode,
        init_scale=args.edge_scale,
        min_scale=args.edge_scale_min,
        max_scale=args.edge_scale_max,
        warmup_steps=args.edge_warmup_steps,
        target_ratio=args.edge_target_ratio,
        adjustment_rate=args.edge_adjustment_rate
    )
    logging.info(f"Edge scale mode: {args.edge_scale_mode}, init_scale: {args.edge_scale}")
    # 创建 depth-aware edge 融合比例调度器
    da_scheduler = DepthAwareEdgeScheduler(
        mode=args.da_weight_mode,
        rgb_w_init=args.rgb_edge_weight, rgb_w_final=args.rgb_edge_weight_final,
        disp_w_init=args.disp_edge_weight, disp_w_final=args.disp_edge_weight_final,
        total_steps=args.num_steps,
        warmup_steps=args.da_warmup_steps
    )
    logging.info(f"Depth-aware edge: {args.depth_aware_edge}, mode: {args.da_weight_mode}")
    logging.info(f"  RGB weight: {args.rgb_edge_weight} -> {args.rgb_edge_weight_final}, "
                f"Disp weight: {args.disp_edge_weight} -> {args.disp_edge_weight_final}, "
                f"warmup: {args.da_warmup_steps} steps")

    total_steps = 0
    global_batch_num = 0
    best_epe = float('inf')  # 跟踪最优EPE

    # 创建checkpoint输出目录
    ckpt_output_dir = Path(args.logdir).parent / 'ckpt_output'
    ckpt_output_dir.mkdir(exist_ok=True, parents=True)
    logging.info(f"Best checkpoints will be saved to: {ckpt_output_dir}")

    # 梯度累加相关
    accumulation_counter = 0
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    logging.info(f"Gradient accumulation: {args.gradient_accumulation_steps} steps")
    logging.info(f"Effective batch size: {effective_batch_size} (batch_size {args.batch_size} × {args.gradient_accumulation_steps})")

    # 使用 tqdm 显示总训练步数进度
    pbar = tqdm(total=args.num_steps, desc="Training", dynamic_ncols=True)

    while total_steps < args.num_steps:
        for i_batch, (_, *data_blob) in enumerate(train_loader):
            if total_steps >= args.num_steps:
                break

            image1, image2, disp_gt, valid = [x.to(args.device) for x in data_blob]

            # 从左图提取边缘 map
            edge_map = extract_edge_from_image(image1)

            assert model.training
            disp_init_pred, disp_preds = model(image1, image2, iters=args.train_iters)
            assert model.training

            # 获取当前的 edge_scale
            current_edge_scale = edge_scaler.get_scale(step=total_steps)

            # 获取当前的 depth-aware 融合权重（随训练进度调整）
            current_rgb_w, current_disp_w = da_scheduler.get_weights(total_steps)

            # 判断是否需要计算详细metrics（每LOG_FREQ步计算一次）
            should_log = (total_steps + 1) % Logger.SUM_FREQ == 0

            loss, metrics = sequence_loss(
                disp_preds, disp_init_pred, disp_gt, valid, edge_map,
                max_disp=args.max_disp, edge_scale=current_edge_scale,
                compute_metrics=should_log,
                depth_aware=args.depth_aware_edge,
                rgb_edge_weight=current_rgb_w,
                disp_edge_weight=current_disp_w
            )

            # 如果计算了metrics，更新adaptive edge_scale
            if metrics is not None and args.edge_scale_mode == 'adaptive':
                edge_scaler.get_scale(
                    step=total_steps,
                    edge_epe=metrics['epe_edge'],
                    flat_epe=metrics['epe_flat']
                )

            # 梯度累加：将loss除以累加步数
            loss = loss / args.gradient_accumulation_steps

            global_batch_num += 1
            scaler.scale(loss).backward()

            accumulation_counter += 1

            # 只在累加步数达到时才更新参数
            if accumulation_counter >= args.gradient_accumulation_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                optimizer.zero_grad()
                accumulation_counter = 0

            # 更新进度条（显示原始loss，乘回来）
            pbar.update(1)
            pbar.set_postfix({
                'Loss': f"{loss.item() * args.gradient_accumulation_steps:.3f}",
                'Scale': f"{current_edge_scale:.2f}",
                'RGB/Disp': f"{current_rgb_w:.2f}/{current_disp_w:.2f}",
            })

            # 只在计算了metrics时更新logger（记录原始loss）
            if metrics is not None:
                metrics['loss'] = loss.item() * args.gradient_accumulation_steps
                logger.push(metrics, current_edge_scale)

            if total_steps > 0 and total_steps % args.valid_freq == 0:
                # 定期保存checkpoint
                save_path = Path(args.logdir + '/%d_%s.pth' % (total_steps, args.name))
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save(model.state_dict(), save_path)

                # Validation
                if 'sceneflow' in args.train_datasets:
                    results = validate_sceneflow(model.module, iters=args.valid_iters, root=args.sceneflow_root, val_samples=args.val_samples)
                elif 'kitti' in args.train_datasets:
                    results = validate_kitti(model.module, iters=args.valid_iters, root=args.kitti_root, val_samples=args.val_samples)
                else:
                    raise Exception('Unknown validation dataset.')
                logger.write_dict(results)

                # 打印Train vs Val对比
                logger.print_train_val_comparison(results)

                # 根据EPE保存最优checkpoint
                current_epe = results.get('scene-disp-epe', results.get('kitti-epe', float('inf')))
                if current_epe < best_epe:
                    best_epe = current_epe
                    best_ckpt_path = ckpt_output_dir / f'{args.name}_best_epe{current_epe:.4f}_step{total_steps}.pth'
                    torch.save(model.state_dict(), best_ckpt_path)
                    logging.info(f"✓ New best EPE: {current_epe:.4f}, saved to {best_ckpt_path}")

                model.train()
                model.module.freeze_bn()

            total_steps += 1

    pbar.close()
    print("FINISHED TRAINING")
    logger.close()
    PATH = args.logdir + '/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='depth_aware_edge', help="name your experiment")
    parser.add_argument('--restore_ckpt',
                        #default='/root/autodl-tmp/stereo/model_cache/sceneflow.pth',
                        # default='/root/autodl-tmp/stereo/logs/ckpt_output/igev-stereo_best_epe1.2078_step24500.pth',
                        default=None,
                        help="load the weights from a specific checkpoint")
    parser.add_argument('--lr', type=float, default=0.001, help="max learning rate.")
    parser.add_argument('--use_constant_lr', action='store_true')
    parser.add_argument('--num_steps', type=int, default=500000, help="length of training schedule.")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size used during training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="number of gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)")
    # 标准 SceneFlow 训练尺寸是 [320, 736]。
    # 如果显存吃紧，可以折中改为 [320, 640] 或 [288, 576]，但 256x448 确实太小了。
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 512],
                        help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=12,
                        help="number of updates to the disparity field in each forward pass.")


    parser.add_argument('--mixed_precision', default=True, action='store_false', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='bfloat16', choices=['float16', 'bfloat16', 'float32'],
                        help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--logdir', default='/root/autodl-tmp/stereo/logs/ours3_depth_aware_edge', help='the directory to save logs and checkpoints')

    # Training parameters - 默认快速测试配置
    # (注：根据你实际显存情况调整，能开到 8 最好，开不到就 4 + Gradient Accumulation)
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")
    # Validation parameters
    parser.add_argument('--valid_freq', type=int, default=1000, help='validation frequency (steps)')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of disp-field updates during validation forward pass')
    parser.add_argument('--val_samples', type=int, default=1000, help='number of samples to validate (0 for all)')

    # Architecure choices
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    # 标准设置为 192。这对于 SceneFlow 是必须的。
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4], help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.2, 0.4], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')

    # Edge-weighted loss parameters
    parser.add_argument('--edge_scale', type=float, default=2.0,
                        help='Initial edge weight scale factor. Final weight = 1.0 + edge_scale * edge_strength')
    parser.add_argument('--edge_scale_mode', type=str, default='schedule',
                        choices=['fixed', 'schedule', 'adaptive'],
                        help='Edge scale mode: fixed (constant), schedule (warmup), adaptive (EPE-driven)')
    parser.add_argument('--edge_scale_min', type=float, default=0.5,
                        help='Minimum edge scale (for schedule/adaptive modes)')
    parser.add_argument('--edge_scale_max', type=float, default=2.0,
                        help='Maximum edge scale (for schedule/adaptive modes)')
    parser.add_argument('--edge_warmup_steps', type=int, default=5000,
                        help='Warmup steps for schedule mode')
    parser.add_argument('--edge_target_ratio', type=float, default=1.5,
                        help='Target edge_EPE/flat_EPE ratio for adaptive mode')
    parser.add_argument('--edge_adjustment_rate', type=float, default=0.01,
                        help='Adjustment rate for adaptive mode')

    # Depth-aware edge parameters (Ours_3 核心创新)
    parser.add_argument('--depth_aware_edge', action='store_true', default=True,
                        help='Enable depth-aware edge: combine RGB edge + disparity edge per iteration')
    parser.add_argument('--no_depth_aware_edge', dest='depth_aware_edge', action='store_false',
                        help='Disable depth-aware edge (fallback to RGB-only edge like Ours)')
    # 融合权重调度：训练初期信任 RGB（视差不准），后期逐步引入 disp edge
    parser.add_argument('--da_weight_mode', type=str, default='schedule', choices=['fixed', 'schedule'],
                        help='Depth-aware weight mode: fixed (constant ratio) or schedule (progressive)')
    parser.add_argument('--rgb_edge_weight', type=float, default=0.9,
                        help='Initial RGB edge weight (default 0.9, high because early disp is noisy)')
    parser.add_argument('--rgb_edge_weight_final', type=float, default=0.5,
                        help='Final RGB edge weight after schedule (default 0.5)')
    parser.add_argument('--disp_edge_weight', type=float, default=0.1,
                        help='Initial disparity edge weight (default 0.1, low because early disp is noisy)')
    parser.add_argument('--disp_edge_weight_final', type=float, default=0.5,
                        help='Final disparity edge weight after schedule (default 0.5)')
    parser.add_argument('--da_warmup_steps', type=int, default=10000,
                        help='Steps before starting to transition disp_edge_weight (default 10000)')

    # Dataset paths
    parser.add_argument('--dataset_root', type=str, default='/root/autodl-tmp/stereo/dataset_cache', help='root directory for all datasets')
    parser.add_argument('--sceneflow_root', type=str, default=None, help='path to SceneFlow dataset')
    parser.add_argument('--kitti_root', type=str, default=None, help='path to KITTI dataset')
    parser.add_argument('--middlebury_root', type=str, default=None, help='path to Middlebury dataset')
    parser.add_argument('--eth3d_root', type=str, default=None, help='path to ETH3D dataset')

    args = parser.parse_args()

    # Set default dataset paths if not specified
    if args.sceneflow_root is None:
        args.sceneflow_root = os.path.join(args.dataset_root, 'SceneFlow')
    if args.kitti_root is None:
        args.kitti_root = os.path.join(args.dataset_root, 'KITTI')
    if args.middlebury_root is None:
        args.middlebury_root = os.path.join(args.dataset_root, 'Middlebury/MiddEval3')
    if args.eth3d_root is None:
        args.eth3d_root = os.path.join(args.dataset_root, 'ETH3D')

    torch.manual_seed(666)
    np.random.seed(666)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path(args.logdir).mkdir(exist_ok=True, parents=True)

    # Auto-detect device (GPU if available, otherwise CPU)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {args.device}")

    train(args)
