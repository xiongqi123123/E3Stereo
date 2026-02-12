"""
● 目的：
  生成 HTML 可视化报告，对比 baseline 模型和改进模型（ours）的性能

  输入：
  - 2个模型 checkpoint：--ckpt_base（baseline）和 --ckpt_our（改进版）
  - SceneFlow 数据集路径
  - 采样数量（默认10个样本）

  输出：
  - web_vis/images/ 目录：
    - {id}_left.jpg：左图
    - {id}_gt.jpg：GT 视差图
    - {id}_base.jpg：baseline 预测（带 EPE/D1 标注）
    - {id}_our.jpg：改进模型预测（带 EPE/D1 标注）
    - {id}_err_base.jpg：baseline 误差图
    - {id}_err_our.jpg：改进模型误差图
  - web_vis/index.html：交互式 HTML 报告，展示对比结果和统计指标

"""

import time
import os
import argparse
import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from glob import glob
import os.path as osp
from tqdm import tqdm
from IGEV.igev_stereo import IGEVStereo
from IGEV.utils import InputPadder
import IGEV.stereo_datasets as datasets


# ---------------- 工具函数 ----------------
def extract_edges_rgb(img_rgb, low_thresh=50, high_thresh=150):
    """从RGB图像提取边缘"""
    if isinstance(img_rgb, torch.Tensor):
        img_rgb = img_rgb.cpu().numpy()
    img_rgb = np.squeeze(img_rgb)
    if img_rgb.ndim == 3 and img_rgb.shape[0] == 3:
        img_rgb = img_rgb.transpose(1, 2, 0)
    gray = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    return edges


def extract_edges_disparity(disp, low_thresh=3, high_thresh=10):
    """从视差图提取边缘"""
    if isinstance(disp, torch.Tensor):
        disp_np = disp.detach().cpu().numpy()
    else:
        disp_np = disp
    disp_np = np.squeeze(disp_np)
    if disp_np.ndim == 3:
        disp_np = disp_np[0]
    disp_np = np.clip(disp_np, 0, 192)
    disp_8u = (disp_np / 192 * 255).astype(np.uint8)
    edges = cv2.Canny(disp_8u, low_thresh, high_thresh)
    return edges


def disp_to_color(disp, valid_mask=None, max_disp=192):
    """将视差图转换为彩色热力图

    Args:
        disp: 视差图
        valid_mask: 可选的有效像素mask，如果提供则只显示有效区域
        max_disp: 最大视差值，用于归一化
    """
    if isinstance(disp, torch.Tensor):
        disp_np = disp.detach().cpu().numpy()
    else:
        disp_np = disp
    if isinstance(valid_mask, torch.Tensor):
        valid_mask = valid_mask.detach().cpu().numpy()

    # 修复：移除多余维度，确保是 (H, W)
    disp_np = np.squeeze(disp_np)
    # 如果 squeeze 后还是 3 维 (例如原本是 1,1,H,W)，则强制取索引
    if disp_np.ndim == 3:
        disp_np = disp_np[0]
    if valid_mask is not None:
        valid_mask = np.squeeze(valid_mask)
        if valid_mask.ndim == 3:
            valid_mask = valid_mask[0]

    disp_np = np.clip(disp_np, 0, max_disp)
    norm_disp = (disp_np / max_disp * 255).astype(np.uint8)
    disp_color = cv2.applyColorMap(norm_disp, cv2.COLORMAP_MAGMA)

    # 如果提供了valid mask，将无效区域设置为黑色
    if valid_mask is not None:
        disp_color[~valid_mask] = [0, 0, 0]

    return disp_color


def error_to_color(disp_pred, disp_gt, valid_mask=None, max_err=5.0):
    """将误差图转换为彩色热力图 (Error Map)

    Args:
        disp_pred: 预测视差图
        disp_gt: GT视差图
        valid_mask: 有效像素mask (True表示有GT标注)
        max_err: 最大误差值，用于归一化
    """
    if isinstance(disp_pred, torch.Tensor):
        disp_pred = disp_pred.detach().cpu().numpy()
    if isinstance(disp_gt, torch.Tensor):
        disp_gt = disp_gt.detach().cpu().numpy()
    if isinstance(valid_mask, torch.Tensor):
        valid_mask = valid_mask.detach().cpu().numpy()

    # 修复：移除多余维度
    disp_pred = np.squeeze(disp_pred)
    disp_gt = np.squeeze(disp_gt)
    if valid_mask is not None:
        valid_mask = np.squeeze(valid_mask)

    if disp_pred.ndim == 3: disp_pred = disp_pred[0]
    if disp_gt.ndim == 3: disp_gt = disp_gt[0]
    if valid_mask is not None and valid_mask.ndim == 3: valid_mask = valid_mask[0]

    # 计算误差
    err = np.abs(disp_pred - disp_gt)

    # 应用valid mask：只在有GT的区域显示误差
    if valid_mask is not None:
        err_masked = np.where(valid_mask, err, 0)  # 无效区域设为0
    else:
        err_masked = err

    err_vis = np.clip(err_masked, 0, max_err)
    norm_err = (err_vis / max_err * 255).astype(np.uint8)
    err_color = cv2.applyColorMap(norm_err, cv2.COLORMAP_JET)

    # 将无效区域设置为黑色
    if valid_mask is not None:
        err_color[~valid_mask] = [0, 0, 0]

    return err_color


def write_text(img, text):
    """在图片左上角写入文字"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 确保图片是可写的 contiguous array
    img = np.ascontiguousarray(img)
    cv2.putText(img, text, (10, 30), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


# ---------------- 主逻辑 ----------------
def run_comparison(args):
    # 1. 确定设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"正在使用设备: {device}")

    # 2. 准备目录
    save_dir = args.save_dir
    img_dir = os.path.join(save_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    # 3. 加载模型
    def load_model(model_path, args, device):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型不存在: {model_path}")

        print(f"正在加载模型: {model_path} ...")
        model = IGEVStereo(args)
        state_dict = torch.load(model_path, map_location=device)

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(device)
        model.eval()

        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        return model

    model_base = load_model(args.ckpt_base, args, device)
    model_our = load_model(args.ckpt_our, args, device)

    # 4. 加载数据
    print(f"正在加载数据集: {args.dataset_type} from {args.dataset_root}")

    if args.dataset_type == 'sceneflow':
        val_dataset = datasets.SceneFlowDatasets(
            root=args.dataset_root,
            dstype='frames_finalpass',
            things_test=True
        )
    elif args.dataset_type == 'kitti':
        val_dataset = datasets.KITTI(
            aug_params=None,
            root=args.dataset_root,
            image_set='training'
        )
    elif args.dataset_type == 'kitti2012':
        # 只加载KITTI 2012
        val_dataset = datasets.StereoDataset(aug_params=None, sparse=True, reader=datasets.frame_utils.readDispKITTI)
        root_12 = osp.join(args.dataset_root, 'KITTI_2012') if 'KITTI_2012' not in args.dataset_root else args.dataset_root
        image1_list = sorted(glob(os.path.join(root_12, 'training', 'colored_0/*_10.png')))
        image2_list = sorted(glob(os.path.join(root_12, 'training', 'colored_1/*_10.png')))
        disp_list = sorted(glob(os.path.join(root_12, 'training', 'disp_occ/*_10.png')))
        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            val_dataset.image_list.append([img1, img2])
            val_dataset.disparity_list.append(disp)
    elif args.dataset_type == 'kitti2015':
        # 只加载KITTI 2015
        val_dataset = datasets.StereoDataset(aug_params=None, sparse=True, reader=datasets.frame_utils.readDispKITTI)
        root_15 = osp.join(args.dataset_root, 'KITTI_2015') if 'KITTI_2015' not in args.dataset_root else args.dataset_root
        image1_list = sorted(glob(os.path.join(root_15, 'training', 'image_2/*_10.png')))
        image2_list = sorted(glob(os.path.join(root_15, 'training', 'image_3/*_10.png')))
        disp_list = sorted(glob(os.path.join(root_15, 'training', 'disp_occ_0/*_10.png')))
        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            val_dataset.image_list.append([img1, img2])
            val_dataset.disparity_list.append(disp)
    elif args.dataset_type == 'middlebury':
        val_dataset = datasets.Middlebury(
            aug_params=None,
            root=args.dataset_root,
            split='H'  # Full resolution
        )
    elif args.dataset_type == 'eth3d':
        val_dataset = datasets.ETH3D(
            aug_params=None,
            root=args.dataset_root,
            split='training'
        )
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")

    total_len = len(val_dataset)
    if total_len == 0:
        raise ValueError(f"数据集为空！请检查路径: {args.dataset_root}")

    sample_size = min(args.num_samples, total_len)
    np.random.seed(42)
    indices = np.random.choice(total_len, sample_size, replace=False)

    results = []

    # 用于收集所有像素的误差（pixel-level）
    all_errors_base = []
    all_errors_our = []
    all_errors_base_edge = []
    all_errors_our_edge = []
    all_errors_base_flat = []
    all_errors_our_flat = []

    print(f"开始评估 {sample_size} 个样本...")
    for idx in tqdm(indices, desc="处理样本"):
        _, image1, image2, flow_gt, valid_gt = val_dataset[idx]

        image1 = image1[None].to(device)
        image2 = image2[None].to(device)
        gt = flow_gt.to(device)
        valid_gt = valid_gt.to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1_pad, image2_pad = padder.pad(image1, image2)

        with torch.no_grad():
            disp_base = model_base(image1_pad, image2_pad, iters=32, test_mode=True)
            disp_base = padder.unpad(disp_base).squeeze(0)

            disp_our = model_our(image1_pad, image2_pad, iters=32, test_mode=True)
            disp_our = padder.unpad(disp_our).squeeze(0)

        # Extract edges
        img_vis = image1[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        edge_rgb = extract_edges_rgb(img_vis)
        edge_gt = extract_edges_disparity(gt)

        # Dilate edges to create edge regions
        kernel = np.ones((5, 5), np.uint8)
        edge_mask_rgb = cv2.dilate(edge_rgb, kernel, iterations=1) > 0
        edge_mask_gt = cv2.dilate(edge_gt, kernel, iterations=1) > 0
        edge_mask = edge_mask_rgb | edge_mask_gt

        # Convert edge mask to tensor
        edge_mask_tensor = torch.from_numpy(edge_mask).to(device)
        flat_mask = ~edge_mask_tensor

        # Metrics
        mask = (valid_gt > 0.5) & (gt < 192)

        # Base Metrics
        diff_base = torch.abs(disp_base - gt)
        epe_base_mean = diff_base[mask].mean().item() if mask.sum() > 0 else 0.0
        epe_base_edge = diff_base[mask & edge_mask_tensor].mean().item() if (mask & edge_mask_tensor).sum() > 0 else 0.0
        epe_base_flat = diff_base[mask & flat_mask].mean().item() if (mask & flat_mask).sum() > 0 else 0.0
        d1_base = (diff_base[mask] > 1.0).float().mean().item() * 100 if mask.sum() > 0 else 0.0
        d3_base = (diff_base[mask] > 3.0).float().mean().item() * 100 if mask.sum() > 0 else 0.0
        d5_base = (diff_base[mask] > 5.0).float().mean().item() * 100 if mask.sum() > 0 else 0.0

        # Our Metrics
        diff_our = torch.abs(disp_our - gt)
        epe_our_mean = diff_our[mask].mean().item() if mask.sum() > 0 else 0.0
        epe_our_edge = diff_our[mask & edge_mask_tensor].mean().item() if (mask & edge_mask_tensor).sum() > 0 else 0.0
        epe_our_flat = diff_our[mask & flat_mask].mean().item() if (mask & flat_mask).sum() > 0 else 0.0
        d1_our = (diff_our[mask] > 1.0).float().mean().item() * 100 if mask.sum() > 0 else 0.0
        d3_our = (diff_our[mask] > 3.0).float().mean().item() * 100 if mask.sum() > 0 else 0.0
        d5_our = (diff_our[mask] > 5.0).float().mean().item() * 100 if mask.sum() > 0 else 0.0

        # Collect pixel-level errors for global statistics
        if mask.sum() > 0:
            all_errors_base.append(diff_base[mask].cpu().numpy())
            all_errors_our.append(diff_our[mask].cpu().numpy())
        if (mask & edge_mask_tensor).sum() > 0:
            all_errors_base_edge.append(diff_base[mask & edge_mask_tensor].cpu().numpy())
            all_errors_our_edge.append(diff_our[mask & edge_mask_tensor].cpu().numpy())
        if (mask & flat_mask).sum() > 0:
            all_errors_base_flat.append(diff_base[mask & flat_mask].cpu().numpy())
            all_errors_our_flat.append(diff_our[mask & flat_mask].cpu().numpy())

        # Visualization
        cv2.imwrite(f"{img_dir}/{idx}_left.jpg", cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
        # GT视差图：只显示有效区域（KITTI数据集GT是稀疏的）
        cv2.imwrite(f"{img_dir}/{idx}_gt.jpg", disp_to_color(gt, valid_mask=mask))
        cv2.imwrite(f"{img_dir}/{idx}_edge_rgb.jpg", edge_rgb)
        cv2.imwrite(f"{img_dir}/{idx}_edge_gt.jpg", edge_gt)

        # Predicted视差图：显示完整预测（不应用mask）
        base_vis = disp_to_color(disp_base)
        base_vis = write_text(base_vis, f"EPE: {epe_base_mean:.2f}")
        cv2.imwrite(f"{img_dir}/{idx}_base.jpg", base_vis)

        our_vis = disp_to_color(disp_our)
        our_vis = write_text(our_vis, f"EPE: {epe_our_mean:.2f}")
        cv2.imwrite(f"{img_dir}/{idx}_our.jpg", our_vis)

        # 创建valid mask用于error map可视化
        # mask = (valid_gt > 0.5) & (gt < 192) 已在第256行定义
        cv2.imwrite(f"{img_dir}/{idx}_err_base.jpg", error_to_color(disp_base, gt, valid_mask=mask))
        cv2.imwrite(f"{img_dir}/{idx}_err_our.jpg", error_to_color(disp_our, gt, valid_mask=mask))

        results.append({
            'id': idx,
            'epe_base_mean': epe_base_mean,
            'epe_base_edge': epe_base_edge,
            'epe_base_flat': epe_base_flat,
            'epe_our_mean': epe_our_mean,
            'epe_our_edge': epe_our_edge,
            'epe_our_flat': epe_our_flat,
            'd1_base': d1_base,
            'd3_base': d3_base,
            'd5_base': d5_base,
            'd1_our': d1_our,
            'd3_our': d3_our,
            'd5_our': d5_our,
        })

    # Generate HTML
    def compute_gain_class(base_val, our_val):
        """计算gain的颜色类别"""
        gain = base_val - our_val
        if gain > 0.05:
            return 'better'
        elif gain < -0.05:
            return 'worse'
        else:
            return 'neutral'

    # Compute image-level average metrics
    avg_metrics_img = {
        'epe_base_mean': np.mean([r['epe_base_mean'] for r in results]),
        'epe_base_edge': np.mean([r['epe_base_edge'] for r in results]),
        'epe_base_flat': np.mean([r['epe_base_flat'] for r in results]),
        'epe_our_mean': np.mean([r['epe_our_mean'] for r in results]),
        'epe_our_edge': np.mean([r['epe_our_edge'] for r in results]),
        'epe_our_flat': np.mean([r['epe_our_flat'] for r in results]),
        'd1_base': np.mean([r['d1_base'] for r in results]),
        'd3_base': np.mean([r['d3_base'] for r in results]),
        'd5_base': np.mean([r['d5_base'] for r in results]),
        'd1_our': np.mean([r['d1_our'] for r in results]),
        'd3_our': np.mean([r['d3_our'] for r in results]),
        'd5_our': np.mean([r['d5_our'] for r in results]),
    }

    # Compute pixel-level global EPE
    avg_metrics_px = {
        'epe_base_overall': np.concatenate(all_errors_base).mean() if len(all_errors_base) > 0 else 0.0,
        'epe_base_edge': np.concatenate(all_errors_base_edge).mean() if len(all_errors_base_edge) > 0 else 0.0,
        'epe_base_flat': np.concatenate(all_errors_base_flat).mean() if len(all_errors_base_flat) > 0 else 0.0,
        'epe_our_overall': np.concatenate(all_errors_our).mean() if len(all_errors_our) > 0 else 0.0,
        'epe_our_edge': np.concatenate(all_errors_our_edge).mean() if len(all_errors_our_edge) > 0 else 0.0,
        'epe_our_flat': np.concatenate(all_errors_our_flat).mean() if len(all_errors_our_flat) > 0 else 0.0,
    }

    html = f"""
    <html><head>
    <meta charset="UTF-8">
    <style>
    body {{font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5;}}
    .summary {{background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}}
    .controls {{background: white; padding: 15px; margin-bottom: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}}
    .controls label {{margin-right: 10px; font-weight: bold;}}
    .controls select {{padding: 5px; border-radius: 4px; margin-right: 20px;}}
    table {{border-collapse: collapse; width: 100%; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}}
    th {{background: #4CAF50; color: white; padding: 10px; text-align: center; position: sticky; top: 0; font-size: 14px;}}
    td {{border: 1px solid #ddd; padding: 6px; text-align: center; font-size: 12px;}}
    tr:hover {{background: #f5f5f5;}}
    tr.avg-row {{background: #e3f2fd; font-weight: bold;}}
    img {{width: 180px; cursor: pointer; transition: transform 0.2s;}}
    img:hover {{transform: scale(1.8); z-index: 100; position: relative;}}
    .better {{color: green; font-weight: bold;}}
    .worse {{color: red; font-weight: bold;}}
    .neutral {{color: gray;}}
    .metric-table {{font-size: 11px; line-height: 1.4;}}
    .metric-table td {{padding: 2px; border: none;}}
    </style>
    </head><body>
    <div class="summary">
        <h2>IGEV Baseline vs Ours - Case Study</h2>
        <p>Total Samples: {len(results)}</p>
    </div>
    <div class="controls">
        <label>Sort By:</label>
        <select id="sortSelect" onchange="sortTable()">
            <option value="id">Sample ID</option>
            <option value="epe_base_mean">Base EPE Mean</option>
            <option value="epe_base_edge">Base EPE Edge</option>
            <option value="epe_base_flat">Base EPE Flat</option>
            <option value="diff_mean">Gain EPE Mean</option>
            <option value="diff_edge">Gain EPE Edge</option>
            <option value="diff_flat">Gain EPE Flat</option>
        </select>
    </div>
    <table id="resultsTable">
    <tr>
        <th>ID</th>
        <th>Left Image<br>GT Disparity</th>
        <th>Edge RGB<br>Edge GT</th>
        <th>Base Pred<br>Base Error</th>
        <th>Ours Pred<br>Ours Error</th>
        <th>Metrics</th>
    </tr>
    """

    # Average row
    html += f"""
    <tr class="avg-row">
        <td><b>MEAN</b></td>
        <td colspan="4">Average across {len(results)} samples</td>
        <td>
            <table class="metric-table">
            <tr><td colspan="4"><b>Image-Level EPE</b></td></tr>
            <tr><td></td><td><b>Base</b></td><td><b>Ours</b></td><td><b>Gain</b></td></tr>
            <tr><td>EPE Mean</td><td>{avg_metrics_img['epe_base_mean']:.3f}</td><td>{avg_metrics_img['epe_our_mean']:.3f}</td>
                <td class="{compute_gain_class(avg_metrics_img['epe_base_mean'], avg_metrics_img['epe_our_mean'])}">{avg_metrics_img['epe_base_mean'] - avg_metrics_img['epe_our_mean']:.3f}</td></tr>
            <tr><td>EPE Edge</td><td>{avg_metrics_img['epe_base_edge']:.3f}</td><td>{avg_metrics_img['epe_our_edge']:.3f}</td>
                <td class="{compute_gain_class(avg_metrics_img['epe_base_edge'], avg_metrics_img['epe_our_edge'])}">{avg_metrics_img['epe_base_edge'] - avg_metrics_img['epe_our_edge']:.3f}</td></tr>
            <tr><td>EPE Flat</td><td>{avg_metrics_img['epe_base_flat']:.3f}</td><td>{avg_metrics_img['epe_our_flat']:.3f}</td>
                <td class="{compute_gain_class(avg_metrics_img['epe_base_flat'], avg_metrics_img['epe_our_flat'])}">{avg_metrics_img['epe_base_flat'] - avg_metrics_img['epe_our_flat']:.3f}</td></tr>
            <tr><td colspan="4" style="border-top: 2px solid #666;"><b>Pixel-Level EPE</b></td></tr>
            <tr><td>EPE Overall</td><td>{avg_metrics_px['epe_base_overall']:.3f}</td><td>{avg_metrics_px['epe_our_overall']:.3f}</td>
                <td class="{compute_gain_class(avg_metrics_px['epe_base_overall'], avg_metrics_px['epe_our_overall'])}">{avg_metrics_px['epe_base_overall'] - avg_metrics_px['epe_our_overall']:.3f}</td></tr>
            <tr><td>EPE Edge</td><td>{avg_metrics_px['epe_base_edge']:.3f}</td><td>{avg_metrics_px['epe_our_edge']:.3f}</td>
                <td class="{compute_gain_class(avg_metrics_px['epe_base_edge'], avg_metrics_px['epe_our_edge'])}">{avg_metrics_px['epe_base_edge'] - avg_metrics_px['epe_our_edge']:.3f}</td></tr>
            <tr><td>EPE Flat</td><td>{avg_metrics_px['epe_base_flat']:.3f}</td><td>{avg_metrics_px['epe_our_flat']:.3f}</td>
                <td class="{compute_gain_class(avg_metrics_px['epe_base_flat'], avg_metrics_px['epe_our_flat'])}">{avg_metrics_px['epe_base_flat'] - avg_metrics_px['epe_our_flat']:.3f}</td></tr>
            <tr><td colspan="4" style="border-top: 2px solid #666;"><b>Other Metrics</b></td></tr>
            <tr><td>D1 (%)</td><td>{avg_metrics_img['d1_base']:.2f}</td><td>{avg_metrics_img['d1_our']:.2f}</td>
                <td class="{compute_gain_class(avg_metrics_img['d1_base'], avg_metrics_img['d1_our'])}">{avg_metrics_img['d1_base'] - avg_metrics_img['d1_our']:.2f}</td></tr>
            <tr><td>D3 (%)</td><td>{avg_metrics_img['d3_base']:.2f}</td><td>{avg_metrics_img['d3_our']:.2f}</td>
                <td class="{compute_gain_class(avg_metrics_img['d3_base'], avg_metrics_img['d3_our'])}">{avg_metrics_img['d3_base'] - avg_metrics_img['d3_our']:.2f}</td></tr>
            <tr><td>D5 (%)</td><td>{avg_metrics_img['d5_base']:.2f}</td><td>{avg_metrics_img['d5_our']:.2f}</td>
                <td class="{compute_gain_class(avg_metrics_img['d5_base'], avg_metrics_img['d5_our'])}">{avg_metrics_img['d5_base'] - avg_metrics_img['d5_our']:.2f}</td></tr>
            </table>
        </td>
    </tr>
    """

    # Data rows
    for r in results:
        html += f"""
        <tr data-id="{r['id']}"
            data-epe_base_mean="{r['epe_base_mean']}"
            data-epe_base_edge="{r['epe_base_edge']}"
            data-epe_base_flat="{r['epe_base_flat']}"
            data-diff_mean="{r['epe_base_mean'] - r['epe_our_mean']}"
            data-diff_edge="{r['epe_base_edge'] - r['epe_our_edge']}"
            data-diff_flat="{r['epe_base_flat'] - r['epe_our_flat']}">
            <td><b>{r['id']}</b></td>
            <td>
                <img src='images/{r['id']}_left.jpg'><br>
                <img src='images/{r['id']}_gt.jpg'>
            </td>
            <td>
                <img src='images/{r['id']}_edge_rgb.jpg'><br>
                <img src='images/{r['id']}_edge_gt.jpg'>
            </td>
            <td>
                <img src='images/{r['id']}_base.jpg'><br>
                <img src='images/{r['id']}_err_base.jpg'>
            </td>
            <td>
                <img src='images/{r['id']}_our.jpg'><br>
                <img src='images/{r['id']}_err_our.jpg'>
            </td>
            <td>
                <table class="metric-table">
                <tr><td></td><td><b>Base</b></td><td><b>Ours</b></td><td><b>Gain</b></td></tr>
                <tr><td>EPE Mean</td><td>{r['epe_base_mean']:.3f}</td><td>{r['epe_our_mean']:.3f}</td>
                    <td class="{compute_gain_class(r['epe_base_mean'], r['epe_our_mean'])}">{r['epe_base_mean'] - r['epe_our_mean']:.3f}</td></tr>
                <tr><td>EPE Edge</td><td>{r['epe_base_edge']:.3f}</td><td>{r['epe_our_edge']:.3f}</td>
                    <td class="{compute_gain_class(r['epe_base_edge'], r['epe_our_edge'])}">{r['epe_base_edge'] - r['epe_our_edge']:.3f}</td></tr>
                <tr><td>EPE Flat</td><td>{r['epe_base_flat']:.3f}</td><td>{r['epe_our_flat']:.3f}</td>
                    <td class="{compute_gain_class(r['epe_base_flat'], r['epe_our_flat'])}">{r['epe_base_flat'] - r['epe_our_flat']:.3f}</td></tr>
                <tr><td>D1 (%)</td><td>{r['d1_base']:.2f}</td><td>{r['d1_our']:.2f}</td>
                    <td class="{compute_gain_class(r['d1_base'], r['d1_our'])}">{r['d1_base'] - r['d1_our']:.2f}</td></tr>
                <tr><td>D3 (%)</td><td>{r['d3_base']:.2f}</td><td>{r['d3_our']:.2f}</td>
                    <td class="{compute_gain_class(r['d3_base'], r['d3_our'])}">{r['d3_base'] - r['d3_our']:.2f}</td></tr>
                <tr><td>D5 (%)</td><td>{r['d5_base']:.2f}</td><td>{r['d5_our']:.2f}</td>
                    <td class="{compute_gain_class(r['d5_base'], r['d5_our'])}">{r['d5_base'] - r['d5_our']:.2f}</td></tr>
                </table>
            </td>
        </tr>
        """

    html += """
    </table>
    <script>
    function sortTable() {
        const select = document.getElementById('sortSelect');
        const sortBy = select.value;
        const table = document.getElementById('resultsTable');
        const tbody = table.tBodies[0] || table;
        const rows = Array.from(tbody.querySelectorAll('tr:not(.avg-row)'));

        rows.sort((a, b) => {
            if (sortBy === 'id') {
                return parseInt(a.dataset.id) - parseInt(b.dataset.id);
            } else {
                const aVal = parseFloat(a.dataset[sortBy]) || 0;
                const bVal = parseFloat(b.dataset[sortBy]) || 0;
                return bVal - aVal;  // Descending order
            }
        });

        const avgRow = tbody.querySelector('.avg-row');
        rows.forEach(row => tbody.appendChild(row));
        tbody.insertBefore(avgRow, tbody.firstChild);
    }
    </script>
    </body></html>
    """

    with open(f"{save_dir}/index.html", "w") as f:
        f.write(html)

    print("\n" + "=" * 60)
    print("评估完成！")
    print(f"Web页面: {save_dir}/index.html")
    print("=" * 60)


if __name__ == '__main__':
    """
     1. SceneFlow（默认）                                                                                                                                                     
  python test/01_generate_case_study.py \
      --dataset_type sceneflow \                                                                                                                                           
      --dataset_root ../dataset_cache/SceneFlow \                                                                                                                          
      --save_dir web_vis_sceneflow \
      --num_samples 100

  2. KITTI 2012/2015
  python test/01_generate_case_study.py \
      --dataset_type kitti \
      --dataset_root ../dataset_cache/KITTI \
      --save_dir web_vis_kitti \
      --num_samples 50

  3. Middlebury
  python test/01_generate_case_study.py \
      --dataset_type middlebury \
      --dataset_root ../dataset_cache/Middlebury/MiddEval3 \
      --save_dir web_vis_middlebury \
      --num_samples 15

  4. ETH3D
  python test/01_generate_case_study.py \
      --dataset_type eth3d \
      --dataset_root ../dataset_cache/ETH3D \
      --save_dir web_vis_eth3d \
      --num_samples 27

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='auto', choices=['cpu', 'cuda', 'auto'], help='选择设备')
    parser.add_argument('--ckpt_base', default='../model_cache/sceneflow.pth', help='Baseline路径')

    # parser.add_argument('--ckpt_our', default='../logs/gt_depth_aware/100000_gt_depth_aware_edge_bs6.pth')
    # parser.add_argument('--ckpt_our', default='../logs/edge_d1_26/188500_igev_edge_pt_2_6.pth')
    # parser.add_argument('--ckpt_our', default='../logs/edge_cpt/64000_edge_cpt.pth')
    parser.add_argument('--ckpt_our', default='../logs/our3_211/90000_gt_lr0002.pth')

    parser.add_argument('--dataset_type', default='sceneflow',
                        choices=['sceneflow', 'kitti', 'kitti2012', 'kitti2015', 'middlebury', 'eth3d'],
                        help='数据集类型')
    parser.add_argument('--save_dir',
                        default='web_vis_sceneflow'
                        # default='web_vis_kitti2012'
                        # default='web_vis_kitti2015'
                        # default='web_vis_middlebury'
                        # default='web_vis_eth3d'
                        )
    parser.add_argument('--dataset_root',
                        default='../dataset_cache/SceneFlow',
                        # default='../dataset_cache/KITTI/KITTI_2012',
                        # default='../dataset_cache/KITTI/KITTI_2015',
                        # default='../dataset_cache/Middlebury/Middlebury',
                        # default='../dataset_cache/ETH3D',
                        help='数据路径')
    parser.add_argument('--num_samples', type=int, default=50, help='采样数')

    # IGEV Args
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3)
    parser.add_argument('--corr_levels', type=int, default=2)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--n_gru_layers', type=int, default=3)
    parser.add_argument('--max_disp', type=int, default=192)
    parser.add_argument('--mixed_precision', action='store_true', default=True)
    parser.add_argument('--precision_dtype', default='bfloat16', choices=['float16', 'float32', 'bfloat16'])

    args = parser.parse_args()
    run_comparison(args)