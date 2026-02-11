"""
 - 目的: 使用IGEV模型推理视差，计算误差图和EPE统计（整体/边缘区域/平坦区域）
  - Edge regions mask: 通过GT视差的Sobel梯度计算边缘强度，edge_strength > 2.0 的区域为边缘区域（第299-305行）
  - Error map: np.abs(pred_disp - gt_disp)，是预测视差和GT视差的绝对误差（第295行）
  - 不是直接EPE值，EPE是error map在某个区域的平均值（第308-310行）
"""

import sys
sys.path.insert(0, '/root/autodl-tmp/stereo')

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import re
import argparse
from pathlib import Path
from PIL import Image
from collections import namedtuple

from IGEV.igev_stereo import IGEVStereo
from IGEV.utils import InputPadder


def read_pfm(file):
    """读取 PFM 格式的视差图"""
    file = open(file, 'rb')
    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    scale = float(file.readline().decode('utf-8').rstrip())
    endian = '<' if scale < 0 else '>'
    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = np.flipud(np.reshape(data, shape))
    return data, abs(scale)


def load_image(imfile, device):
    """加载图像并转换为模型输入格式"""
    img = Image.open(imfile)
    # 处理 RGBA 图像，转换为 RGB
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)


def load_model(ckpt_path, device):
    """加载 IGEV 模型"""
    # 模型参数配置
    Args = namedtuple('Args', [
        'hidden_dims', 'corr_levels', 'corr_radius',
        'n_downsample', 'n_gru_layers', 'max_disp',
        'mixed_precision', 'precision_dtype'
    ])
    args = Args(
        hidden_dims=[128] * 3,
        corr_levels=2,
        corr_radius=4,
        n_downsample=2,
        n_gru_layers=3,
        max_disp=192,
        mixed_precision=False,
        precision_dtype='float32'
    )

    model = torch.nn.DataParallel(IGEVStereo(args))
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=True)
    model = model.module.to(device)
    model.eval()
    return model


def inference(model, left_img_path, right_img_path, device, iters=32):
    """使用模型进行视差推理"""
    image1 = load_image(left_img_path, device)
    image2 = load_image(right_img_path, device)

    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)

    with torch.no_grad():
        disp = model(image1, image2, iters=iters, test_mode=True)

    disp = padder.unpad(disp)
    return disp.squeeze().cpu().numpy()


def get_sample_from_path(left_img_path):
    """根据左图路径自动推断右图和 GT 路径（支持 SceneFlow 和 KITTI 格式）"""
    left_path = Path(left_img_path)

    if not left_path.exists():
        print(f"错误: 左图不存在: {left_img_path}")
        return None

    # SceneFlow 格式: .../frames_finalpass/TRAIN/A/0130/left/0015.png
    # 右图: .../frames_finalpass/TRAIN/A/0130/right/0015.png
    # GT: .../disparity/TRAIN/A/0130/left/0015.pfm
    if 'frames_finalpass' in str(left_path) or 'frames_cleanpass' in str(left_path):
        right_path = Path(str(left_path).replace('/left/', '/right/'))
        # disparity 路径替换 frames_finalpass 或 frames_cleanpass
        gt_path = Path(str(left_path).replace('frames_finalpass', 'disparity')
                                     .replace('frames_cleanpass', 'disparity')
                                     .replace('.png', '.pfm'))
        sample_id = f"{left_path.parent.parent.name}_{left_path.stem}"

    # KITTI 格式: image_2/000000_10.png -> image_3/..., disp_occ_0/...
    elif 'image_2' in str(left_path):
        right_path = Path(str(left_path).replace('image_2', 'image_3'))
        gt_path = Path(str(left_path).replace('image_2', 'disp_occ_0'))
        sample_id = left_path.stem

    elif 'image_0' in str(left_path):
        right_path = Path(str(left_path).replace('image_0', 'image_1'))
        gt_path = Path(str(left_path).replace('image_0', 'disp_occ'))
        sample_id = left_path.stem

    # ETH3D 格式: two_view_training/scene/im0.png -> im1.png
    # GT: two_view_training_gt/scene/disp0GT.pfm
    elif 'two_view_training' in str(left_path) and left_path.name == 'im0.png':
        right_path = left_path.parent / 'im1.png'
        gt_path = Path(str(left_path.parent).replace('two_view_training', 'two_view_training_gt')) / 'disp0GT.pfm'
        sample_id = left_path.parent.name

    # Middlebury 格式: scene/im0.png -> im1.png, disp0GT.pfm
    elif left_path.name == 'im0.png':
        right_path = left_path.parent / 'im1.png'
        gt_path = left_path.parent / 'disp0GT.pfm'
        sample_id = left_path.parent.name

    else:
        print(f"警告: 无法识别数据集格式，尝试通用推断")
        right_path = Path(str(left_path).replace('left', 'right'))
        gt_path = None
        sample_id = left_path.stem

    if not right_path.exists():
        print(f"错误: 右图不存在: {right_path}")
        return None

    gt_str = str(gt_path) if gt_path and gt_path.exists() else None
    if gt_str is None:
        print(f"警告: GT 视差图不存在: {gt_path}")

    return (str(left_path), str(right_path), gt_str, sample_id)


def get_dataset_samples(dataset, data_root):
    """获取数据集的样本列表，返回 (left_img, right_img, gt_disp, sample_id)"""
    samples = []

    if dataset == 'kitti2015':
        base = Path(data_root) / 'KITTI' / 'KITTI_2015' / 'training'
        left_dir = base / 'image_2'
        right_dir = base / 'image_3'
        gt_dir = base / 'disp_occ_0'

        if not left_dir.exists():
            print(f"错误: KITTI 2015 数据集路径不存在: {left_dir}")
            return samples

        for gt_file in sorted(gt_dir.glob('*_10.png')):
            sample_id = gt_file.stem  # e.g., '000000_10'
            left_img = left_dir / f'{sample_id}.png'
            right_img = right_dir / f'{sample_id}.png'
            if left_img.exists() and right_img.exists():
                samples.append((str(left_img), str(right_img), str(gt_file), sample_id))

    elif dataset == 'kitti2012':
        base = Path(data_root) / 'KITTI' / 'KITTI_2012' / 'training'
        left_dir = base / 'image_0'
        right_dir = base / 'image_1'
        gt_dir = base / 'disp_occ'

        if not left_dir.exists():
            print(f"错误: KITTI 2012 数据集路径不存在: {left_dir}")
            return samples

        for gt_file in sorted(gt_dir.glob('*_10.png')):
            sample_id = gt_file.stem
            left_img = left_dir / f'{sample_id}.png'
            right_img = right_dir / f'{sample_id}.png'
            if left_img.exists() and right_img.exists():
                samples.append((str(left_img), str(right_img), str(gt_file), sample_id))

    elif dataset == 'middlebury':
        base = Path(data_root) / 'Middlebury' / 'Middlebury' / 'trainingH'

        if not base.exists():
            print(f"错误: Middlebury 数据集路径不存在: {base}")
            return samples

        for scene_dir in sorted(base.iterdir()):
            if scene_dir.is_dir():
                left_img = scene_dir / 'im0.png'
                right_img = scene_dir / 'im1.png'
                gt_disp = scene_dir / 'disp0GT.pfm'

                # Middlebury trainingH 可能没有 GT，检查是否存在
                if left_img.exists() and right_img.exists():
                    gt_path = str(gt_disp) if gt_disp.exists() else None
                    samples.append((str(left_img), str(right_img), gt_path, scene_dir.name))

    elif dataset == 'sceneflow':
        # SceneFlow FlyingThings3D
        base = Path(data_root) / 'SceneFlow' / 'FlyingThings3D'
        frames_dir = base / 'frames_finalpass' / 'TRAIN'
        disp_dir = base / 'disparity' / 'TRAIN'

        if not frames_dir.exists():
            print(f"错误: SceneFlow 数据集路径不存在: {frames_dir}")
            return samples

        # 遍历 A/B/C 子目录
        for subset in ['C']:
            subset_frames = frames_dir / subset
            subset_disp = disp_dir / subset
            if not subset_frames.exists():
                continue

            for scene_dir in sorted(subset_frames.iterdir()):
                if not scene_dir.is_dir():
                    continue
                left_dir = scene_dir / 'left'
                right_dir = scene_dir / 'right'
                gt_left_dir = subset_disp / scene_dir.name / 'left'

                if not left_dir.exists():
                    continue

                for left_img in sorted(left_dir.glob('*.png')):
                    right_img = right_dir / left_img.name
                    gt_file = gt_left_dir / left_img.name.replace('.png', '.pfm')

                    if right_img.exists() and gt_file.exists():
                        sample_id = f"{subset}_{scene_dir.name}_{left_img.stem}"
                        samples.append((str(left_img), str(right_img), str(gt_file), sample_id))

    elif dataset == 'eth3d':
        # ETH3D two_view_training
        img_dir = Path(data_root) / 'ETH3D' / 'two_view_training'
        gt_dir = Path(data_root) / 'ETH3D' / 'two_view_training_gt'

        if not img_dir.exists():
            print(f"错误: ETH3D 数据集路径不存在: {img_dir}")
            return samples

        for scene_dir in sorted(img_dir.iterdir()):
            if not scene_dir.is_dir():
                continue

            left_img = scene_dir / 'im0.png'
            right_img = scene_dir / 'im1.png'
            gt_file = gt_dir / scene_dir.name / 'disp0GT.pfm'

            if left_img.exists() and right_img.exists() and gt_file.exists():
                samples.append((str(left_img), str(right_img), str(gt_file), scene_dir.name))

    return samples


def load_gt_disparity(gt_path):
    """加载 GT 视差图"""
    if gt_path is None:
        return None

    if gt_path.endswith('.pfm'):
        gt, _ = read_pfm(gt_path)
    else:
        # KITTI 格式: uint16 PNG, 除以 256 得到真实视差
        gt = cv2.imread(gt_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 256.0

    return gt


def compute_edge_from_rgb(img_path):
    """从RGB图像计算边缘区域"""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength = np.sqrt(grad_x ** 2 + grad_y ** 2)

    edge_strength = edge_strength / (edge_strength.max() + 1e-8)
    return edge_strength


def analyze_error_map(pred_baseline, pred_our, gt_disp, left_img_path, output_path):
    """分析两个模型的误差图并计算 EPE"""
    left_img = cv2.imread(left_img_path)
    left_img_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)

    mask = (gt_disp > 0) & (gt_disp < 192) & (~np.isinf(gt_disp))

    # 两个 Error Map
    error_baseline = np.abs(pred_baseline - gt_disp)
    error_baseline[~mask] = 0
    error_our = np.abs(pred_our - gt_disp)
    error_our[~mask] = 0

    # 边缘区域（从GT视差）
    grad_x = cv2.Sobel(gt_disp, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gt_disp, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength_gt = np.sqrt(grad_x ** 2 + grad_y ** 2)
    edge_mask_gt = (edge_strength_gt > 2.0) & mask

    # 边缘区域（从RGB图像）
    edge_strength_rgb = compute_edge_from_rgb(left_img_path)
    edge_threshold_rgb = np.percentile(edge_strength_rgb, 90)
    edge_mask_rgb = (edge_strength_rgb > edge_threshold_rgb)

    flat_mask = (edge_strength_gt <= 2.0) & mask

    # 统计 EPE
    epe_baseline_total = np.mean(error_baseline[mask]) if mask.sum() > 0 else 0
    epe_baseline_edge = np.mean(error_baseline[edge_mask_gt]) if edge_mask_gt.sum() > 0 else 0
    epe_baseline_flat = np.mean(error_baseline[flat_mask]) if flat_mask.sum() > 0 else 0
    d3_baseline_total = np.mean(error_baseline[mask] > 3) * 100 if mask.sum() > 0 else 0

    epe_our_total = np.mean(error_our[mask]) if mask.sum() > 0 else 0
    epe_our_edge = np.mean(error_our[edge_mask_gt]) if edge_mask_gt.sum() > 0 else 0
    epe_our_flat = np.mean(error_our[flat_mask]) if flat_mask.sum() > 0 else 0
    d3_our_total = np.mean(error_our[mask] > 3) * 100 if mask.sum() > 0 else 0

    print(f"\n--- EPE & D3 统计结果 ---")
    print(f"Baseline - EPE整体: {epe_baseline_total:.4f}, 边缘: {epe_baseline_edge:.4f}, 平坦: {epe_baseline_flat:.4f}, D3: {d3_baseline_total:.2f}%")
    print(f"Our      - EPE整体: {epe_our_total:.4f}, 边缘: {epe_our_edge:.4f}, 平坦: {epe_our_flat:.4f}, D3: {d3_our_total:.2f}%")

    # 可视化 (3行3列)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # 第一行
    axes[0, 0].set_title("Left Image")
    axes[0, 0].imshow(left_img_rgb)
    axes[0, 0].axis('off')

    axes[0, 1].set_title("Edge (RGB)")
    axes[0, 1].imshow(edge_mask_rgb, cmap='gray')
    axes[0, 1].axis('off')

    axes[0, 2].set_title("GT Disparity Edge (from GT)")
    axes[0, 2].imshow(edge_mask_gt, cmap='gray')
    axes[0, 2].axis('off')

    # 第二行 - Baseline
    axes[1, 0].set_title("Predicted Disparity (Baseline)")
    axes[1, 0].imshow(pred_baseline, cmap='magma')
    axes[1, 0].axis('off')

    axes[1, 1].set_title("Error Map (Baseline)")
    im1 = axes[1, 1].imshow(error_baseline, cmap='hot', vmin=0, vmax=5)
    axes[1, 1].axis('off')
    plt.colorbar(im1, ax=axes[1, 1], fraction=0.046)

    axes[1, 2].axis('off')
    stats_baseline = f"""Baseline Results:
EPE Mean:  {epe_baseline_total:.4f}
EPE Edge:  {epe_baseline_edge:.4f}
EPE Flat:  {epe_baseline_flat:.4f}
D3:        {d3_baseline_total:.2f}%"""
    axes[1, 2].text(0.1, 0.5, stats_baseline, fontsize=12, verticalalignment='center',
                    fontfamily='monospace', transform=axes[1, 2].transAxes)

    # 第三行 - Our
    axes[2, 0].set_title("Predicted Disparity (Ours)")
    axes[2, 0].imshow(pred_our, cmap='magma')
    axes[2, 0].axis('off')

    axes[2, 1].set_title("Error Map (Ours)")
    im2 = axes[2, 1].imshow(error_our, cmap='hot', vmin=0, vmax=5)
    axes[2, 1].axis('off')
    plt.colorbar(im2, ax=axes[2, 1], fraction=0.046)

    axes[2, 2].axis('off')
    stats_our = f"""Ours Results:
EPE Mean:  {epe_our_total:.4f}
EPE Edge:  {epe_our_edge:.4f}
EPE Flat:  {epe_our_flat:.4f}
D3:        {d3_our_total:.2f}%"""
    axes[2, 2].text(0.1, 0.5, stats_our, fontsize=12, verticalalignment='center',
                    fontfamily='monospace', transform=axes[2, 2].transAxes)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close()

    print(f"误差分析图已保存至: {output_path}")
    return (epe_baseline_total, epe_baseline_edge, epe_baseline_flat, d3_baseline_total,
            epe_our_total, epe_our_edge, epe_our_flat, d3_our_total)


def main():
    parser = argparse.ArgumentParser(description='IGEV Stereo Error Map and EPE Analysis')
    parser.add_argument('--baseline_model', type=str,
                        default='/root/autodl-tmp/stereo/model_cache/sceneflow.pth',
                        help='Baseline model checkpoint path')
    parser.add_argument('--our_model', type=str,
                        default='../logs/gt_depth_aware/95000_gt_depth_aware_edge_bs6.pth',
                        help='Our model checkpoint path')
    parser.add_argument('--dataset', type=str, default='middlebury',
                        choices=['kitti2015', 'kitti2012', 'middlebury', 'sceneflow', 'eth3d'],
                        help='Dataset to evaluate')
    parser.add_argument('--max_samples', type=int, default=5,
                        help='Maximum number of samples to evaluate (default: 5)')
    parser.add_argument('--sample_path', type=str,
                        # default='/root/autodl-tmp/stereo/dataset_cache/SceneFlow/FlyingThings3D/frames_finalpass/TRAIN/A/0130/left/0015.png',
                        # default='/root/autodl-tmp/stereo/dataset_cache/SceneFlow/FlyingThings3D/frames_finalpass/TRAIN/A/0230/left/0015.png',
                        # default='/root/autodl-tmp/stereo/dataset_cache/SceneFlow/FlyingThings3D/frames_finalpass/TRAIN/A/0030/left/0015.png',
                        default='/root/autodl-tmp/stereo/dataset_cache/SceneFlow/FlyingThings3D/frames_finalpass/TRAIN/A/0004/left/0012.png',
                        # default=None,
                        help='Direct path to left image (auto-infers right and GT)')
    parser.add_argument('--data_root', type=str,
                        default='/root/autodl-tmp/stereo/dataset_cache',
                        help='Root directory of datasets')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for results')
    parser.add_argument('--sample_id', type=str,
                        default=None,
                        help='Specific sample ID to evaluate (e.g., 000015_10 for KITTI)')
    parser.add_argument('--iters', type=int, default=32,
                        help='Number of iterations for disparity refinement')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    if not os.path.exists(args.baseline_model):
        print(f"错误: Baseline模型不存在: {args.baseline_model}")
        return
    if not os.path.exists(args.our_model):
        print(f"错误: Our模型不存在: {args.our_model}")
        return

    print(f"加载Baseline模型: {args.baseline_model}")
    baseline_model = load_model(args.baseline_model, device)

    print(f"加载Our模型: {args.our_model}")
    our_model = load_model(args.our_model, device)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取样本列表
    if args.sample_path:
        # 直接指定左图路径
        sample = get_sample_from_path(args.sample_path)
        if sample is None:
            return
        samples = [sample]
        print(f"使用指定样本: {args.sample_path}")
    else:
        # 从数据集获取样本
        samples = get_dataset_samples(args.dataset, args.data_root)
        if not samples:
            print(f"错误: 未找到 {args.dataset} 数据集的样本")
            return

        print(f"找到 {len(samples)} 个样本")

        # 如果指定了特定样本 ID，只处理该样本
        if args.sample_id:
            samples = [(l, r, g, sid) for l, r, g, sid in samples if args.sample_id in sid]
            if not samples:
                print(f"错误: 未找到样本 ID: {args.sample_id}")
                return
        else:
            # 限制处理数量
            samples = samples[:args.max_samples]

    print(f"将处理 {len(samples)} 个样本")

    # 处理每个样本
    epe_results = []
    for left_img, right_img, gt_path, sample_id in samples:
        print(f"\n{'='*50}")
        print(f"处理样本: {sample_id}")

        print("正在推理Baseline模型...")
        pred_baseline = inference(baseline_model, left_img, right_img, device, args.iters)

        print("正在推理Our模型...")
        pred_our = inference(our_model, left_img, right_img, device, args.iters)

        gt_disp = load_gt_disparity(gt_path)

        if gt_disp is None:
            print(f"警告: 样本 {sample_id} 没有 GT，跳过")
            continue

        output_name = f"{args.dataset}_{sample_id}_comparison.png"
        output_path = output_dir / output_name

        epe_baseline_total, epe_baseline_edge, epe_baseline_flat, d3_baseline, \
        epe_our_total, epe_our_edge, epe_our_flat, d3_our = analyze_error_map(
            pred_baseline, pred_our, gt_disp, left_img, str(output_path)
        )

        epe_results.append({
            'sample_id': sample_id,
            'baseline_total': epe_baseline_total,
            'baseline_edge': epe_baseline_edge,
            'baseline_flat': epe_baseline_flat,
            'baseline_d3': d3_baseline,
            'our_total': epe_our_total,
            'our_edge': epe_our_edge,
            'our_flat': epe_our_flat,
            'our_d3': d3_our
        })

    # 汇总统计
    if epe_results:
        print(f"\n{'='*130}")
        print(f"各样本详细对比:")
        print(f"{'Sample ID':<30} {'B_Mean':>10} {'O_Mean':>10} {'B_Edge':>10} {'O_Edge':>10} {'B_Flat':>10} {'O_Flat':>10} {'B_D3':>10} {'O_D3':>10}")
        print("-" * 130)
        for r in epe_results:
            print(f"{r['sample_id']:<30} {r['baseline_total']:>10.4f} {r['our_total']:>10.4f} "
                  f"{r['baseline_edge']:>10.4f} {r['our_edge']:>10.4f} "
                  f"{r['baseline_flat']:>10.4f} {r['our_flat']:>10.4f} "
                  f"{r['baseline_d3']:>9.2f}% {r['our_d3']:>9.2f}%")

        avg_baseline_total = np.mean([r['baseline_total'] for r in epe_results])
        avg_baseline_edge = np.mean([r['baseline_edge'] for r in epe_results])
        avg_baseline_flat = np.mean([r['baseline_flat'] for r in epe_results])
        avg_baseline_d3 = np.mean([r['baseline_d3'] for r in epe_results])

        avg_our_total = np.mean([r['our_total'] for r in epe_results])
        avg_our_edge = np.mean([r['our_edge'] for r in epe_results])
        avg_our_flat = np.mean([r['our_flat'] for r in epe_results])
        avg_our_d3 = np.mean([r['our_d3'] for r in epe_results])

        print(f"\n{'='*130}")
        print(f"平均值汇总 (共 {len(epe_results)} 个样本):")
        print(f"Baseline - EPE Mean: {avg_baseline_total:.4f}, EPE Edge: {avg_baseline_edge:.4f}, EPE Flat: {avg_baseline_flat:.4f}, D3: {avg_baseline_d3:.2f}%")
        print(f"Ours     - EPE Mean: {avg_our_total:.4f}, EPE Edge: {avg_our_edge:.4f}, EPE Flat: {avg_our_flat:.4f}, D3: {avg_our_d3:.2f}%")
        print(f"{'='*130}")


if __name__ == '__main__':
    main()
