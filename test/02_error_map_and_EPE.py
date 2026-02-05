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


def analyze_error_map(pred_disp, gt_disp, left_img_path, output_path):
    """分析误差图并计算 EPE"""
    # 读取原图用于显示
    left_img = cv2.imread(left_img_path)
    left_img_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)

    # 掩码处理（排除无效像素）
    mask = (gt_disp > 0) & (gt_disp < 192) & (~np.isinf(gt_disp))

    # 计算 Error Map
    error_map = np.abs(pred_disp - gt_disp)
    error_map[~mask] = 0

    # 提取边缘区域（基于 GT 视差的梯度）
    grad_x = cv2.Sobel(gt_disp, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gt_disp, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 设定阈值定义边缘区
    edge_mask = (edge_strength > 2.0) & mask
    flat_mask = (edge_strength <= 2.0) & mask

    # 统计 EPE
    epe_total = np.mean(error_map[mask]) if mask.sum() > 0 else 0
    epe_edge = np.mean(error_map[edge_mask]) if edge_mask.sum() > 0 else 0
    epe_flat = np.mean(error_map[flat_mask]) if flat_mask.sum() > 0 else 0

    print(f"--- EPE 统计结果 ---")
    print(f"整体 EPE: {epe_total:.4f}")
    print(f"边缘区域 EPE: {epe_edge:.4f} (关键关注点)")
    print(f"平坦区域 EPE: {epe_flat:.4f}")
    if epe_flat > 0:
        print(f"倍数差异: {epe_edge / epe_flat:.2f}x")

    # 可视化误差分布
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 第一行
    axes[0, 0].set_title("Left Image")
    axes[0, 0].imshow(left_img_rgb)
    axes[0, 0].axis('off')

    axes[0, 1].set_title("Ground Truth Disparity")
    axes[0, 1].imshow(gt_disp, cmap='magma')
    axes[0, 1].axis('off')

    axes[0, 2].set_title("Predicted Disparity")
    axes[0, 2].imshow(pred_disp, cmap='magma')
    axes[0, 2].axis('off')

    # 第二行
    axes[1, 0].set_title("Error Map (Bright = High Error)")
    im = axes[1, 0].imshow(error_map, cmap='hot', vmin=0, vmax=5)
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].set_title("Edge Regions Mask")
    axes[1, 1].imshow(edge_mask, cmap='gray')
    axes[1, 1].axis('off')

    # EPE 统计文字
    axes[1, 2].axis('off')
    stats_text = f"""EPE Statistics:

Overall EPE: {epe_total:.4f}
Edge EPE: {epe_edge:.4f}
Flat EPE: {epe_flat:.4f}
Edge/Flat Ratio: {epe_edge / epe_flat:.2f}x""" if epe_flat > 0 else f"""EPE Statistics:

Overall EPE: {epe_total:.4f}
Edge EPE: {epe_edge:.4f}
Flat EPE: {epe_flat:.4f}"""

    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=14, verticalalignment='center',
                    fontfamily='monospace', transform=axes[1, 2].transAxes)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"误差分析图已保存至: {output_path}")
    return epe_total, epe_edge, epe_flat


def main():
    parser = argparse.ArgumentParser(description='IGEV Stereo Error Map and EPE Analysis')
    parser.add_argument('--model', type=str, default='sceneflow',
                        choices=['sceneflow', 'finetuned'],
                        help='Model to use: sceneflow (pretrained) or finetuned')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Custom model checkpoint path (overrides --model)')
    parser.add_argument('--dataset', type=str, default='middlebury',
                        choices=['kitti2015', 'kitti2012', 'middlebury', 'sceneflow', 'eth3d'],
                        help='Dataset to evaluate')
    parser.add_argument('--max_samples', type=int, default=5,
                        help='Maximum number of samples to evaluate (default: 5)')
    parser.add_argument('--sample_path', type=str,
                        # default='/root/autodl-tmp/stereo/dataset_cache/SceneFlow/FlyingThings3D/frames_finalpass/TRAIN/A/0130/left/0015.png',
                        # default='/root/autodl-tmp/stereo/dataset_cache/SceneFlow/FlyingThings3D/frames_finalpass/TRAIN/A/0230/left/0015.png',
                        # default='/root/autodl-tmp/stereo/dataset_cache/SceneFlow/FlyingThings3D/frames_finalpass/TRAIN/A/0030/left/0015.png',
                        default=None,
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

    # 自适应设备选择（一次判断）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 确定模型路径
    if args.model_path:
        ckpt_path = args.model_path
        model_name = Path(args.model_path).stem
    elif args.model == 'sceneflow':
        ckpt_path = '/root/autodl-tmp/stereo/model_cache/sceneflow.pth'
        model_name = 'sceneflow'
    else:  # finetuned
        # 假设微调模型的默认路径
        ckpt_path = '/root/autodl-tmp/stereo/model_cache/finetuned.pth'
        model_name = 'finetuned'

    if not os.path.exists(ckpt_path):
        print(f"错误: 模型文件不存在: {ckpt_path}")
        return

    print(f"加载模型: {ckpt_path}")
    model = load_model(ckpt_path, device)

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
    all_epe = []
    epe_results = []  # 存储每个样本的详细结果
    for left_img, right_img, gt_path, sample_id in samples:
        print(f"\n{'='*50}")
        print(f"处理样本: {sample_id}")
        print(f"左图: {left_img}")
        print(f"右图: {right_img}")

        # 推理
        print("正在推理...")
        pred_disp = inference(model, left_img, right_img, device, args.iters)

        # 加载 GT
        gt_disp = load_gt_disparity(gt_path)

        if gt_disp is None:
            print(f"警告: 样本 {sample_id} 没有 GT 视差图，跳过误差分析")
            # 仅保存预测结果
            output_name = f"{args.dataset}_{model_name}_{sample_id}_pred.png"
            output_path = output_dir / output_name
            plt.figure(figsize=(10, 5))
            plt.imshow(pred_disp, cmap='magma')
            plt.colorbar()
            plt.title(f'Predicted Disparity - {sample_id}')
            plt.savefig(output_path, bbox_inches='tight')
            plt.show()
            plt.close()
            print(f"预测视差图已保存至: {output_path}")
            continue

        # 误差分析
        output_name = f"{args.dataset}_{model_name}_{sample_id}_error.png"
        output_path = output_dir / output_name

        epe_total, epe_edge, epe_flat = analyze_error_map(
            pred_disp, gt_disp, left_img, str(output_path)
        )
        all_epe.append(epe_total)
        epe_results.append({
            'sample_id': sample_id,
            'epe_total': epe_total,
            'epe_edge': epe_edge,
            'epe_flat': epe_flat
        })

    # 汇总统计
    if epe_results:
        # 提取各类 EPE 列表
        all_epe_total = [r['epe_total'] for r in epe_results]
        all_epe_edge = [r['epe_edge'] for r in epe_results]
        all_epe_flat = [r['epe_flat'] for r in epe_results]

        print(f"\n{'='*70}")
        print(f"各样本 EPE 详情 ({args.dataset if not args.sample_path else 'custom'}, {model_name}):")
        print(f"{'Sample ID':<30} {'Total EPE':>12} {'Edge EPE':>12} {'Flat EPE':>12}")
        print("-" * 70)
        for r in epe_results:
            print(f"{r['sample_id']:<30} {r['epe_total']:>12.4f} {r['epe_edge']:>12.4f} {r['epe_flat']:>12.4f}")

        print(f"\n{'='*70}")
        print(f"汇总统计 (共 {len(epe_results)} 个样本):")
        print(f"{'指标':<20} {'Total EPE':>12} {'Edge EPE':>12} {'Flat EPE':>12}")
        print("-" * 70)
        print(f"{'平均值 (Mean)':<20} {np.mean(all_epe_total):>12.4f} {np.mean(all_epe_edge):>12.4f} {np.mean(all_epe_flat):>12.4f}")
        print(f"{'最小值 (Min)':<20} {np.min(all_epe_total):>12.4f} {np.min(all_epe_edge):>12.4f} {np.min(all_epe_flat):>12.4f}")
        print(f"{'最大值 (Max)':<20} {np.max(all_epe_total):>12.4f} {np.max(all_epe_edge):>12.4f} {np.max(all_epe_flat):>12.4f}")
        print(f"{'标准差 (Std)':<20} {np.std(all_epe_total):>12.4f} {np.std(all_epe_edge):>12.4f} {np.std(all_epe_flat):>12.4f}")
        print(f"{'='*70}")


if __name__ == '__main__':
    main()
