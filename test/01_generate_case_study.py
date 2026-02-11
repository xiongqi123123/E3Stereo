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
from tqdm import tqdm
from IGEV.igev_stereo import IGEVStereo
from IGEV.utils import InputPadder
import IGEV.stereo_datasets as datasets


# ---------------- 工具函数 ----------------
def disp_to_color(disp, max_disp=192):
    """将视差图转换为彩色热力图"""
    if isinstance(disp, torch.Tensor):
        disp_np = disp.detach().cpu().numpy()
    else:
        disp_np = disp

    # 修复：移除多余维度，确保是 (H, W)
    disp_np = np.squeeze(disp_np)
    # 如果 squeeze 后还是 3 维 (例如原本是 1,1,H,W)，则强制取索引
    if disp_np.ndim == 3:
        disp_np = disp_np[0]

    disp_np = np.clip(disp_np, 0, max_disp)
    norm_disp = (disp_np / max_disp * 255).astype(np.uint8)
    return cv2.applyColorMap(norm_disp, cv2.COLORMAP_MAGMA)


def error_to_color(disp_pred, disp_gt, max_err=5.0):
    """将误差图转换为彩色热力图 (Error Map)"""
    if isinstance(disp_pred, torch.Tensor):
        disp_pred = disp_pred.detach().cpu().numpy()
    if isinstance(disp_gt, torch.Tensor):
        disp_gt = disp_gt.detach().cpu().numpy()

    # 修复：移除多余维度
    disp_pred = np.squeeze(disp_pred)
    disp_gt = np.squeeze(disp_gt)

    if disp_pred.ndim == 3: disp_pred = disp_pred[0]
    if disp_gt.ndim == 3: disp_gt = disp_gt[0]

    err = np.abs(disp_pred - disp_gt)
    err_vis = np.clip(err, 0, max_err)
    norm_err = (err_vis / max_err * 255).astype(np.uint8)
    return cv2.applyColorMap(norm_err, cv2.COLORMAP_JET)


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
    save_dir = 'web_vis_cpt'
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
    print(f"正在加载数据集: {args.dataset_root}")
    val_dataset = datasets.SceneFlowDatasets(
        root=args.dataset_root,
        dstype='frames_finalpass',
        things_test=True
    )

    total_len = len(val_dataset)
    if total_len == 0:
        raise ValueError(f"数据集为空！请检查路径: {args.dataset_root}")

    sample_size = min(args.num_samples, total_len)
    np.random.seed(42)
    indices = np.random.choice(total_len, sample_size, replace=False)

    results = []

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

        # Metrics
        mask = (valid_gt > 0.5) & (gt < 192)

        # Base Metric
        diff_base = torch.abs(disp_base - gt)
        if mask.sum() > 0:
            epe_base = diff_base[mask].mean().item()
            d1_base = (diff_base[mask] > 1.0).float().mean().item() * 100
        else:
            epe_base, d1_base = 0.0, 0.0

        # Our Metric
        diff_our = torch.abs(disp_our - gt)
        if mask.sum() > 0:
            epe_our = diff_our[mask].mean().item()
            d1_our = (diff_our[mask] > 1.0).float().mean().item() * 100
        else:
            epe_our, d1_our = 0.0, 0.0

        # Visualization
        img_vis = image1[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        cv2.imwrite(f"{img_dir}/{idx}_left.jpg", cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))

        cv2.imwrite(f"{img_dir}/{idx}_gt.jpg", disp_to_color(gt))

        base_vis = disp_to_color(disp_base)
        base_vis = write_text(base_vis, f"EPE: {epe_base:.2f}, D1: {d1_base:.1f}%")
        cv2.imwrite(f"{img_dir}/{idx}_base.jpg", base_vis)

        our_vis = disp_to_color(disp_our)
        our_vis = write_text(our_vis, f"EPE: {epe_our:.2f}, D1: {d1_our:.1f}%")
        cv2.imwrite(f"{img_dir}/{idx}_our.jpg", our_vis)

        cv2.imwrite(f"{img_dir}/{idx}_err_base.jpg", error_to_color(disp_base, gt))
        cv2.imwrite(f"{img_dir}/{idx}_err_our.jpg", error_to_color(disp_our, gt))

        results.append({
            'id': idx,
            'epe_base': epe_base,
            'd1_base': d1_base,
            'epe_our': epe_our,
            'd1_our': d1_our,
            'diff': epe_base - epe_our
        })

    # Generate HTML
    results.sort(key=lambda x: x['epe_base'], reverse=True)
    avg_epe_base = np.mean([r['epe_base'] for r in results])
    avg_epe_our = np.mean([r['epe_our'] for r in results])

    html = f"""
    <html><head>
    <meta charset="UTF-8">
    <style>
    body {{font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5;}}
    .summary {{background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}}
    .stats {{display: flex; gap: 20px; margin-top: 15px;}}
    .stat-box {{background: #f0f0f0; padding: 15px; border-radius: 5px; flex: 1;}}
    .stat-value {{font-size: 24px; font-weight: bold; color: #333;}}
    .stat-label {{color: #666; margin-top: 5px;}}
    table {{border-collapse: collapse; width: 100%; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}}
    th {{background: #4CAF50; color: white; padding: 12px; text-align: center; position: sticky; top: 0;}}
    td {{border: 1px solid #ddd; padding: 8px; text-align: center;}}
    tr:hover {{background: #f5f5f5;}}
    img {{width: 250px; cursor: pointer; transition: transform 0.2s;}}
    img:hover {{transform: scale(1.8); z-index: 100; position: relative;}}
    .better {{color: green; font-weight: bold;}}
    .worse {{color: red; font-weight: bold;}}
    .neutral {{color: gray;}}
    </style>
    </head><body>
    <div class="summary">
        <h2>IGEV Baseline vs Ours</h2>
        <div class="stats">
            <div class="stat-box"><div class="stat-value">{len(results)}</div><div class="stat-label">Samples</div></div>
            <div class="stat-box"><div class="stat-value">{avg_epe_base:.3f}</div><div class="stat-label">Base EPE</div></div>
            <div class="stat-box"><div class="stat-value">{avg_epe_our:.3f}</div><div class="stat-label">Ours EPE</div></div>
            <div class="stat-box"><div class="stat-value" style="color: {'green' if avg_epe_base > avg_epe_our else 'red'}">{avg_epe_base - avg_epe_our:.3f}</div><div class="stat-label">EPE Gain</div></div>
        </div>
    </div>
    <table>
    <tr><th>ID</th><th>Left</th><th>GT</th><th>Base (Pred/Err)</th><th>Ours (Pred/Err)</th><th>Metrics</th></tr>
    """

    for r in results:
        if r['diff'] > 0.05:
            c, s = "better", "↓"
        elif r['diff'] < -0.05:
            c, s = "worse", "↑"
        else:
            c, s = "neutral", "≈"

        html += f"""
        <tr>
            <td><b>{r['id']}</b></td>
            <td><img src='images/{r['id']}_left.jpg'></td>
            <td><img src='images/{r['id']}_gt.jpg'></td>
            <td><img src='images/{r['id']}_base.jpg'><br><img src='images/{r['id']}_err_base.jpg'></td>
            <td><img src='images/{r['id']}_our.jpg'><br><img src='images/{r['id']}_err_our.jpg'></td>
            <td>
                B: {r['epe_base']:.3f}<br>O: {r['epe_our']:.3f}<br>
                <hr><span class='{c}'>Gain: {s} {abs(r['diff']):.3f}</span>
            </td>
        </tr>
        """
    html += "</table></body></html>"
    with open(f"{save_dir}/index.html", "w") as f:
        f.write(html)

    print("\n" + "=" * 60)
    print("评估完成！")
    print(f"Web页面: {save_dir}/index.html")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='auto', choices=['cpu', 'cuda', 'auto'], help='选择设备')
    parser.add_argument('--ckpt_base', default='../model_cache/sceneflow.pth', help='Baseline路径')
    # parser.add_argument('--ckpt_our', default='../logs/gt_depth_aware/80000_gt_depth_aware_edge_bs6.pth')
    # parser.add_argument('--ckpt_our', default='../logs/edge_d1_26/188500_igev_edge_pt_2_6.pth')
    parser.add_argument('--ckpt_our', default='../logs/edge_cpt/64000_edge_cpt.pth')
    parser.add_argument('--dataset_root', default='../dataset_cache/SceneFlow', help='数据路径')
    parser.add_argument('--num_samples', type=int, default=10, help='采样数')

    # IGEV Args
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3)
    parser.add_argument('--corr_levels', type=int, default=2)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--n_gru_layers', type=int, default=3)
    parser.add_argument('--max_disp', type=int, default=192)
    parser.add_argument('--mixed_precision', action='store_true', default=True)
    parser.add_argument('--precision_dtype', default='float16')

    args = parser.parse_args()
    run_comparison(args)