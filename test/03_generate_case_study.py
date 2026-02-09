import time
import os
import argparse
import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from IGEV.igev_stereo import IGEVStereo
from IGEV import frame_utils
import IGEV.stereo_datasets as datasets
from IGEV.utils import InputPadder


# ---------------- 工具函数 ----------------
def disp_to_color(disp, max_disp=192):
    """将视差图转换为彩色热力图"""
    disp_np = disp.cpu().numpy()
    disp_np = np.clip(disp_np, 0, max_disp)
    norm_disp = (disp_np / max_disp * 255).astype(np.uint8)
    return cv2.applyColorMap(norm_disp, cv2.COLORMAP_MAGMA)


def error_to_color(disp_pred, disp_gt, max_err=5.0):
    """将误差图转换为彩色热力图 (Error Map)"""
    err = torch.abs(disp_pred - disp_gt)
    err_np = err.cpu().numpy()
    # 截断误差，让微小误差显色，巨大误差不爆表
    err_vis = np.clip(err_np, 0, max_err)
    norm_err = (err_vis / max_err * 255).astype(np.uint8)
    return cv2.applyColorMap(norm_err, cv2.COLORMAP_JET)


def write_text(img, text):
    """在图片左上角写入文字"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (10, 30), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


# ---------------- 主逻辑 ----------------
def run_comparison(args):
    # 1. 准备目录
    save_dir = 'web_vis'
    img_dir = os.path.join(save_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    # 2. 加载模型
    # 检查checkpoint文件是否存在
    if not os.path.exists(args.ckpt_base):
        raise FileNotFoundError(f"Baseline模型不存在: {args.ckpt_base}")
    if not os.path.exists(args.ckpt_our):
        raise FileNotFoundError(f"Our模型不存在: {args.ckpt_our}")

    print("正在加载 Baseline 模型...")
    model_base = torch.nn.DataParallel(IGEVStereo(args))
    model_base.load_state_dict(torch.load(args.ckpt_base), strict=False)
    model_base = model_base.module.cuda().eval()
    print("Baseline 模型加载完成！")

    print("正在加载 Ours 模型...")
    model_our = torch.nn.DataParallel(IGEVStereo(args))
    model_our.load_state_dict(torch.load(args.ckpt_our), strict=False)
    model_our = model_our.module.cuda().eval()
    print("Ours 模型加载完成！")

    # 3. 加载数据
    print(f"正在加载数据集: {args.dataset_root}")
    val_dataset = datasets.SceneFlowDatasets(
        root=args.dataset_root,
        dstype='frames_finalpass',
        things_test=True
    )

    # 随机采样（如果数据集不足则使用全部）
    total_len = len(val_dataset)
    print(f"数据集总样本数: {total_len}")

    if total_len == 0:
        raise ValueError(
            f"数据集为空！请检查路径: {args.dataset_root}\n"
            f"确保以下路径存在:\n"
            f"  - {args.dataset_root}/FlyingThings3D/frames_finalpass/TEST/\n"
            f"  - {args.dataset_root}/Monkaa/frames_finalpass/\n"
            f"  - {args.dataset_root}/Driving/frames_finalpass/\n"
        )

    sample_size = min(args.num_samples, total_len)
    print(f"将评估 {sample_size} 个样本")

    # 设置随机种子以便复现
    np.random.seed(42)
    indices = np.random.choice(total_len, sample_size, replace=False)

    results = []  # 存储 metadata

    print(f"开始评估 {sample_size} 个样本...")
    for idx in tqdm(indices, desc="处理样本"):
        _, image1, image2, flow_gt, valid_gt = val_dataset[idx]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        # Padding
        padder = InputPadder(image1.shape, divis_by=32)
        image1_pad, image2_pad = padder.pad(image1, image2)

        with torch.no_grad():
            # 推理 Baseline
            disp_base = model_base(image1_pad, image2_pad, iters=32, test_mode=True)
            disp_base = padder.unpad(disp_base).squeeze(0)

            # 推理 Ours
            disp_our = model_our(image1_pad, image2_pad, iters=32, test_mode=True)
            disp_our = padder.unpad(disp_our).squeeze(0)

        # 计算指标
        gt = flow_gt
        mask = (valid_gt > 0.5) & (gt < 192)

        # Base Metric
        diff_base = torch.abs(disp_base - gt)
        epe_base = diff_base[mask].mean().item()
        d1_base = (diff_base[mask] > 1.0).float().mean().item() * 100

        # Our Metric
        diff_our = torch.abs(disp_our - gt)
        epe_our = diff_our[mask].mean().item()
        d1_our = (diff_our[mask] > 1.0).float().mean().item() * 100

        # ---- 生成可视化图片 ----
        # 1. Left Image
        img_vis = image1[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        cv2.imwrite(f"{img_dir}/{idx}_left.jpg", cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))

        # 2. GT Disparity
        gt_vis = disp_to_color(gt)
        cv2.imwrite(f"{img_dir}/{idx}_gt.jpg", gt_vis)

        # 3. Base Prediction
        base_vis = disp_to_color(disp_base)
        base_vis = write_text(base_vis, f"EPE: {epe_base:.2f}, D1: {d1_base:.1f}%")
        cv2.imwrite(f"{img_dir}/{idx}_base.jpg", base_vis)

        # 4. Our Prediction
        our_vis = disp_to_color(disp_our)
        our_vis = write_text(our_vis, f"EPE: {epe_our:.2f}, D1: {d1_our:.1f}%")
        cv2.imwrite(f"{img_dir}/{idx}_our.jpg", our_vis)

        # 5. Error Maps (Optional but recommended)
        err_base_vis = error_to_color(disp_base, gt)
        err_our_vis = error_to_color(disp_our, gt)
        cv2.imwrite(f"{img_dir}/{idx}_err_base.jpg", err_base_vis)
        cv2.imwrite(f"{img_dir}/{idx}_err_our.jpg", err_our_vis)

        results.append({
            'id': idx,
            'epe_base': epe_base,
            'd1_base': d1_base,
            'epe_our': epe_our,
            'd1_our': d1_our,
            'diff': epe_base - epe_our  # 正值表示 Our 更好
        })

    # 4. 生成 HTML
    # 按 Baseline EPE 降序排列 (看 worst cases)
    results.sort(key=lambda x: x['epe_base'], reverse=True)

    # 计算统计信息
    avg_epe_base = np.mean([r['epe_base'] for r in results])
    avg_epe_our = np.mean([r['epe_our'] for r in results])
    better_count = sum(1 for r in results if r['diff'] > 0)

    html = f"""
    <html><head>
    <meta charset="UTF-8">
    <style>
    body {{font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5;}}
    .summary {{background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}}
    .summary h2 {{margin-top: 0;}}
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
        <h2>IGEV Baseline vs Ours - 对比分析</h2>
        <div class="stats">
            <div class="stat-box">
                <div class="stat-value">{len(results)}</div>
                <div class="stat-label">总样本数</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{avg_epe_base:.3f}</div>
                <div class="stat-label">Baseline 平均 EPE</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{avg_epe_our:.3f}</div>
                <div class="stat-label">Ours 平均 EPE</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" style="color: {'green' if avg_epe_base > avg_epe_our else 'red'}">{avg_epe_base - avg_epe_our:.3f}</div>
                <div class="stat-label">平均 EPE 提升</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{better_count}/{len(results)}</div>
                <div class="stat-label">Ours 更好的样本</div>
            </div>
        </div>
        <p style="margin-top: 15px; color: #666;">
            <b>说明:</b> 表格按 Baseline EPE 降序排列（最困难的样本在前）。鼠标悬停图片可放大查看细节。
        </p>
    </div>
    <table>
    <tr><th>样本ID</th><th>Left Image</th><th>GT Disp</th><th>Baseline Pred<br>(Pred / Error)</th><th>Ours Pred<br>(Pred / Error)</th><th>Metrics</th></tr>
    """

    for r in results:
        if r['diff'] > 0.1:
            color_class = "better"
            gain_symbol = "↓"
        elif r['diff'] < -0.1:
            color_class = "worse"
            gain_symbol = "↑"
        else:
            color_class = "neutral"
            gain_symbol = "≈"

        diff_text = f"{gain_symbol} {abs(r['diff']):.3f}"

        row = f"""
        <tr>
            <td><b>{r['id']}</b></td>
            <td><img src='images/{r['id']}_left.jpg' title='Left Image'></td>
            <td><img src='images/{r['id']}_gt.jpg' title='Ground Truth Disparity'></td>
            <td>
                <img src='images/{r['id']}_base.jpg' title='Baseline Prediction'><br>
                <small style="color: #666;">Error Map:</small><br>
                <img src='images/{r['id']}_err_base.jpg' title='Baseline Error Map'>
            </td>
            <td>
                <img src='images/{r['id']}_our.jpg' title='Ours Prediction'><br>
                <small style="color: #666;">Error Map:</small><br>
                <img src='images/{r['id']}_err_our.jpg' title='Ours Error Map'>
            </td>
            <td>
                <b>Base EPE:</b> {r['epe_base']:.3f}<br>
                <b>Base D1:</b> {r['d1_base']:.2f}%<br>
                <hr style="margin: 8px 0; border: none; border-top: 1px solid #ddd;">
                <b>Our EPE:</b> {r['epe_our']:.3f}<br>
                <b>Our D1:</b> {r['d1_our']:.2f}%<br>
                <hr style="margin: 8px 0; border: none; border-top: 1px solid #ddd;">
                <span class='{color_class}' style="font-size: 16px;"><b>EPE Gain: {diff_text}</b></span>
            </td>
        </tr>
        """
        html += row

    html += "</table></body></html>"

    with open(f"{save_dir}/index.html", "w") as f:
        f.write(html)

    # 5. 输出统计摘要
    print("\n" + "="*60)
    print("评估完成！统计摘要:")
    print("="*60)

    avg_epe_base = np.mean([r['epe_base'] for r in results])
    avg_epe_our = np.mean([r['epe_our'] for r in results])
    avg_d1_base = np.mean([r['d1_base'] for r in results])
    avg_d1_our = np.mean([r['d1_our'] for r in results])
    avg_gain = np.mean([r['diff'] for r in results])

    print(f"Baseline 平均 EPE: {avg_epe_base:.3f}")
    print(f"Ours 平均 EPE:     {avg_epe_our:.3f}")
    print(f"平均 EPE 提升:     {avg_gain:.3f}")
    print(f"\nBaseline 平均 D1:  {avg_d1_base:.2f}%")
    print(f"Ours 平均 D1:      {avg_d1_our:.2f}%")
    print(f"平均 D1 提升:      {avg_d1_base - avg_d1_our:.2f}%")
    print("="*60)

    better_count = sum(1 for r in results if r['diff'] > 0)
    print(f"\nOur方法更好的样本: {better_count}/{len(results)} ({better_count/len(results)*100:.1f}%)")
    print(f"总样本数: {len(results)}")
    print(f"图片保存位置: {img_dir}")
    print(f"Web页面: {save_dir}/index.html")
    print("\n请用浏览器打开 web_vis/index.html 查看详细对比结果。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 模型checkpoint路径
    parser.add_argument('--ckpt_base', default='../logs/baseline_igev/1001_igev-stereo.pth',
                        help='Baseline IGEV模型路径')
    parser.add_argument('--ckpt_our', default='../logs/ours_edge_weight/1001_igev-stereo.pth',
                        help='Our方法模型路径')

    # 数据集路径
    parser.add_argument('--dataset_root', default='../dataset_cache/SceneFlow',
                        help='SceneFlow数据集根目录')

    # 采样数量
    parser.add_argument('--num_samples', type=int, default=50,
                        help='要评估的样本数量（如果数据集不足则使用全部）')

    # 其他 IGEV 默认参数
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