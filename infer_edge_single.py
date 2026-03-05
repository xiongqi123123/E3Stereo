#!/usr/bin/env python
"""
单张图片几何边缘推理：使用训练好的 GeoEdgeNet 对单张 RGB 图像进行边缘预测。
"""
import argparse
import os
import os.path as osp
import numpy as np
import torch
import cv2

from core.edge_models import GeoEdgeNet
from core.utils import frame_utils


def load_image(path):
    """加载 RGB 图像，支持灰度图转 3 通道"""
    img = np.array(frame_utils.read_gen(path)).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.tile(img[..., None], (1, 1, 3))
    return img[..., :3]


def infer_single_image(model, img_path, device, thresh=0.5):
    """
    对单张图片进行边缘推理。
    Args:
        model: GeoEdgeNet 模型
        img_path: 图片路径
        device: torch device
        thresh: 二值化阈值
    Returns:
        img: 原图 [H,W,3]
        pred_edge: 边缘概率图 [H,W]，值域 [0,1]
        pred_bin: 二值边缘图 [H,W]
    """
    img = load_image(img_path)
    img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    img_t = (2 * (img_t / 255.0) - 1.0).to(device)

    with torch.no_grad():
        pred_logits = model(img_t)
    pred_edge = torch.sigmoid(pred_logits).squeeze(1).cpu().numpy()[0]

    if pred_edge.shape != img.shape[:2]:
        pred_edge = cv2.resize(
            pred_edge, (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
    pred_bin = (pred_edge > thresh).astype(np.float32)
    return img, pred_edge, pred_bin


def main():
    parser = argparse.ArgumentParser(description="单张图片几何边缘推理")
    parser.add_argument("--ckpt", required=True, help="GeoEdgeNet 权重路径")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--output", default=None, help="输出路径，默认保存到图片同目录")
    parser.add_argument("--thresh", type=float, default=0.5, help="二值化阈值")
    parser.add_argument("--no_refinement", action="store_true",
                        help="关闭 EdgeRefinementModule（加载旧 ckpt 时使用）")
    parser.add_argument("--no_spatial_attn", action="store_true",
                        help="关闭 SpatialAttention（加载旧 ckpt 时使用）")
    parser.add_argument("--refine_iters", type=int, default=1,
                        help="Refine 迭代次数，需与训练时一致")
    parser.add_argument("--save_prob", action="store_true",
                        help="额外保存概率图（0-255）")
    parser.add_argument("--vis_overlay", action="store_true",
                        help="保存原图+边缘叠加可视化")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoEdgeNet(
        use_refinement=not args.no_refinement,
        refine_iters=args.refine_iters,
        use_spatial_attn=not args.no_spatial_attn,
    )
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    img, pred_edge, pred_bin = infer_single_image(
        model, args.image, device, thresh=args.thresh
    )

    # 确定输出路径
    base_dir = osp.dirname(args.image)
    base_name = osp.splitext(osp.basename(args.image))[0]
    if args.output:
        out_dir = osp.dirname(args.output)
        out_base = osp.splitext(osp.basename(args.output))[0]
    else:
        out_dir = base_dir
        out_base = f"{base_name}_edge"

    os.makedirs(out_dir, exist_ok=True)

    # 保存二值边缘图
    edge_path = osp.join(out_dir, f"{out_base}.png")
    cv2.imwrite(edge_path, (pred_bin * 255).astype(np.uint8))
    print(f"二值边缘图已保存: {edge_path}")

    if args.save_prob:
        prob_path = osp.join(out_dir, f"{out_base}_prob.png")
        cv2.imwrite(prob_path, (pred_edge * 255).astype(np.uint8))
        print(f"概率图已保存: {prob_path}")

    if args.vis_overlay:
        overlay = img.copy().astype(np.float32) / 255.0
        overlay[..., 0] = np.clip(overlay[..., 0] + pred_bin * 0.5, 0, 1)
        overlay[..., 1] = np.clip(overlay[..., 1] + pred_bin * 0.5, 0, 1)
        overlay = (overlay * 255).astype(np.uint8)
        overlay_path = osp.join(out_dir, f"{out_base}_overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"叠加可视化已保存: {overlay_path}")


if __name__ == "__main__":
    main()
