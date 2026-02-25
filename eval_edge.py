#!/usr/bin/env python
"""
几何边缘模型评估与可视化：加载训练好的 GeoEdgeNet，在 SceneFlow 上推理并保存可视化结果。
"""
import argparse
import os
import os.path as osp
import numpy as np
import torch
import cv2
from tqdm import tqdm
from glob import glob

from core.edge_models import GeoEdgeNet
from core.utils import frame_utils


def collect_sceneflow_pairs(root, dstype, split, max_samples=None):
    """收集 SceneFlow (image, edge) 路径对，无增强"""
    left_images = sorted(glob(osp.join(root, dstype, split, "*/*/left/*.png")))
    left_images += sorted(glob(osp.join(root, dstype, split, "*/left/*.png")))
    left_images += sorted(glob(osp.join(root, dstype, split, "*/*/*/left/*.png")))
    left_images = sorted(set(left_images))

    pairs = []
    for left_path in left_images:
        edge_path = left_path.replace(dstype, "gtedge")
        if osp.exists(edge_path):
            pairs.append((left_path, edge_path))
            if max_samples and len(pairs) >= max_samples:
                break
    return pairs


def load_image(path):
    img = np.array(frame_utils.read_gen(path)).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.tile(img[..., None], (1, 1, 3))
    return img[..., :3]


def load_edge(path):
    edge = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if edge is None:
        return np.zeros((1, 1), dtype=np.float32)
    return edge.astype(np.float32) / 255.0


def visualize_row(img, gt_edge, pred_edge, thresh=0.5):
    """
    生成一行可视化：[RGB | GT Edge | Pred Edge | Overlay]
    """
    h, w = img.shape[:2]
    gt_3ch = np.stack([gt_edge] * 3, axis=-1)
    pred_bin = (pred_edge > thresh).astype(np.float32)
    pred_3ch = np.stack([pred_bin] * 3, axis=-1)

    # Overlay: 绿色=GT, 红色=Pred, 黄色=重叠
    overlay = img.copy().astype(np.float32) / 255.0
    overlay[..., 0] = np.clip(overlay[..., 0] + pred_bin * 0.5, 0, 1)  # 预测边缘偏红
    overlay[..., 1] = np.clip(overlay[..., 1] + gt_edge * 0.5, 0, 1)   # GT 边缘偏绿
    overlay = (overlay * 255).astype(np.uint8)

    # 统一尺寸
    gt_vis = (gt_3ch * 255).astype(np.uint8)
    pred_vis = (pred_3ch * 255).astype(np.uint8)

    row = np.concatenate([img, gt_vis, pred_vis, overlay], axis=1)
    return row


def run_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoEdgeNet(
        use_refinement=not getattr(args, "no_refinement", False),
        refine_iters=getattr(args, "refine_iters", 1),
        use_spatial_attn=not getattr(args, "no_spatial_attn", False),
    )
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    pairs = collect_sceneflow_pairs(
        args.data_root, args.dstype, args.split, max_samples=args.num_vis
    )
    if len(pairs) == 0:
        print(f"No samples found in {args.data_root}/{args.dstype}/{args.split}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    thresh = args.thresh
    metrics_list = [] if args.save_metrics else None

    for idx, (img_path, edge_path) in enumerate(tqdm(pairs, desc="Eval")):
        img = load_image(img_path)
        gt_edge = load_edge(edge_path)
        if gt_edge.shape != img.shape[:2]:
            gt_edge = cv2.resize(gt_edge, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Inference
        img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        img_t = (2 * (img_t / 255.0) - 1.0).to(device)
        with torch.no_grad():
            pred_logits = model(img_t)
        pred_edge = torch.sigmoid(pred_logits).squeeze(1).cpu().numpy()[0]

        if pred_edge.shape != img.shape[:2]:
            pred_edge = cv2.resize(pred_edge, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Visualize
        row = visualize_row(img, gt_edge, pred_edge, thresh=thresh)

        # 添加标题
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(row, "RGB", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(row, "GT Edge", (img.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(row, "Pred", (img.shape[1] * 2 + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(row, "Overlay", (img.shape[1] * 3 + 10, 30), font, 1, (255, 255, 255), 2)

        out_path = osp.join(args.output_dir, f"{idx:05d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

        if args.save_metrics:
            tp = ((pred_edge > thresh) & (gt_edge > 0.5)).sum()
            fp = ((pred_edge > thresh) & (gt_edge <= 0.5)).sum()
            fn = ((pred_edge <= thresh) & (gt_edge > 0.5)).sum()
            prec = float(tp / (tp + fp + 1e-8))
            rec = float(tp / (tp + fn + 1e-8))
            f1 = float(2 * prec * rec / (prec + rec + 1e-8))
            metrics_list.append((idx, prec, rec, f1))

    if args.save_metrics and metrics_list:
        with open(osp.join(args.output_dir, "metrics.txt"), "w") as f:
            for idx, prec, rec, f1 in metrics_list:
                f.write(f"{idx:05d} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}\n")
            avg = np.mean([m[1:] for m in metrics_list], axis=0)
            f.write(f"\nAvg: prec={avg[0]:.4f} rec={avg[1]:.4f} f1={avg[2]:.4f}\n")
        print(f"Metrics saved to {osp.join(args.output_dir, 'metrics.txt')}")

    print(f"Saved {len(pairs)} visualizations to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval & visualize geometric edge model")
    parser.add_argument("--ckpt", required=True, help="path to GeoEdgeNet checkpoint")
    parser.add_argument("--data_root", default="/home/qi.xiong/StereoMatching/IGEV-Improve/data/sceneflow")
    parser.add_argument("--dstype", default="frames_finalpass")
    parser.add_argument("--split", default="TEST", choices=["TRAIN", "TEST"])
    parser.add_argument("--output_dir", default="./eval_edge_vis")
    parser.add_argument("--num_vis", type=int, default=100, help="max samples to visualize")
    parser.add_argument("--thresh", type=float, default=0.5, help="binary threshold for pred")
    parser.add_argument("--save_metrics", action="store_true", help="append per-image metrics to file")
    parser.add_argument("--no_refinement", action="store_true",
                        help="关闭 EdgeRefinementModule（加载旧 ckpt 时使用）")
    parser.add_argument("--no_spatial_attn", action="store_true",
                        help="关闭 SpatialAttention（加载旧 ckpt 时使用）")
    parser.add_argument("--refine_iters", type=int, default=1,
                        help="Refine 迭代次数，需与训练时一致")
    args = parser.parse_args()
    run_eval(args)
