"""
生成立体匹配的 Error Map（EPE 热力图）可视化。
支持 KITTI、SceneFlow、Middlebury、ETH3D 等数据集。
支持 folder 模式：推理整个文件夹（如 SceneFlow TEST），按原文件架构保存。
"""
import sys
sys.path.append('core')

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

from igev_stereo import IGEVStereo, autocast
from utils.utils import InputPadder
import stereo_datasets as datasets

DEVICE = 'cuda'


def load_image(imfile):
    """加载图像为 [1, 3, H, W] tensor"""
    img = np.array(Image.open(imfile).convert('RGB')).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def error_to_colormap(epe_np, valid_mask, vmax=5.0, invalid_color=(0.5, 0.5, 0.5)):
    """
    将 EPE 转为 RGB 热力图。
    - 无效区域：灰色
    - 有效区域：0=绿，低=青，中=黄，高=红（RdYlGn_r 风格）
    """
    h, w = epe_np.shape
    rgb = np.ones((h, w, 3), dtype=np.float32) * np.array(invalid_color)
    epe_clip = np.clip(epe_np, 0, vmax)
    # 有效区域用 jet：蓝(0) -> 青 -> 绿 -> 黄 -> 红(高)
    cmap = plt.get_cmap('jet')
    rgb[valid_mask] = cmap(epe_clip[valid_mask] / vmax)[:, :3]
    return (rgb * 255).astype(np.uint8)


def overlay_errormap(image_rgb, errormap_rgb, alpha=0.5):
    """将 error map 叠加到原图上"""
    return (alpha * errormap_rgb + (1 - alpha) * image_rgb).astype(np.uint8)


def collect_pairs_from_folder(root_folder):
    """
    递归扫描 left/right 子目录结构（SceneFlow），使用 os.walk(followlinks=True）。
    返回 [(left_path, right_path, rel_dir, out_filename), ...]
    """
    root = Path(root_folder).resolve()
    pairs = []
    seen = set()

    def add_pair(left_p, right_p, rel_d, out_fn=None):
        key = (str(left_p), str(right_p))
        if key not in seen:
            seen.add(key)
            pairs.append((str(left_p), str(right_p), str(rel_d), out_fn))

    for dirpath, dirnames, _ in os.walk(root, followlinks=True):
        if 'left' in dirnames and 'right' in dirnames:
            parent = Path(dirpath)
            left_d = parent / 'left'
            right_d = parent / 'right'
            if not left_d.is_dir() or not right_d.is_dir():
                continue
            rel_dir = parent.relative_to(root)
            for left_img in left_d.iterdir():
                if left_img.suffix.lower() in ('.png', '.jpg', '.jpeg', '.ppm'):
                    right_img = right_d / left_img.name
                    if right_img.exists():
                        add_pair(left_img, right_img, rel_dir, out_fn=left_img.name)

    return sorted(pairs, key=lambda x: x[0])


def _left_to_disp_path(left_path, frames_subdir='frames_finalpass', disp_subdir='disparity'):
    """从 left 图像路径推导 SceneFlow GT disparity 路径"""
    left_path = str(left_path)
    if frames_subdir not in left_path:
        return None
    disp_path = left_path.replace(frames_subdir, disp_subdir).replace('.png', '.pfm').replace('.jpg', '.pfm')
    return disp_path


def demo_folder(args):
    """
    文件夹模式：推理 SceneFlow TEST 等，按原文件架构保存 errormap。
    GT disparity 从 data_root/disparity 推导（frames_finalpass -> disparity, .png -> .pfm）。
    预测可从 pred_dir 加载 .npy（若 demo_imgs 用了 --save_numpy）或实时推理。
    """
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    ckpt = torch.load(args.restore_ckpt, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt, strict=False)
    model = model.module
    model.to(DEVICE)
    model.eval()

    from utils.frame_utils import readPFM

    pairs = collect_pairs_from_folder(args.folder)
    if not pairs:
        print(f"No left/right pairs found in {args.folder}")
        return

    output_dir = Path(args.output_directory)
    output_dir.mkdir(exist_ok=True, parents=True)
    # data_root: 含 frames_finalpass 和 disparity 的 sceneflow 根目录
    data_root = Path(args.data_root).resolve() if args.data_root else Path(args.folder).resolve().parent.parent
    pred_dir = Path(args.pred_dir) if args.pred_dir else None
    vmax = args.error_vmax
    max_disp = getattr(args, 'max_disp', 192)

    print(f"Found {len(pairs)} pairs. Output: {output_dir}")
    if pred_dir:
        print(f"Will try loading .npy from {pred_dir} (skip inference if found)")

    with torch.no_grad():
        for (imfile1, imfile2, rel_dir, out_filename) in tqdm(pairs, desc="Errormap"):
            # GT disparity
            disp_gt_path = _left_to_disp_path(imfile1)
            if not disp_gt_path or not os.path.exists(disp_gt_path):
                continue
            disp_gt = readPFM(disp_gt_path)
            disp_gt = np.array(disp_gt).astype(np.float32)
            valid = disp_gt > 0

            # Prediction: 优先从 pred_dir 加载 .npy，否则推理
            flow_pr = None
            if pred_dir:
                npy_path = pred_dir / rel_dir / Path(out_filename).with_suffix('.npy')
                if npy_path.exists():
                    flow_pr = np.load(npy_path)
            if flow_pr is None:
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)
                padder = InputPadder(image1.shape, divis_by=32)
                image1, image2 = padder.pad(image1, image2)
                with autocast(enabled=args.mixed_precision):
                    flow_pr = model(image1, image2, iters=args.valid_iters, test_mode=True)
                flow_pr = padder.unpad(flow_pr).cpu().squeeze(0).numpy()
                if flow_pr.ndim == 3:
                    flow_pr = flow_pr[0]

            if disp_gt.shape != flow_pr.shape:
                flow_pr = cv2.resize(flow_pr, (disp_gt.shape[1], disp_gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            val = valid & (np.abs(disp_gt) < max_disp)
            epe = np.abs(flow_pr.astype(np.float32) - disp_gt)
            epe[~val] = 0
            errormap_rgb = error_to_colormap(epe, val, vmax=vmax)

            out_subdir = output_dir / rel_dir
            out_subdir.mkdir(parents=True, exist_ok=True)
            out_name = Path(out_filename).stem + '_errormap.png'
            plt.imsave(out_subdir / out_name, errormap_rgb.astype(np.uint8) / 255.0)
            if args.save_overlay:
                img_rgb = np.array(Image.open(imfile1).convert('RGB'))
                if img_rgb.shape[:2] != errormap_rgb.shape[:2]:
                    errormap_resized = cv2.resize(errormap_rgb, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
                else:
                    errormap_resized = errormap_rgb
                overlay = overlay_errormap(img_rgb, errormap_resized, alpha=args.overlay_alpha)
                Image.fromarray(overlay).save(out_subdir / (Path(out_filename).stem + '_overlay.png'))
            if args.save_numpy:
                np.save(out_subdir / (Path(out_filename).stem + '_epe.npy'), epe)

    print(f"Saved errormaps to {output_dir}")


def demo_dataset(args):
    """基于数据集的 Error Map 生成"""
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    ckpt = torch.load(args.restore_ckpt, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt, strict=False)
    model = model.module
    model.to(DEVICE)
    model.eval()

    # 构建验证数据集
    aug_params = {}
    edge_source = getattr(args, 'edge_source', 'rcf')
    if args.dataset == 'kitti':
        val_dataset = datasets.KITTI(aug_params, image_set='training', edge_source=edge_source)
    elif args.dataset == 'sceneflow':
        val_dataset = datasets.SceneFlowDatasets(
            dstype='frames_finalpass', things_test=True, edge_source=edge_source
        )
    elif args.dataset == 'eth3d':
        val_dataset = datasets.ETH3D(aug_params, split='training', edge_source=edge_source)
    elif args.dataset.startswith('middlebury'):
        split = args.dataset.replace('middlebury_', '') or 'F'
        val_dataset = datasets.Middlebury(aug_params, split=split, edge_source=edge_source)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    output_dir = Path(args.output_directory)
    output_dir.mkdir(exist_ok=True, parents=True)

    max_disp = getattr(args, 'max_disp', 192)
    vmax = args.error_vmax

    with torch.no_grad():
        for val_id in tqdm(range(min(len(val_dataset), args.max_samples) if args.max_samples else len(val_dataset)), desc="Errormap"):
            data_item = val_dataset[val_id]
            meta = data_item[0]
            image1 = data_item[1][None].to(DEVICE)
            image2 = data_item[2][None].to(DEVICE)
            flow_gt = data_item[3]  # [1, H, W] disp
            valid_gt = data_item[4]  # [1, H, W]

            # 数据集特定：occ mask（ETH3D / Middlebury）
            occ_mask = None
            if args.dataset == 'eth3d':
                gt_file = meta[2] if len(meta) > 2 else None
                if gt_file and gt_file.replace('disp0GT.pfm', 'mask0nocc.png'):
                    occ_path = gt_file.replace('disp0GT.pfm', 'mask0nocc.png')
                    if os.path.exists(occ_path):
                        occ_mask = np.array(Image.open(occ_path).convert('L')) == 255
            elif args.dataset.startswith('middlebury'):
                imgL_file = meta[0]
                occ_path = imgL_file.replace('im0.png', 'mask0nocc.png')
                if os.path.exists(occ_path):
                    occ_mask = np.array(Image.open(occ_path).convert('L')) == 255

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            with autocast(enabled=args.mixed_precision):
                flow_pr = model(image1, image2, iters=args.valid_iters, test_mode=True)

            flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)  # [1, H, W] or [H, W]
            if flow_pr.dim() == 3:
                flow_pr = flow_pr[0]

            flow_gt_np = flow_gt[0].numpy() if flow_gt.dim() == 3 else flow_gt.numpy()
            valid_np = valid_gt[0].numpy() >= 0.5 if valid_gt.dim() == 3 else valid_gt.numpy() >= 0.5
            flow_pr_np = flow_pr.numpy()

            # 有效 mask：valid 且 disp < max_disp
            val = valid_np & (np.abs(flow_gt_np) < max_disp)
            if occ_mask is not None:
                if occ_mask.shape != val.shape:
                    occ_mask = cv2.resize(occ_mask.astype(np.uint8), (val.shape[1], val.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                val = val & occ_mask

            # EPE
            epe = np.abs(flow_pr_np - flow_gt_np)
            epe[~val] = np.nan  # 无效区域不参与可视化时的颜色映射

            # 用于 colormap 的 mask：仅在有 GT 的区域显示误差
            valid_for_viz = val

            # 生成 error map
            epe_for_cmap = np.nan_to_num(epe, nan=0.0)
            errormap_rgb = error_to_colormap(epe_for_cmap, valid_for_viz, vmax=vmax)

            # 文件名
            if args.dataset == 'kitti':
                stem = Path(meta[0]).stem
            elif args.dataset == 'eth3d':
                stem = Path(meta[0]).parent.name
            elif args.dataset.startswith('middlebury'):
                stem = Path(meta[0]).parent.name
            else:
                stem = f"sample_{val_id:04d}"

            # 保存 error map
            plt.imsave(output_dir / f"{stem}_errormap.png", errormap_rgb.astype(np.uint8) / 255.0)
            if args.save_overlay:
                img_path = meta[0]
                img_rgb = np.array(Image.open(img_path).convert('RGB'))
                if img_rgb.shape[:2] != errormap_rgb.shape[:2]:
                    errormap_resized = cv2.resize(errormap_rgb, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
                else:
                    errormap_resized = errormap_rgb
                overlay = overlay_errormap(img_rgb, errormap_resized, alpha=args.overlay_alpha)
                Image.fromarray(overlay).save(output_dir / f"{stem}_overlay.png")
            if args.save_numpy:
                np.save(output_dir / f"{stem}_epe.npy", epe)

    print(f"Saved errormaps to {output_dir}")


def demo_manual(args):
    """手动指定 left/right/gt 路径的 Error Map 生成"""
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    ckpt = torch.load(args.restore_ckpt, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt, strict=False)
    model = model.module
    model.to(DEVICE)
    model.eval()

    import glob
    left_list = sorted(glob.glob(args.left_imgs, recursive=True))
    right_list = sorted(glob.glob(args.right_imgs, recursive=True))
    gt_list = sorted(glob.glob(args.disp_gt, recursive=True)) if args.disp_gt else []

    if len(gt_list) != len(left_list):
        gt_list = [None] * len(left_list)  # 无 GT 时只保存预测

    output_dir = Path(args.output_directory)
    output_dir.mkdir(exist_ok=True, parents=True)
    vmax = args.error_vmax
    max_disp = getattr(args, 'max_disp', 192)

    def read_disp(path):
        ext = os.path.splitext(path)[-1].lower()
        if ext == '.pfm':
            from utils.frame_utils import readPFM
            d = readPFM(path)
            return np.array(d).astype(np.float32), np.ones_like(d, dtype=bool)
        elif ext == '.png':
            disp = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
            if disp is None:
                return None, None
            if disp.dtype == np.uint16:
                disp = disp.astype(np.float32) / 256.0
            valid = disp > 0
            return disp.astype(np.float32), valid
        return None, None

    with torch.no_grad():
        for i, (imL, imR) in enumerate(tqdm(list(zip(left_list, right_list)), desc="Errormap")):
            if args.max_samples and i >= args.max_samples:
                break
            image1 = load_image(imL)
            image2 = load_image(imR)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            with autocast(enabled=args.mixed_precision):
                flow_pr = model(image1, image2, iters=args.valid_iters, test_mode=True)

            flow_pr = padder.unpad(flow_pr).cpu().squeeze(0).numpy()
            if flow_pr.ndim == 3:
                flow_pr = flow_pr[0]

            stem = Path(imL).stem
            if stem == Path(imL).parent.name:
                stem = f"{Path(imL).parent.name}_{stem}"

            if i < len(gt_list) and gt_list[i]:
                disp_gt, valid = read_disp(gt_list[i])
                if disp_gt is not None:
                    if disp_gt.shape != flow_pr.shape:
                        flow_pr = cv2.resize(flow_pr, (disp_gt.shape[1], disp_gt.shape[0]), interpolation=cv2.INTER_LINEAR)
                    val = valid & (np.abs(disp_gt) < max_disp)
                    epe = np.abs(flow_pr - disp_gt)
                    epe[~val] = 0
                    errormap_rgb = error_to_colormap(epe, val, vmax=vmax)
                    plt.imsave(output_dir / f"{stem}_errormap.png", errormap_rgb.astype(np.uint8) / 255.0)
                    if args.save_overlay:
                        img_rgb = np.array(Image.open(imL).convert('RGB'))
                        if img_rgb.shape[:2] != errormap_rgb.shape[:2]:
                            errormap_resized = cv2.resize(errormap_rgb, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
                        else:
                            errormap_resized = errormap_rgb
                        overlay = overlay_errormap(img_rgb, errormap_resized, alpha=args.overlay_alpha)
                        Image.fromarray(overlay).save(output_dir / f"{stem}_overlay.png")
                    if args.save_numpy:
                        np.save(output_dir / f"{stem}_epe.npy", epe)

    print(f"Saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate stereo matching error maps (EPE heatmaps)')
    parser.add_argument('--mode', choices=['dataset', 'manual', 'folder'], default='dataset',
                        help='dataset: built-in; manual: left/right/disp_gt glob; folder: SceneFlow TEST 等按原结构')
    parser.add_argument('--restore_ckpt', required=True, help='model checkpoint path')
    parser.add_argument('--output_directory', default='./demo-errormap/', help='output directory')
    parser.add_argument('--folder', help='folder 模式：输入目录，如 frames_finalpass/TEST')
    parser.add_argument('--data_root', help='folder 模式：sceneflow 根目录（含 frames_finalpass、disparity）')
    parser.add_argument('--pred_dir', help='folder 模式：预计算 disparity 的 .npy 目录，有则跳过推理')
    parser.add_argument('--dataset', default='kitti', help='for mode=dataset: kitti, sceneflow, eth3d, middlebury_F/H/Q')
    parser.add_argument('-l', '--left_imgs', default='./demo-imgs/*/im0.png', help='for mode=manual: left image glob')
    parser.add_argument('-r', '--right_imgs', default='./demo-imgs/*/im1.png', help='for mode=manual: right image glob')
    parser.add_argument('--disp_gt', default='', help='for mode=manual: GT disparity glob (e.g. ./demo-imgs/*/disp0GT.pfm)')
    parser.add_argument('--max_samples', type=int, default=None, help='max samples to process (default: all)')
    parser.add_argument('--error_vmax', type=float, default=5.0, help='EPE colormap max (pixels)')
    parser.add_argument('--save_overlay', action='store_true', help='save overlay of errormap on image')
    parser.add_argument('--overlay_alpha', type=float, default=0.5, help='overlay alpha (0-1)')
    parser.add_argument('--save_numpy', action='store_true', help='save EPE as .npy')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--valid_iters', type=int, default=32)

    # Architecture (match checkpoint)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3)
    parser.add_argument('--corr_levels', type=int, default=2)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--n_gru_layers', type=int, default=3)
    parser.add_argument('--max_disp', type=int, default=192)
    parser.add_argument('--edge_source', type=str, default='shared', choices=['rcf', 'gt', 'geo', 'shared'])
    parser.add_argument('--edge_init_from_geo', type=str, default=None)
    parser.add_argument('--edge_model', type=str, default='../RCF-PyTorch/rcf.pth')
    parser.add_argument('--edge_context_fusion', action='store_true')
    parser.add_argument('--edge_fusion_mode', type=str, default='film', choices=['concat', 'film', 'gated'])
    parser.add_argument('--edge_floor', type=float, default=0.0)
    parser.add_argument('--edge_context_film_gamma_min', type=float, default=0.0)
    parser.add_argument('--edge_guided_upsample', action='store_true')
    parser.add_argument('--edge_upsample_fusion_mode', type=str, default='film', choices=['concat', 'film', 'gated', 'mlp'])
    parser.add_argument('--edge_guided_disp_head', action='store_true')
    parser.add_argument('--edge_disp_fusion_mode', type=str, default='film', choices=['concat', 'film', 'gated', 'mlp'])
    parser.add_argument('--edge_guided_cost_agg', action='store_true')
    parser.add_argument('--edge_cost_agg_fusion_mode', type=str, default='film', choices=['concat', 'film', 'gated'])
    parser.add_argument('--edge_guided_gwc', action='store_true')
    parser.add_argument('--edge_gwc_fusion_mode', type=str, default='film', choices=['concat', 'film', 'gated'])
    parser.add_argument('--edge_motion_encoder', action='store_true')
    parser.add_argument('--edge_motion_fusion_mode', type=str, default='film', choices=['concat', 'film', 'gated'])
    parser.add_argument('--edge_guided_refinement', action='store_true')
    parser.add_argument('--boundary_only_refinement', action='store_true')
    parser.add_argument('--edge_refinement_fusion_mode', type=str, default='film', choices=['concat', 'film', 'gated'])
    parser.add_argument('--feature_edge_x4_film', action='store_true')
    parser.add_argument('--feature_edge_x4_film_strength', type=float, default=1.0)
    parser.add_argument('--edge_geo_radius_aware', action='store_true')
    parser.add_argument('--edge_geo_radius_shrink', type=float, default=0.5)

    args = parser.parse_args()

    if args.mode == 'dataset':
        demo_dataset(args)
    elif args.mode == 'folder':
        if not args.folder:
            raise SystemExit("--folder required for mode=folder")
        demo_folder(args)
    else:
        demo_manual(args)
