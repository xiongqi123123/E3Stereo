from __future__ import print_function, division
import sys
import os
import argparse
import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from igev_stereo import IGEVStereo, autocast
import stereo_datasets as datasets
from utils import InputPadder
from PIL import Image
from tqdm import tqdm

# Default dataset paths
DEFAULT_DATASET_ROOT = '/root/autodl-tmp/stereo/dataset_cache'
DEFAULT_SCENEFLOW_ROOT = os.path.join(DEFAULT_DATASET_ROOT, 'SceneFlow')
DEFAULT_KITTI_ROOT = os.path.join(DEFAULT_DATASET_ROOT, 'KITTI')
DEFAULT_ETH3D_ROOT = os.path.join(DEFAULT_DATASET_ROOT, 'ETH3D')
DEFAULT_MIDDLEBURY_ROOT = os.path.join(DEFAULT_DATASET_ROOT, 'Middlebury/MiddEval3')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_image_range(image, name="Image"):
    """Debug helper to check input value range"""
    if isinstance(image, list): return
    min_val, max_val = image.min().item(), image.max().item()
    if max_val <= 1.1:
        logging.warning(f"⚠️ [WARNING] {name} range is [{min_val:.4f}, {max_val:.4f}]. "
                        f"IGEV/RAFT typically expects [0, 255]. If your DataLoader uses ToTensor(), "
                        f"input is 0-1, which will cause POOR performance.")


@torch.no_grad()
def validate_eth3d(model, iters=32, mixed_prec=False, device=None, root=None, val_samples=500):
    if root is None: root = DEFAULT_ETH3D_ROOT
    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    val_dataset = datasets.ETH3D({}, root=root)
    total_samples = min(val_samples, len(val_dataset)) if val_samples > 0 else len(val_dataset)

    total_epe_sum = 0.0
    total_valid_pixels = 0
    total_d1_pixels = 0

    print(f"Evaluating ETH3D on {total_samples} samples...")

    for val_id in range(total_samples):
        (imageL_file, _, GT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]

        if val_id == 0: check_image_range(image1, "ETH3D Image1")

        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)

        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)

        if flow_pr.dim() == 3: flow_pr = flow_pr.squeeze(0)
        if flow_gt.dim() == 3: flow_gt = flow_gt.squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

        epe_map = torch.abs(flow_pr - flow_gt)

        occ_mask = Image.open(GT_file.replace('disp0GT.pfm', 'mask0nocc.png'))
        occ_mask = np.ascontiguousarray(occ_mask).flatten()

        valid_gt_flat = valid_gt.flatten()
        epe_flat = epe_map.flatten()

        val = (valid_gt_flat >= 0.5) & (occ_mask == 255)

        if val.sum() > 0:
            valid_err = epe_flat[val]
            total_epe_sum += valid_err.sum().item()
            total_valid_pixels += val.sum().item()
            total_d1_pixels += (valid_err > 1.0).float().sum().item()

    global_epe = total_epe_sum / total_valid_pixels if total_valid_pixels > 0 else 0
    global_d1 = 100 * (total_d1_pixels / total_valid_pixels) if total_valid_pixels > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"ETH3D Validation Results ({total_samples} samples)")
    print(f"{'=' * 60}")
    print(f"  EPE: {global_epe:.4f}")
    print(f"  D1 (>1px): {global_d1:.2f}%")
    print(f"{'=' * 60}\n")
    return {'eth3d-epe': global_epe, 'eth3d-d1': global_d1}


@torch.no_grad()
def validate_kitti(model, iters=32, mixed_prec=False, device=None, root=None, val_samples=500):
    if root is None: root = DEFAULT_KITTI_ROOT
    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    val_dataset = datasets.KITTI({}, root=root, image_set='training')

    total_samples = min(val_samples, len(val_dataset)) if val_samples > 0 else len(val_dataset)

    total_epe_sum = 0.0
    total_valid_pixels = 0
    total_d1_pixels = 0
    elapsed_list = []

    print(f"Evaluating KITTI on {total_samples} samples...")

    for val_id in range(total_samples):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]

        if val_id == 0: check_image_range(image1, "KITTI Image1")

        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()

        if val_id > 50: elapsed_list.append(end - start)

        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        if flow_pr.dim() == 3: flow_pr = flow_pr.squeeze(0)
        if flow_gt.dim() == 3: flow_gt = flow_gt.squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

        epe_map = torch.abs(flow_pr - flow_gt)
        mask = (valid_gt.squeeze() >= 0.5) & (flow_gt < 192)

        if mask.sum() > 0:
            valid_err = epe_map[mask]
            total_epe_sum += valid_err.sum().item()
            total_valid_pixels += mask.sum().item()
            total_d1_pixels += (valid_err > 3.0).float().sum().item()

    global_epe = total_epe_sum / total_valid_pixels if total_valid_pixels > 0 else 0
    global_d1 = 100 * (total_d1_pixels / total_valid_pixels) if total_valid_pixels > 0 else 0
    avg_runtime = np.mean(elapsed_list) if elapsed_list else 0.0

    print(f"\n{'=' * 60}")
    print(f"KITTI Validation Results ({total_samples} samples)")
    print(f"{'=' * 60}")
    print(f"  EPE: {global_epe:.4f}")
    print(f"  D1 (>3px): {global_d1:.2f}%")
    if avg_runtime > 0:
        print(f"  FPS: {1 / avg_runtime:.2f} ({avg_runtime * 1000:.1f}ms/frame)")
    print(f"{'=' * 60}\n")
    return {'kitti-epe': global_epe, 'kitti-d1': global_d1}


@torch.no_grad()
def validate_sceneflow(model, iters=32, mixed_prec=False, device=None, root=None, val_samples=500, batch_size=4):
    if root is None: root = DEFAULT_SCENEFLOW_ROOT
    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    val_dataset = datasets.SceneFlowDatasets(root=root, dstype='frames_finalpass', things_test=True)

    try:
        sample_data = val_dataset[0]
        if len(sample_data) > 1 and isinstance(sample_data[1], torch.Tensor):
            check_image_range(sample_data[1], "SceneFlow Image")
        else:
            for item in sample_data:
                if isinstance(item, torch.Tensor) and item.ndim == 3:
                    check_image_range(item, "SceneFlow Image (detected)")
                    break
    except Exception as e:
        print(f"Skipping range check due to: {e}")

    total_samples = min(val_samples, len(val_dataset)) if val_samples > 0 else len(val_dataset)

    if val_samples > 0 and val_samples < len(val_dataset):
        val_dataset = Subset(val_dataset, range(total_samples))

    num_workers = min(batch_size * 4, 16)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False,
                            persistent_workers=True, prefetch_factor=2)

    total_epe_sum = 0.0
    total_valid_pixels = 0
    total_d1_pixels = 0

    total_epe_edge_sum = 0.0
    total_edge_pixels = 0
    total_epe_flat_sum = 0.0
    total_flat_pixels = 0

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

    print(f"Evaluating {total_samples} samples with batch_size={batch_size}...")

    for batch_data in tqdm(val_loader, desc="SceneFlow Validation", total=len(val_loader)):
        _, image1_batch, image2_batch, flow_gt_batch, valid_gt_batch = batch_data

        image1_gpu = image1_batch.to(device, non_blocking=True)
        image2_gpu = image2_batch.to(device, non_blocking=True)
        flow_gt_gpu = flow_gt_batch.to(device, non_blocking=True)
        valid_gt_gpu = valid_gt_batch.to(device, non_blocking=True)

        # Edge Extraction
        gray = 0.299 * image1_gpu[:, 0:1] + 0.587 * image1_gpu[:, 1:2] + 0.114 * image1_gpu[:, 2:3]
        if gray.max() > 1.0: gray = gray / 255.0

        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        edge_maps = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8).squeeze(1)

        q98 = torch.quantile(edge_maps.flatten(1), 0.98, dim=1).view(-1, 1, 1)
        edge_maps = edge_maps / (q98 + 1e-8)

        edge_bool = (edge_maps > 0.3)
        flat_bool = (edge_maps <= 0.3)

        # Padding
        h, w = image1_gpu.shape[2], image1_gpu.shape[3]
        pad_h = ((h // 32) + 1) * 32 if h % 32 != 0 else h
        pad_w = ((w // 32) + 1) * 32 if w % 32 != 0 else w

        if pad_h != h or pad_w != w:
            image1_padded = F.pad(image1_gpu, (0, pad_w - w, 0, pad_h - h))
            image2_padded = F.pad(image2_gpu, (0, pad_w - w, 0, pad_h - h))
        else:
            image1_padded, image2_padded = image1_gpu, image2_gpu

        # Inference
        with autocast(enabled=mixed_prec):
            flow_pr_batch = model(image1_padded, image2_padded, iters=iters, test_mode=True)

        if pad_h != h or pad_w != w:
            flow_pr_batch = flow_pr_batch[..., :h, :w]

        if flow_pr_batch.dim() == 4:
            flow_pr_batch = flow_pr_batch.squeeze(1)

        # === FIX: Ensure dimensions match [B, H, W] for mask ===
        # flow_gt_gpu is usually [B, 1, H, W], we must squeeze it before comparison
        flow_gt_squeezed = flow_gt_gpu.squeeze(1)
        valid_gt_squeezed = valid_gt_gpu.squeeze(1)

        valid_mask = (valid_gt_squeezed >= 0.5) & (flow_gt_squeezed.abs() < 192)

        if valid_mask.sum() == 0: continue

        error_batch = torch.abs(flow_pr_batch - flow_gt_squeezed)

        # Now both error_batch and valid_mask are [B, H, W]
        valid_errors = error_batch[valid_mask]

        total_epe_sum += valid_errors.sum().item()
        total_valid_pixels += valid_errors.numel()
        total_d1_pixels += (valid_errors > 3.0).float().sum().item()

        mask_edge = valid_mask & edge_bool
        mask_flat = valid_mask & flat_bool

        if mask_edge.sum() > 0:
            total_epe_edge_sum += error_batch[mask_edge].sum().item()
            total_edge_pixels += mask_edge.sum().item()

        if mask_flat.sum() > 0:
            total_epe_flat_sum += error_batch[mask_flat].sum().item()
            total_flat_pixels += mask_flat.sum().item()

    global_epe = total_epe_sum / total_valid_pixels if total_valid_pixels > 0 else 0
    global_d1 = 100 * (total_d1_pixels / total_valid_pixels) if total_valid_pixels > 0 else 0

    global_epe_edge = total_epe_edge_sum / total_edge_pixels if total_edge_pixels > 0 else 0
    global_epe_flat = total_epe_flat_sum / total_flat_pixels if total_flat_pixels > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"SceneFlow Validation Results ({total_samples} samples)")
    print(f"{'=' * 60}")
    print(f"  EPE:       {global_epe:.4f}")
    print(f"  EPE_edge:  {global_epe_edge:.4f}")
    print(f"  EPE_flat:  {global_epe_flat:.4f}")
    print(f"  Ratio:     {global_epe_edge / global_epe_flat:.2f}x" if global_epe_flat > 0 else "N/A")
    print(f"  D1 (>3px): {global_d1:.2f}%")
    print(f"{'=' * 60}\n")
    return {'scene-disp-epe': global_epe, 'scene-disp-d1': global_d1}


@torch.no_grad()
def validate_middlebury(model, iters=32, split='F', mixed_prec=False, device=None, root=None, val_samples=500):
    if root is None: root = DEFAULT_MIDDLEBURY_ROOT
    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    val_dataset = datasets.Middlebury({}, root=root, split=split)
    total_samples = min(val_samples, len(val_dataset)) if val_samples > 0 else len(val_dataset)

    total_epe_sum = 0.0
    total_valid_pixels = 0
    total_d1_pixels = 0

    print(f"Evaluating Middlebury-{split} on {total_samples} samples...")

    for val_id in range(total_samples):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]

        if val_id == 0: check_image_range(image1, "MB Image")

        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)

        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

        epe_map = torch.abs(flow_pr - flow_gt)

        occ_mask = Image.open(imageL_file.replace('im0.png', 'mask0nocc.png')).convert('L')
        occ_mask = np.ascontiguousarray(occ_mask, dtype=np.float32).flatten()

        val = (valid_gt.reshape(-1) >= 0.5) & (flow_gt[0].reshape(-1) < 192) & (occ_mask == 255)

        if val.sum() > 0:
            valid_err = epe_map.flatten()[val]
            total_epe_sum += valid_err.sum().item()
            total_valid_pixels += val.sum().item()
            total_d1_pixels += (valid_err > 2.0).float().sum().item()

    global_epe = total_epe_sum / total_valid_pixels if total_valid_pixels > 0 else 0
    global_d1 = 100 * (total_d1_pixels / total_valid_pixels) if total_valid_pixels > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"Middlebury-{split} Validation Results ({total_samples} samples)")
    print(f"{'=' * 60}")
    print(f"  EPE: {global_epe:.4f}")
    print(f"  D1 (>2px): {global_d1:.2f}%")
    print(f"{'=' * 60}\n")
    return {f'middlebury{split}-epe': global_epe, f'middlebury{split}-d1': global_d1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    # === PATH UPDATED HERE ===
    parser.add_argument('--restore_ckpt', help="restore checkpoint",
                        # default='/root/autodl-tmp/stereo/logs/edge_cpt/64000_edge_cpt.pth'
                        # default='/root/autodl-tmp/stereo/logs/edge_d1_26/188500_igev_edge_pt_2_6.pth'
                        default='/root/autodl-tmp/stereo/logs/ours3_depth_aware_edge/65000_depth_aware_edge.pth'
                        # default='/root/autodl-tmp/stereo/model_cache/sceneflow.pth'
                        )

    parser.add_argument('--dataset', help="dataset for evaluation", default='sceneflow',
                        choices=["eth3d", "kitti", "sceneflow"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='bfloat16', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--val_samples', type=int, default=0, help='number of samples to validate (0 for all)')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size for evaluation')
    parser.add_argument('--no_compile', action='store_true', help='disable torch.compile')

    parser.add_argument('--dataset_root', type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument('--sceneflow_root', type=str, default=None)
    parser.add_argument('--kitti_root', type=str, default=None)
    parser.add_argument('--eth3d_root', type=str, default=None)
    parser.add_argument('--middlebury_root', type=str, default=None)

    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3)
    parser.add_argument('--corr_levels', type=int, default=2)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--n_gru_layers', type=int, default=3)
    parser.add_argument('--max_disp', type=int, default=192)
    args = parser.parse_args()

    if args.sceneflow_root is None: args.sceneflow_root = os.path.join(args.dataset_root, 'SceneFlow')
    if args.kitti_root is None: args.kitti_root = os.path.join(args.dataset_root, 'KITTI')
    if args.eth3d_root is None: args.eth3d_root = os.path.join(args.dataset_root, 'ETH3D')
    if args.middlebury_root is None: args.middlebury_root = os.path.join(args.dataset_root, 'Middlebury/MiddEval3')

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    model = IGEVStereo(args)
    logging.info(f"Using device: {args.device}")

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info(f"Loading checkpoint: {args.restore_ckpt}")
        checkpoint = torch.load(args.restore_ckpt, map_location=args.device)

        state_dict = checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']

        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict, strict=True)
        logging.info(f"Done loading checkpoint")

    model.to(args.device)
    model.eval()

    if not args.no_compile and hasattr(torch, 'compile'):
        logging.info("Compiling model (first batch will be slow)...")
        model = torch.compile(model, mode='reduce-overhead')

    print(f"The model has {format(count_parameters(model) / 1e6, '.2f')}M learnable parameters.")

    if args.dataset == 'eth3d':
        validate_eth3d(model, iters=args.valid_iters, mixed_prec=args.mixed_precision, device=args.device,
                       root=args.eth3d_root, val_samples=args.val_samples)

    elif args.dataset == 'kitti':
        validate_kitti(model, iters=args.valid_iters, mixed_prec=args.mixed_precision, device=args.device,
                       root=args.kitti_root, val_samples=args.val_samples)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, iters=args.valid_iters, split=args.dataset[-1], mixed_prec=args.mixed_precision,
                            device=args.device, root=args.middlebury_root, val_samples=args.val_samples)

    elif args.dataset == 'sceneflow':
        validate_sceneflow(model, iters=args.valid_iters, mixed_prec=args.mixed_precision, device=args.device,
                           root=args.sceneflow_root, val_samples=args.val_samples, batch_size=args.batch_size)