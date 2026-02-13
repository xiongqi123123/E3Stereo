from __future__ import print_function, division
import sys
import os
import argparse
import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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


def extract_edge_sobel_batch(image):
    """
    Batch-wise Edge Extraction (Sobel) - 参照IGEV标准实现
    Input: [B, 3, H, W]
    Output: [B, H, W] normalized edge map
    """
    # 转灰度
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    if gray.max() > 1.0:
        gray = gray / 255.0

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)

    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    # ===== 关键修复：使用quantile(0.98)归一化（参照IGEV） =====
    B, C, H, W = edge.shape
    edge_list = []
    for b in range(B):
        edge_b = edge[b].squeeze()  # [H, W]
        edge_b = edge_b / (edge_b.quantile(0.98) + 1e-8)
        edge_b = torch.clamp(edge_b, 0, 1)
        edge_list.append(edge_b)

    return torch.stack(edge_list)  # [B, H, W]


@torch.no_grad()
def validate_eth3d(model, iters=32, mixed_prec=False, device=None, root=None, val_samples=500, batch_size=1):
    """
    ETH3D Validation with comprehensive metrics
    - Image-level and Pixel-level EPE (overall, edge, flat)
    - Separated by noc (non-occluded) and all regions
    - Bad 1.0 metric (>1px error rate)
    """
    if root is None: root = DEFAULT_ETH3D_ROOT
    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    val_dataset = datasets.ETH3D(aug_params=None, root=root)
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    edge_threshold = 0.3

    # Pixel-level accumulators for NOC (non-occluded)
    pix_noc_all_epe, pix_noc_all_cnt, pix_noc_all_bad1 = 0.0, 0, 0
    pix_noc_edge_epe, pix_noc_edge_cnt, pix_noc_edge_bad1 = 0.0, 0, 0
    pix_noc_flat_epe, pix_noc_flat_cnt, pix_noc_flat_bad1 = 0.0, 0, 0

    # Pixel-level accumulators for ALL (including occluded)
    pix_all_all_epe, pix_all_all_cnt, pix_all_all_bad1 = 0.0, 0, 0
    pix_all_edge_epe, pix_all_edge_cnt, pix_all_edge_bad1 = 0.0, 0, 0
    pix_all_flat_epe, pix_all_flat_cnt, pix_all_flat_bad1 = 0.0, 0, 0

    # Image-level lists for NOC
    img_noc_all_epe, img_noc_all_bad1 = [], []
    img_noc_edge_epe, img_noc_edge_bad1 = [], []
    img_noc_flat_epe, img_noc_flat_bad1 = [], []

    # Image-level lists for ALL
    img_all_all_epe, img_all_all_bad1 = [], []
    img_all_edge_epe, img_all_edge_bad1 = [], []
    img_all_flat_epe, img_all_flat_bad1 = [], []

    print(f"Start validation on ETH3D...")

    for i, data_blob in enumerate(tqdm(loader, desc="ETH3D")):
        if val_samples > 0 and i >= val_samples: break

        (imageL_file, imageR_file, GT_file), image1, image2, flow_gt, valid_gt = data_blob

        # Extract edge map before padding
        edge_map = extract_edge_sobel_batch(image1).cpu()  # [B, H, W]

        image1, image2 = image1.to(device), image2.to(device)
        flow_gt, valid_gt = flow_gt.to(device), valid_gt.to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)

        flow_pr = padder.unpad(flow_pr.float()).cpu()
        flow_gt = flow_gt.cpu()
        valid_gt = valid_gt.cpu()

        # Load occlusion mask
        mask_path = GT_file[0].replace('disp0GT.pfm', 'mask0nocc.png')
        if os.path.exists(mask_path):
            occ_mask = np.array(Image.open(mask_path))
            occ_mask = torch.from_numpy(occ_mask)
            noc_mask_bool = (occ_mask == 255)
        else:
            noc_mask_bool = torch.ones_like(valid_gt.squeeze(), dtype=torch.bool)

        # Ensure dimension consistency
        if flow_pr.dim() == 4: flow_pr = flow_pr.squeeze(1)
        if flow_gt.dim() == 4: flow_gt = flow_gt.squeeze(1)
        if valid_gt.dim() == 4: valid_gt = valid_gt.squeeze(1)

        epe = (flow_pr - flow_gt).abs()

        # Process each image in batch (BS=1, so just one iteration)
        for b in range(flow_pr.shape[0]):
            b_epe = epe[b]
            b_edge_map = edge_map[b]
            b_valid = valid_gt[b] >= 0.5
            b_noc = noc_mask_bool if noc_mask_bool.dim() == 2 else noc_mask_bool[b]

            # ALL region (valid pixels)
            mask_all = b_valid
            if mask_all.sum() > 0:
                all_epe_vals = b_epe[mask_all]
                pix_all_all_epe += all_epe_vals.sum().item()
                pix_all_all_cnt += mask_all.sum().item()
                pix_all_all_bad1 += (all_epe_vals > 1.0).sum().item()
                img_all_all_epe.append(all_epe_vals.mean().item())
                img_all_all_bad1.append((all_epe_vals > 1.0).float().mean().item())

                # Edge region (ALL)
                mask_all_edge = (b_edge_map > edge_threshold) & mask_all
                if mask_all_edge.sum() > 0:
                    edge_epe_vals = b_epe[mask_all_edge]
                    pix_all_edge_epe += edge_epe_vals.sum().item()
                    pix_all_edge_cnt += mask_all_edge.sum().item()
                    pix_all_edge_bad1 += (edge_epe_vals > 1.0).sum().item()
                    img_all_edge_epe.append(edge_epe_vals.mean().item())
                    img_all_edge_bad1.append((edge_epe_vals > 1.0).float().mean().item())

                # Flat region (ALL)
                mask_all_flat = (b_edge_map <= edge_threshold) & mask_all
                if mask_all_flat.sum() > 0:
                    flat_epe_vals = b_epe[mask_all_flat]
                    pix_all_flat_epe += flat_epe_vals.sum().item()
                    pix_all_flat_cnt += mask_all_flat.sum().item()
                    pix_all_flat_bad1 += (flat_epe_vals > 1.0).sum().item()
                    img_all_flat_epe.append(flat_epe_vals.mean().item())
                    img_all_flat_bad1.append((flat_epe_vals > 1.0).float().mean().item())

            # NOC region (non-occluded)
            mask_noc = b_valid & b_noc
            if mask_noc.sum() > 0:
                noc_epe_vals = b_epe[mask_noc]
                pix_noc_all_epe += noc_epe_vals.sum().item()
                pix_noc_all_cnt += mask_noc.sum().item()
                pix_noc_all_bad1 += (noc_epe_vals > 1.0).sum().item()
                img_noc_all_epe.append(noc_epe_vals.mean().item())
                img_noc_all_bad1.append((noc_epe_vals > 1.0).float().mean().item())

                # Edge region (NOC)
                mask_noc_edge = (b_edge_map > edge_threshold) & mask_noc
                if mask_noc_edge.sum() > 0:
                    edge_epe_vals = b_epe[mask_noc_edge]
                    pix_noc_edge_epe += edge_epe_vals.sum().item()
                    pix_noc_edge_cnt += mask_noc_edge.sum().item()
                    pix_noc_edge_bad1 += (edge_epe_vals > 1.0).sum().item()
                    img_noc_edge_epe.append(edge_epe_vals.mean().item())
                    img_noc_edge_bad1.append((edge_epe_vals > 1.0).float().mean().item())

                # Flat region (NOC)
                mask_noc_flat = (b_edge_map <= edge_threshold) & mask_noc
                if mask_noc_flat.sum() > 0:
                    flat_epe_vals = b_epe[mask_noc_flat]
                    pix_noc_flat_epe += flat_epe_vals.sum().item()
                    pix_noc_flat_cnt += mask_noc_flat.sum().item()
                    pix_noc_flat_bad1 += (flat_epe_vals > 1.0).sum().item()
                    img_noc_flat_epe.append(flat_epe_vals.mean().item())
                    img_noc_flat_bad1.append((flat_epe_vals > 1.0).float().mean().item())

    # Calculate final metrics
    # Pixel-level NOC
    pix_noc_epe_all = pix_noc_all_epe / (pix_noc_all_cnt + 1e-12)
    pix_noc_epe_edge = pix_noc_edge_epe / (pix_noc_edge_cnt + 1e-12)
    pix_noc_epe_flat = pix_noc_flat_epe / (pix_noc_flat_cnt + 1e-12)
    pix_noc_bad1_all = 100 * pix_noc_all_bad1 / (pix_noc_all_cnt + 1e-12)
    pix_noc_bad1_edge = 100 * pix_noc_edge_bad1 / (pix_noc_edge_cnt + 1e-12)
    pix_noc_bad1_flat = 100 * pix_noc_flat_bad1 / (pix_noc_flat_cnt + 1e-12)

    # Pixel-level ALL
    pix_all_epe_all = pix_all_all_epe / (pix_all_all_cnt + 1e-12)
    pix_all_epe_edge = pix_all_edge_epe / (pix_all_edge_cnt + 1e-12)
    pix_all_epe_flat = pix_all_flat_epe / (pix_all_flat_cnt + 1e-12)
    pix_all_bad1_all = 100 * pix_all_all_bad1 / (pix_all_all_cnt + 1e-12)
    pix_all_bad1_edge = 100 * pix_all_edge_bad1 / (pix_all_edge_cnt + 1e-12)
    pix_all_bad1_flat = 100 * pix_all_flat_bad1 / (pix_all_flat_cnt + 1e-12)

    # Image-level NOC
    img_noc_epe_all = np.mean(img_noc_all_epe) if img_noc_all_epe else 0
    img_noc_epe_edge = np.mean(img_noc_edge_epe) if img_noc_edge_epe else 0
    img_noc_epe_flat = np.mean(img_noc_flat_epe) if img_noc_flat_epe else 0
    img_noc_bad1_all = 100 * np.mean(img_noc_all_bad1) if img_noc_all_bad1 else 0
    img_noc_bad1_edge = 100 * np.mean(img_noc_edge_bad1) if img_noc_edge_bad1 else 0
    img_noc_bad1_flat = 100 * np.mean(img_noc_flat_bad1) if img_noc_flat_bad1 else 0

    # Image-level ALL
    img_all_epe_all = np.mean(img_all_all_epe) if img_all_all_epe else 0
    img_all_epe_edge = np.mean(img_all_edge_epe) if img_all_edge_epe else 0
    img_all_epe_flat = np.mean(img_all_flat_epe) if img_all_flat_epe else 0
    img_all_bad1_all = 100 * np.mean(img_all_all_bad1) if img_all_all_bad1 else 0
    img_all_bad1_edge = 100 * np.mean(img_all_edge_bad1) if img_all_edge_bad1 else 0
    img_all_bad1_flat = 100 * np.mean(img_all_flat_bad1) if img_all_flat_bad1 else 0

    # Print results
    print(f"\n{'='*90}")
    print(f"ETH3D Validation Results ({len(img_all_all_epe)} images)")
    print(f"{'='*90}")
    print(f"{'Metric':<30} {'Overall':>12} {'Edge':>12} {'Flat':>12}")
    print(f"{'-'*90}")
    print(f"{'Image-level EPE (NOC)':<30} {img_noc_epe_all:>12.4f} {img_noc_epe_edge:>12.4f} {img_noc_epe_flat:>12.4f}")
    print(f"{'Pixel-level EPE (NOC)':<30} {pix_noc_epe_all:>12.4f} {pix_noc_epe_edge:>12.4f} {pix_noc_epe_flat:>12.4f}")
    print(f"{'Image-level EPE (ALL)':<30} {img_all_epe_all:>12.4f} {img_all_epe_edge:>12.4f} {img_all_epe_flat:>12.4f}")
    print(f"{'Pixel-level EPE (ALL)':<30} {pix_all_epe_all:>12.4f} {pix_all_epe_edge:>12.4f} {pix_all_epe_flat:>12.4f}")
    print(f"{'-'*90}")
    print(f"{'Image-level Bad1.0 (NOC)':<30} {img_noc_bad1_all:>11.2f}% {img_noc_bad1_edge:>11.2f}% {img_noc_bad1_flat:>11.2f}%")
    print(f"{'Pixel-level Bad1.0 (NOC)':<30} {pix_noc_bad1_all:>11.2f}% {pix_noc_bad1_edge:>11.2f}% {pix_noc_bad1_flat:>11.2f}%")
    print(f"{'Image-level Bad1.0 (ALL)':<30} {img_all_bad1_all:>11.2f}% {img_all_bad1_edge:>11.2f}% {img_all_bad1_flat:>11.2f}%")
    print(f"{'Pixel-level Bad1.0 (ALL)':<30} {pix_all_bad1_all:>11.2f}% {pix_all_bad1_edge:>11.2f}% {pix_all_bad1_flat:>11.2f}%")
    print(f"{'='*90}\n")

    return {
        # Image-level EPE
        'eth3d-img-epe-noc-all': img_noc_epe_all,
        'eth3d-img-epe-noc-edge': img_noc_epe_edge,
        'eth3d-img-epe-noc-flat': img_noc_epe_flat,
        'eth3d-img-epe-all-all': img_all_epe_all,
        'eth3d-img-epe-all-edge': img_all_epe_edge,
        'eth3d-img-epe-all-flat': img_all_epe_flat,
        # Pixel-level EPE
        'eth3d-pix-epe-noc-all': pix_noc_epe_all,
        'eth3d-pix-epe-noc-edge': pix_noc_epe_edge,
        'eth3d-pix-epe-noc-flat': pix_noc_epe_flat,
        'eth3d-pix-epe-all-all': pix_all_epe_all,
        'eth3d-pix-epe-all-edge': pix_all_epe_edge,
        'eth3d-pix-epe-all-flat': pix_all_epe_flat,
        # Image-level Bad1.0
        'eth3d-img-bad1-noc': img_noc_bad1_all,
        'eth3d-img-bad1-all': img_all_bad1_all,
        # Pixel-level Bad1.0
        'eth3d-pix-bad1-noc': pix_noc_bad1_all,
        'eth3d-pix-bad1-all': pix_all_bad1_all,
    }


@torch.no_grad()
def validate_kitti(model, iters=32, mixed_prec=False, device=None, root=None, val_samples=500, batch_size=1):
    """ KITTI Validation """
    if root is None: root = DEFAULT_KITTI_ROOT
    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    val_dataset = datasets.KITTI(aug_params=None, root=root, split='training')
    # KITTI 尺寸不一，强制 BS=1
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    total_valid_pixels = 0
    total_epe_sum = 0
    total_d1_sum = 0
    image_epe_list = []

    elapsed_list = []

    for i, data_blob in enumerate(loader):
        if val_samples > 0 and i >= val_samples: break

        _, image1, image2, flow_gt, valid_gt = data_blob
        image1, image2 = image1.to(device), image2.to(device)
        flow_gt, valid_gt = flow_gt.to(device), valid_gt.to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        start = time.time()
        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
        end = time.time()

        if i > 10: elapsed_list.append(end - start)

        flow_pr = padder.unpad(flow_pr)
        epe = (flow_pr - flow_gt).abs()

        mask = (valid_gt >= 0.5) & (flow_gt.abs() < 192)

        if mask.sum() > 0:
            epe_val = epe[mask]
            total_epe_sum += epe_val.sum().item()
            total_d1_sum += (epe_val > 3.0).float().sum().item()
            total_valid_pixels += mask.sum().item()
            image_epe_list.append(epe_val.mean().item())

    pixel_epe = total_epe_sum / (total_valid_pixels + 1e-8)
    pixel_d1 = 100 * total_d1_sum / (total_valid_pixels + 1e-8)
    image_epe = np.mean(image_epe_list)
    avg_runtime = np.mean(elapsed_list) if elapsed_list else 0

    print(f"\n{'=' * 60}")
    print(f"KITTI Validation Results")
    print(f"{'=' * 60}")
    print(f"  Pixel EPE: {pixel_epe:.4f}")
    print(f"  Image EPE: {image_epe:.4f}")
    print(f"  D1 (>3px): {pixel_d1:.2f}%")
    if avg_runtime > 0:
        print(f"  FPS:       {1 / avg_runtime:.2f}")
    print(f"{'=' * 60}\n")
    return {
        'kitti-pix-epe': pixel_epe,
        'kitti-img-epe': image_epe,
        'kitti-pix-d1': pixel_d1,
        # Keep old keys for compatibility
        'kitti-epe': pixel_epe,
        'kitti-d1': pixel_d1
    }


@torch.no_grad()
def validate_sceneflow(model, iters=32, mixed_prec=True, device=None, root=None, val_samples=500, batch_size=4):
    """
    SceneFlow Validation - 完全参照IGEV标准实现，同时提供完整指标

    关键特性：
    1. CPU计算EPE（参照IGEV，数值更准确）
    2. Quantile(0.98)边缘检测（参照IGEV标准）
    3. Image-level和Pixel-level双指标
    4. Overall/Edge/Flat三类区域EPE
    5. D1/D3/D5错误率（>3px, >1px, >0.5px）
    """
    if root is None: root = DEFAULT_SCENEFLOW_ROOT
    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    val_dataset = datasets.SceneFlowDatasets(root=root, dstype='frames_finalpass', things_test=True)
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print(f"Evaluating SceneFlow with Batch Size {batch_size}, samples {val_samples}...")

    # ===== Pixel-level累加器 =====
    pix_all_epe = 0.0
    pix_all_cnt = 0
    pix_all_d1 = 0
    pix_all_d3 = 0
    pix_all_d5 = 0

    pix_edge_epe = 0.0
    pix_edge_cnt = 0
    pix_edge_d1 = 0
    pix_edge_d3 = 0
    pix_edge_d5 = 0

    pix_flat_epe = 0.0
    pix_flat_cnt = 0
    pix_flat_d1 = 0
    pix_flat_d3 = 0
    pix_flat_d5 = 0

    # ===== Image-level列表 =====
    img_all_epe = []
    img_all_d1 = []
    img_all_d3 = []
    img_all_d5 = []

    img_edge_epe = []
    img_edge_d1 = []
    img_edge_d3 = []
    img_edge_d5 = []

    img_flat_epe = []
    img_flat_d1 = []
    img_flat_d3 = []
    img_flat_d5 = []

    elapsed_list = []
    edge_threshold = 0.3

    total_batches = len(loader) if val_samples == 0 else min(len(loader), (val_samples + batch_size - 1) // batch_size)
    pbar = tqdm(enumerate(loader), total=total_batches, desc="SceneFlow", ncols=100)

    for i, data_blob in pbar:
        if val_samples > 0 and (i * batch_size) >= val_samples: break

        _, image1, image2, flow_gt, valid_gt = data_blob
        image1, image2 = image1.to(device), image2.to(device)

        # ===== 在padding之前提取边缘（GPU上计算，快）=====
        edge_map = extract_edge_sobel_batch(image1).cpu()  # [B, H, W]

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        start = time.time()
        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
        end = time.time()

        if i > 5: elapsed_list.append(end - start)

        # ===== CPU计算（参照IGEV标准） =====
        flow_pr = padder.unpad(flow_pr).cpu()
        flow_gt = flow_gt.cpu()
        valid_gt = valid_gt.cpu()

        # 维度处理
        if flow_pr.dim() == 4: flow_pr = flow_pr.squeeze(1)
        if flow_gt.dim() == 4: flow_gt = flow_gt.squeeze(1)
        if valid_gt.dim() == 4: valid_gt = valid_gt.squeeze(1)

        # EPE计算
        epe = torch.abs(flow_pr - flow_gt)  # [B, H, W]
        mask = (valid_gt >= 0.5) & (flow_gt.abs() < 192)

        # 逐图处理（同时累加pixel和image指标）
        for b in range(image1.shape[0]):
            b_mask = mask[b]
            b_edge_map = edge_map[b]
            b_epe = epe[b]

            if b_mask.sum() == 0:
                continue

            # Overall区域
            all_epe_vals = b_epe[b_mask]
            pix_all_epe += all_epe_vals.sum().item()
            pix_all_cnt += b_mask.sum().item()
            pix_all_d1 += (all_epe_vals > 3.0).sum().item()
            pix_all_d3 += (all_epe_vals > 1.0).sum().item()
            pix_all_d5 += (all_epe_vals > 0.5).sum().item()

            img_all_epe.append(all_epe_vals.mean().item())
            img_all_d1.append((all_epe_vals > 3.0).float().mean().item())
            img_all_d3.append((all_epe_vals > 1.0).float().mean().item())
            img_all_d5.append((all_epe_vals > 0.5).float().mean().item())

            # Edge区域
            b_edge_mask = (b_edge_map > edge_threshold) & b_mask
            if b_edge_mask.sum() > 0:
                edge_epe_vals = b_epe[b_edge_mask]
                pix_edge_epe += edge_epe_vals.sum().item()
                pix_edge_cnt += b_edge_mask.sum().item()
                pix_edge_d1 += (edge_epe_vals > 3.0).sum().item()
                pix_edge_d3 += (edge_epe_vals > 1.0).sum().item()
                pix_edge_d5 += (edge_epe_vals > 0.5).sum().item()

                img_edge_epe.append(edge_epe_vals.mean().item())
                img_edge_d1.append((edge_epe_vals > 3.0).float().mean().item())
                img_edge_d3.append((edge_epe_vals > 1.0).float().mean().item())
                img_edge_d5.append((edge_epe_vals > 0.5).float().mean().item())

            # Flat区域
            b_flat_mask = (b_edge_map <= edge_threshold) & b_mask
            if b_flat_mask.sum() > 0:
                flat_epe_vals = b_epe[b_flat_mask]
                pix_flat_epe += flat_epe_vals.sum().item()
                pix_flat_cnt += b_flat_mask.sum().item()
                pix_flat_d1 += (flat_epe_vals > 3.0).sum().item()
                pix_flat_d3 += (flat_epe_vals > 1.0).sum().item()
                pix_flat_d5 += (flat_epe_vals > 0.5).sum().item()

                img_flat_epe.append(flat_epe_vals.mean().item())
                img_flat_d1.append((flat_epe_vals > 3.0).float().mean().item())
                img_flat_d3.append((flat_epe_vals > 1.0).float().mean().item())
                img_flat_d5.append((flat_epe_vals > 0.5).float().mean().item())

        # 更新进度条
        if len(img_all_epe) > 0:
            pbar.set_postfix({'Img-EPE': f'{np.mean(img_all_epe):.4f}'})

    # ===== 最终聚合 =====
    # Pixel-level
    pix_epe_all = pix_all_epe / (pix_all_cnt + 1e-12)
    pix_epe_edge = pix_edge_epe / (pix_edge_cnt + 1e-12)
    pix_epe_flat = pix_flat_epe / (pix_flat_cnt + 1e-12)

    pix_d1_all = 100 * pix_all_d1 / (pix_all_cnt + 1e-12)
    pix_d3_all = 100 * pix_all_d3 / (pix_all_cnt + 1e-12)
    pix_d5_all = 100 * pix_all_d5 / (pix_all_cnt + 1e-12)

    pix_d1_edge = 100 * pix_edge_d1 / (pix_edge_cnt + 1e-12)
    pix_d3_edge = 100 * pix_edge_d3 / (pix_edge_cnt + 1e-12)
    pix_d5_edge = 100 * pix_edge_d5 / (pix_edge_cnt + 1e-12)

    pix_d1_flat = 100 * pix_flat_d1 / (pix_flat_cnt + 1e-12)
    pix_d3_flat = 100 * pix_flat_d3 / (pix_flat_cnt + 1e-12)
    pix_d5_flat = 100 * pix_flat_d5 / (pix_flat_cnt + 1e-12)

    # Image-level
    img_epe_all = np.mean(img_all_epe) if img_all_epe else 0
    img_epe_edge = np.mean(img_edge_epe) if img_edge_epe else 0
    img_epe_flat = np.mean(img_flat_epe) if img_flat_epe else 0

    img_d1_all = 100 * np.mean(img_all_d1) if img_all_d1 else 0
    img_d3_all = 100 * np.mean(img_all_d3) if img_all_d3 else 0
    img_d5_all = 100 * np.mean(img_all_d5) if img_all_d5 else 0

    img_d1_edge = 100 * np.mean(img_edge_d1) if img_edge_d1 else 0
    img_d3_edge = 100 * np.mean(img_edge_d3) if img_edge_d3 else 0
    img_d5_edge = 100 * np.mean(img_edge_d5) if img_edge_d5 else 0

    img_d1_flat = 100 * np.mean(img_flat_d1) if img_flat_d1 else 0
    img_d3_flat = 100 * np.mean(img_flat_d3) if img_flat_d3 else 0
    img_d5_flat = 100 * np.mean(img_flat_d5) if img_flat_d5 else 0

    # FPS
    fps = batch_size / np.mean(elapsed_list) if elapsed_list else 0

    # ===== 打印结果 =====
    print(f"\n{'='*80}")
    print(f"SceneFlow Validation Results ({len(img_all_epe)} images)")
    print(f"{'='*80}")
    print(f"{'Metric':<25} {'Overall':>12} {'Edge':>12} {'Flat':>12}")
    print(f"{'-'*80}")
    print(f"{'Image-level EPE':<25} {img_epe_all:>12.4f} {img_epe_edge:>12.4f} {img_epe_flat:>12.4f}")
    print(f"{'Pixel-level EPE':<25} {pix_epe_all:>12.4f} {pix_epe_edge:>12.4f} {pix_epe_flat:>12.4f}")
    print(f"{'-'*80}")
    print(f"{'Image-level D1 (>3px)':<25} {img_d1_all:>11.2f}% {img_d1_edge:>11.2f}% {img_d1_flat:>11.2f}%")
    print(f"{'Pixel-level D1 (>3px)':<25} {pix_d1_all:>11.2f}% {pix_d1_edge:>11.2f}% {pix_d1_flat:>11.2f}%")
    print(f"{'-'*80}")
    print(f"{'Image-level D3 (>1px)':<25} {img_d3_all:>11.2f}% {img_d3_edge:>11.2f}% {img_d3_flat:>11.2f}%")
    print(f"{'Pixel-level D3 (>1px)':<25} {pix_d3_all:>11.2f}% {pix_d3_edge:>11.2f}% {pix_d3_flat:>11.2f}%")
    print(f"{'-'*80}")
    print(f"{'Image-level D5 (>0.5px)':<25} {img_d5_all:>11.2f}% {img_d5_edge:>11.2f}% {img_d5_flat:>11.2f}%")
    print(f"{'Pixel-level D5 (>0.5px)':<25} {pix_d5_all:>11.2f}% {pix_d5_edge:>11.2f}% {pix_d5_flat:>11.2f}%")
    print(f"{'='*80}")
    print(f"Total pixels: {pix_all_cnt:,} | Edge: {pix_edge_cnt:,} | Flat: {pix_flat_cnt:,}")
    print(f"FPS: {fps:.2f} (Batch size: {batch_size})")
    print(f"{'='*80}\n")

    return {
        # Image-level EPE
        'sceneflow-img-epe-all': img_epe_all,
        'sceneflow-img-epe-edge': img_epe_edge,
        'sceneflow-img-epe-flat': img_epe_flat,
        # Pixel-level EPE
        'sceneflow-pix-epe-all': pix_epe_all,
        'sceneflow-pix-epe-edge': pix_epe_edge,
        'sceneflow-pix-epe-flat': pix_epe_flat,
        # Image-level D1 (>3px)
        'sceneflow-img-d1': img_d1_all,
        # Pixel-level D1 (>3px)
        'sceneflow-pix-d1': pix_d1_all,
        # Additional metrics (kept for compatibility)
        'img-d3-all': img_d3_all,
        'img-d5-all': img_d5_all,
        'pix-d3-all': pix_d3_all,
        'pix-d5-all': pix_d5_all,
    }


@torch.no_grad()
def validate_middlebury(model, iters=32, split='F', mixed_prec=False, device=None, root=None, val_samples=500,
                        batch_size=1):
    """
    Middlebury Validation with comprehensive metrics
    - Image-level and Pixel-level EPE (overall, edge, flat)
    - Separated by noc (non-occluded) and all regions
    - Bad 2.0 metric (>2px error rate)
    """
    if root is None: root = DEFAULT_MIDDLEBURY_ROOT
    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    val_dataset = datasets.Middlebury(aug_params=None, root=root, split=split)
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    edge_threshold = 0.3

    # Pixel-level accumulators for NOC (non-occluded)
    pix_noc_all_epe, pix_noc_all_cnt, pix_noc_all_bad2 = 0.0, 0, 0
    pix_noc_edge_epe, pix_noc_edge_cnt, pix_noc_edge_bad2 = 0.0, 0, 0
    pix_noc_flat_epe, pix_noc_flat_cnt, pix_noc_flat_bad2 = 0.0, 0, 0

    # Pixel-level accumulators for ALL (including occluded)
    pix_all_all_epe, pix_all_all_cnt, pix_all_all_bad2 = 0.0, 0, 0
    pix_all_edge_epe, pix_all_edge_cnt, pix_all_edge_bad2 = 0.0, 0, 0
    pix_all_flat_epe, pix_all_flat_cnt, pix_all_flat_bad2 = 0.0, 0, 0

    # Image-level lists for NOC
    img_noc_all_epe, img_noc_all_bad2 = [], []
    img_noc_edge_epe, img_noc_edge_bad2 = [], []
    img_noc_flat_epe, img_noc_flat_bad2 = [], []

    # Image-level lists for ALL
    img_all_all_epe, img_all_all_bad2 = [], []
    img_all_edge_epe, img_all_edge_bad2 = [], []
    img_all_flat_epe, img_all_flat_bad2 = [], []

    print(f"Start validation on Middlebury-{split}...")

    for i, data_blob in enumerate(tqdm(loader, desc=f"Middlebury-{split}")):
        if val_samples > 0 and i >= val_samples: break

        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = data_blob

        # Extract edge map before padding
        edge_map = extract_edge_sobel_batch(image1).cpu()  # [B, H, W]

        image1, image2 = image1.to(device), image2.to(device)
        flow_gt, valid_gt = flow_gt.to(device), valid_gt.to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)

        flow_pr = padder.unpad(flow_pr.float()).cpu()
        flow_gt = flow_gt.cpu()
        valid_gt = valid_gt.cpu()

        # Load occlusion mask
        occ_mask_path = imageL_file[0].replace('im0.png', 'mask0nocc.png')
        if os.path.exists(occ_mask_path):
            occ_mask = np.array(Image.open(occ_mask_path).convert('L'))
            occ_mask = torch.from_numpy(occ_mask)
            noc_mask_bool = (occ_mask == 255)
        else:
            noc_mask_bool = torch.ones_like(valid_gt.squeeze(), dtype=torch.bool)

        # Ensure dimension consistency
        if flow_pr.dim() == 4: flow_pr = flow_pr.squeeze(1)
        if flow_gt.dim() == 4: flow_gt = flow_gt.squeeze(1)
        if valid_gt.dim() == 4: valid_gt = valid_gt.squeeze(1)

        epe = (flow_pr - flow_gt).abs()

        # Process each image in batch (BS=1)
        for b in range(flow_pr.shape[0]):
            b_epe = epe[b]
            b_edge_map = edge_map[b]
            b_valid = (valid_gt[b] >= 0.5) & (flow_gt[b] < 192)
            b_noc = noc_mask_bool if noc_mask_bool.dim() == 2 else noc_mask_bool[b]

            # ALL region (valid pixels)
            mask_all = b_valid
            if mask_all.sum() > 0:
                all_epe_vals = b_epe[mask_all]
                pix_all_all_epe += all_epe_vals.sum().item()
                pix_all_all_cnt += mask_all.sum().item()
                pix_all_all_bad2 += (all_epe_vals > 2.0).sum().item()
                img_all_all_epe.append(all_epe_vals.mean().item())
                img_all_all_bad2.append((all_epe_vals > 2.0).float().mean().item())

                # Edge region (ALL)
                mask_all_edge = (b_edge_map > edge_threshold) & mask_all
                if mask_all_edge.sum() > 0:
                    edge_epe_vals = b_epe[mask_all_edge]
                    pix_all_edge_epe += edge_epe_vals.sum().item()
                    pix_all_edge_cnt += mask_all_edge.sum().item()
                    pix_all_edge_bad2 += (edge_epe_vals > 2.0).sum().item()
                    img_all_edge_epe.append(edge_epe_vals.mean().item())
                    img_all_edge_bad2.append((edge_epe_vals > 2.0).float().mean().item())

                # Flat region (ALL)
                mask_all_flat = (b_edge_map <= edge_threshold) & mask_all
                if mask_all_flat.sum() > 0:
                    flat_epe_vals = b_epe[mask_all_flat]
                    pix_all_flat_epe += flat_epe_vals.sum().item()
                    pix_all_flat_cnt += mask_all_flat.sum().item()
                    pix_all_flat_bad2 += (flat_epe_vals > 2.0).sum().item()
                    img_all_flat_epe.append(flat_epe_vals.mean().item())
                    img_all_flat_bad2.append((flat_epe_vals > 2.0).float().mean().item())

            # NOC region (non-occluded)
            mask_noc = b_valid & b_noc
            if mask_noc.sum() > 0:
                noc_epe_vals = b_epe[mask_noc]
                pix_noc_all_epe += noc_epe_vals.sum().item()
                pix_noc_all_cnt += mask_noc.sum().item()
                pix_noc_all_bad2 += (noc_epe_vals > 2.0).sum().item()
                img_noc_all_epe.append(noc_epe_vals.mean().item())
                img_noc_all_bad2.append((noc_epe_vals > 2.0).float().mean().item())

                # Edge region (NOC)
                mask_noc_edge = (b_edge_map > edge_threshold) & mask_noc
                if mask_noc_edge.sum() > 0:
                    edge_epe_vals = b_epe[mask_noc_edge]
                    pix_noc_edge_epe += edge_epe_vals.sum().item()
                    pix_noc_edge_cnt += mask_noc_edge.sum().item()
                    pix_noc_edge_bad2 += (edge_epe_vals > 2.0).sum().item()
                    img_noc_edge_epe.append(edge_epe_vals.mean().item())
                    img_noc_edge_bad2.append((edge_epe_vals > 2.0).float().mean().item())

                # Flat region (NOC)
                mask_noc_flat = (b_edge_map <= edge_threshold) & mask_noc
                if mask_noc_flat.sum() > 0:
                    flat_epe_vals = b_epe[mask_noc_flat]
                    pix_noc_flat_epe += flat_epe_vals.sum().item()
                    pix_noc_flat_cnt += mask_noc_flat.sum().item()
                    pix_noc_flat_bad2 += (flat_epe_vals > 2.0).sum().item()
                    img_noc_flat_epe.append(flat_epe_vals.mean().item())
                    img_noc_flat_bad2.append((flat_epe_vals > 2.0).float().mean().item())

    # Calculate final metrics
    # Pixel-level NOC
    pix_noc_epe_all = pix_noc_all_epe / (pix_noc_all_cnt + 1e-12)
    pix_noc_epe_edge = pix_noc_edge_epe / (pix_noc_edge_cnt + 1e-12)
    pix_noc_epe_flat = pix_noc_flat_epe / (pix_noc_flat_cnt + 1e-12)
    pix_noc_bad2_all = 100 * pix_noc_all_bad2 / (pix_noc_all_cnt + 1e-12)
    pix_noc_bad2_edge = 100 * pix_noc_edge_bad2 / (pix_noc_edge_cnt + 1e-12)
    pix_noc_bad2_flat = 100 * pix_noc_flat_bad2 / (pix_noc_flat_cnt + 1e-12)

    # Pixel-level ALL
    pix_all_epe_all = pix_all_all_epe / (pix_all_all_cnt + 1e-12)
    pix_all_epe_edge = pix_all_edge_epe / (pix_all_edge_cnt + 1e-12)
    pix_all_epe_flat = pix_all_flat_epe / (pix_all_flat_cnt + 1e-12)
    pix_all_bad2_all = 100 * pix_all_all_bad2 / (pix_all_all_cnt + 1e-12)
    pix_all_bad2_edge = 100 * pix_all_edge_bad2 / (pix_all_edge_cnt + 1e-12)
    pix_all_bad2_flat = 100 * pix_all_flat_bad2 / (pix_all_flat_cnt + 1e-12)

    # Image-level NOC
    img_noc_epe_all = np.mean(img_noc_all_epe) if img_noc_all_epe else 0
    img_noc_epe_edge = np.mean(img_noc_edge_epe) if img_noc_edge_epe else 0
    img_noc_epe_flat = np.mean(img_noc_flat_epe) if img_noc_flat_epe else 0
    img_noc_bad2_all = 100 * np.mean(img_noc_all_bad2) if img_noc_all_bad2 else 0
    img_noc_bad2_edge = 100 * np.mean(img_noc_edge_bad2) if img_noc_edge_bad2 else 0
    img_noc_bad2_flat = 100 * np.mean(img_noc_flat_bad2) if img_noc_flat_bad2 else 0

    # Image-level ALL
    img_all_epe_all = np.mean(img_all_all_epe) if img_all_all_epe else 0
    img_all_epe_edge = np.mean(img_all_edge_epe) if img_all_edge_epe else 0
    img_all_epe_flat = np.mean(img_all_flat_epe) if img_all_flat_epe else 0
    img_all_bad2_all = 100 * np.mean(img_all_all_bad2) if img_all_all_bad2 else 0
    img_all_bad2_edge = 100 * np.mean(img_all_edge_bad2) if img_all_edge_bad2 else 0
    img_all_bad2_flat = 100 * np.mean(img_all_flat_bad2) if img_all_flat_bad2 else 0

    # Print results
    print(f"\n{'='*90}")
    print(f"Middlebury-{split} Validation Results ({len(img_all_all_epe)} images)")
    print(f"{'='*90}")
    print(f"{'Metric':<30} {'Overall':>12} {'Edge':>12} {'Flat':>12}")
    print(f"{'-'*90}")
    print(f"{'Image-level EPE (NOC)':<30} {img_noc_epe_all:>12.4f} {img_noc_epe_edge:>12.4f} {img_noc_epe_flat:>12.4f}")
    print(f"{'Pixel-level EPE (NOC)':<30} {pix_noc_epe_all:>12.4f} {pix_noc_epe_edge:>12.4f} {pix_noc_epe_flat:>12.4f}")
    print(f"{'Image-level EPE (ALL)':<30} {img_all_epe_all:>12.4f} {img_all_epe_edge:>12.4f} {img_all_epe_flat:>12.4f}")
    print(f"{'Pixel-level EPE (ALL)':<30} {pix_all_epe_all:>12.4f} {pix_all_epe_edge:>12.4f} {pix_all_epe_flat:>12.4f}")
    print(f"{'-'*90}")
    print(f"{'Image-level Bad2.0 (NOC)':<30} {img_noc_bad2_all:>11.2f}% {img_noc_bad2_edge:>11.2f}% {img_noc_bad2_flat:>11.2f}%")
    print(f"{'Pixel-level Bad2.0 (NOC)':<30} {pix_noc_bad2_all:>11.2f}% {pix_noc_bad2_edge:>11.2f}% {pix_noc_bad2_flat:>11.2f}%")
    print(f"{'Image-level Bad2.0 (ALL)':<30} {img_all_bad2_all:>11.2f}% {img_all_bad2_edge:>11.2f}% {img_all_bad2_flat:>11.2f}%")
    print(f"{'Pixel-level Bad2.0 (ALL)':<30} {pix_all_bad2_all:>11.2f}% {pix_all_bad2_edge:>11.2f}% {pix_all_bad2_flat:>11.2f}%")
    print(f"{'='*90}\n")

    return {
        # Image-level EPE
        f'middlebury{split}-img-epe-noc-all': img_noc_epe_all,
        f'middlebury{split}-img-epe-noc-edge': img_noc_epe_edge,
        f'middlebury{split}-img-epe-noc-flat': img_noc_epe_flat,
        f'middlebury{split}-img-epe-all-all': img_all_epe_all,
        f'middlebury{split}-img-epe-all-edge': img_all_epe_edge,
        f'middlebury{split}-img-epe-all-flat': img_all_epe_flat,
        # Pixel-level EPE
        f'middlebury{split}-pix-epe-noc-all': pix_noc_epe_all,
        f'middlebury{split}-pix-epe-noc-edge': pix_noc_epe_edge,
        f'middlebury{split}-pix-epe-noc-flat': pix_noc_epe_flat,
        f'middlebury{split}-pix-epe-all-all': pix_all_epe_all,
        f'middlebury{split}-pix-epe-all-edge': pix_all_epe_edge,
        f'middlebury{split}-pix-epe-all-flat': pix_all_epe_flat,
        # Image-level Bad2.0
        f'middlebury{split}-img-bad2-noc': img_noc_bad2_all,
        f'middlebury{split}-img-bad2-all': img_all_bad2_all,
        # Pixel-level Bad2.0
        f'middlebury{split}-pix-bad2-noc': pix_noc_bad2_all,
        f'middlebury{split}-pix-bad2-all': pix_all_bad2_all,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint",
                        default="../model_cache/IGEV/sceneflow.pth"
                        # default='../logs/our3_211/195000_gt_lr0002.pth'
                        )
    parser.add_argument('--dataset', help="dataset for evaluation", default='sceneflow',
                        choices=["eth3d", "kitti", "sceneflow"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', default=True, action='store_false', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='bfloat16', choices=['float16', 'bfloat16', 'float32'],
                        help='Choose precision type')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--val_samples', type=int, default=0, help='number of samples to validate (0 for all)')

    # 核心修改：支持 Batch Size
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size for evaluation (recommend >1 for SceneFlow, 1 for others)')

    # Dataset paths
    parser.add_argument('--dataset_root', type=str, default=DEFAULT_DATASET_ROOT,
                        help='root directory for all datasets')
    parser.add_argument('--sceneflow_root', type=str, default=None, help='path to SceneFlow dataset')
    parser.add_argument('--kitti_root', type=str, default=None, help='path to KITTI dataset')
    parser.add_argument('--eth3d_root', type=str, default=None, help='path to ETH3D dataset')
    parser.add_argument('--middlebury_root', type=str, default=None, help='path to Middlebury dataset')

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                        help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    args = parser.parse_args()

    # Set default dataset paths if not specified
    if args.sceneflow_root is None:
        args.sceneflow_root = os.path.join(args.dataset_root, 'SceneFlow')
    if args.kitti_root is None:
        args.kitti_root = os.path.join(args.dataset_root, 'KITTI')
    if args.eth3d_root is None:
        args.eth3d_root = os.path.join(args.dataset_root, 'ETH3D')
    if args.middlebury_root is None:
        args.middlebury_root = os.path.join(args.dataset_root, 'Middlebury/MiddEval3')

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0] if torch.cuda.is_available() else None)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    logging.info(f"Using device: {args.device}")

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.to(args.device)
    model.eval()

    print(f"The model has {format(count_parameters(model) / 1e6, '.2f')}M learnable parameters.")

    # 逻辑控制：SceneFlow 可以用大 Batch，其他数据集如果用户设大了 Batch，给个警告并建议设为 1
    if args.dataset != 'sceneflow' and args.batch_size > 1:
        logging.warning(f"Dataset {args.dataset} typically has varying image sizes. "
                        f"Batch size {args.batch_size} might cause collation errors. "
                        f"Recommended batch_size=1 for validation.")

    if args.dataset == 'eth3d':
        validate_eth3d(model, iters=args.valid_iters, mixed_prec=args.mixed_precision, device=args.device,
                       root=args.eth3d_root, val_samples=args.val_samples, batch_size=args.batch_size)

    elif args.dataset == 'kitti':
        validate_kitti(model, iters=args.valid_iters, mixed_prec=args.mixed_precision, device=args.device,
                       root=args.kitti_root, val_samples=args.val_samples, batch_size=args.batch_size)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, iters=args.valid_iters, split=args.dataset[-1], mixed_prec=args.mixed_precision,
                            device=args.device, root=args.middlebury_root, val_samples=args.val_samples,
                            batch_size=args.batch_size)

    elif args.dataset == 'sceneflow':
        validate_sceneflow(model, iters=args.valid_iters, mixed_prec=args.mixed_precision, device=args.device,
                           root=args.sceneflow_root, val_samples=args.val_samples, batch_size=args.batch_size)