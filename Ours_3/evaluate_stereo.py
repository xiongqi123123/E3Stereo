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
    Batch-wise Edge Extraction (Sobel)
    Input: [B, 3, H, W]
    Output: [B, H, W] normalized edge map
    """
    # 转灰度
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    # 归一化检查 (Assuming input is [0, 255] based on typical loader, but check max)
    if gray.max() > 1.0:
        gray = gray / 255.0

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)

    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    # Batch-wise normalization (using quantile per image is expensive, roughly map using max)
    # 为了速度，这里使用 batch 内每张图的 max 进行归一化
    B, C, H, W = edge.shape
    edge_flat = edge.view(B, -1)
    max_val = edge_flat.max(dim=1, keepdim=True)[0]
    edge = edge / (max_val.view(B, 1, 1, 1) + 1e-8)

    # 稍微增强对比度，模拟 quantile 0.98 的效果
    edge = torch.clamp(edge * 1.2, 0, 1)

    return edge.squeeze(1)


@torch.no_grad()
def validate_eth3d(model, iters=32, mixed_prec=False, device=None, root=None, val_samples=500, batch_size=1):
    """ ETH3D Validation """
    if root is None: root = DEFAULT_ETH3D_ROOT
    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    val_dataset = datasets.ETH3D({}, root=root)

    # ETH3D 图片尺寸不一，强制 batch_size=1
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    total_valid_pixels = 0
    total_epe_sum = 0
    total_d1_sum = 0

    image_epe_list = []

    print(f"Start validation on ETH3D...")

    for i, data_blob in enumerate(loader):
        if val_samples > 0 and i >= val_samples: break

        (imageL_file, imageR_file, GT_file), image1, image2, flow_gt, valid_gt = data_blob
        image1, image2 = image1.to(device), image2.to(device)
        flow_gt, valid_gt = flow_gt.to(device), valid_gt.to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)

        flow_pr = padder.unpad(flow_pr.float())

        # ETH3D mask logic
        # 这里需要读额外的 mask 文件，比较耗时，且 DataLoader 出来的 paths 是 tuple
        # 为了 Batch 处理，这部分稍微复杂。由于强制 BS=1，我们取第一个元素
        mask_path = GT_file[0].replace('disp0GT.pfm', 'mask0nocc.png')
        if os.path.exists(mask_path):
            occ_mask = np.array(Image.open(mask_path))
            occ_mask = torch.from_numpy(occ_mask).to(device)
            # Resize logic if needed, usually match
            occ_mask_bool = (occ_mask == 255)
        else:
            occ_mask_bool = torch.ones_like(valid_gt, dtype=torch.bool)

        epe = (flow_pr - flow_gt).abs()

        # Valid mask
        mask = (valid_gt >= 0.5) & occ_mask_bool

        if mask.sum() > 0:
            # Metrics
            epe_val = epe[mask]

            # Pixel-wise accumulation
            total_epe_sum += epe_val.sum().item()
            total_d1_sum += (epe_val > 1.0).float().sum().item()
            total_valid_pixels += mask.sum().item()

            # Image-wise accumulation
            image_epe_list.append(epe_val.mean().item())

    pixel_epe = total_epe_sum / (total_valid_pixels + 1e-8)
    pixel_d1 = 100 * total_d1_sum / (total_valid_pixels + 1e-8)
    image_epe = np.mean(image_epe_list)

    print(f"\n{'=' * 60}")
    print(f"ETH3D Validation Results")
    print(f"{'=' * 60}")
    print(f"  Pixel EPE: {pixel_epe:.4f}")
    print(f"  Image EPE: {image_epe:.4f}")
    print(f"  D1 (>1px): {pixel_d1:.2f}%")
    print(f"{'=' * 60}\n")
    return {'eth3d-epe': pixel_epe, 'eth3d-d1': pixel_d1}


@torch.no_grad()
def validate_kitti(model, iters=32, mixed_prec=False, device=None, root=None, val_samples=500, batch_size=1):
    """ KITTI Validation """
    if root is None: root = DEFAULT_KITTI_ROOT
    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    val_dataset = datasets.KITTI({}, root=root, image_set='training')
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
    return {'kitti-epe': pixel_epe, 'kitti-d1': pixel_d1}


@torch.no_grad()
def validate_sceneflow(model, iters=32, mixed_prec=True, device=None, root=None, val_samples=500, batch_size=4):
    """ SceneFlow Validation - Supports Batch Processing """
    if root is None: root = DEFAULT_SCENEFLOW_ROOT
    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    # SceneFlow is fixed size, supports batching
    val_dataset = datasets.SceneFlowDatasets(root=root, dstype='frames_finalpass', things_test=True)
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print(f"Evaluating SceneFlow with Batch Size {batch_size}, samples {val_samples}...")

    # Global counters (Pixel-wise)
    total_valid_pixels = 0
    total_epe_sum = 0
    total_d1_sum = 0
    total_d3_sum = 0
    total_d5_sum = 0

    # Edge analysis counters
    total_edge_pixels = 0
    total_flat_pixels = 0
    total_edge_epe = 0
    total_flat_epe = 0

    # Image-wise list
    image_epe_list = []
    image_epe_edge_list = []
    image_epe_flat_list = []

    elapsed_list = []

    edge_threshold = 0.3

    for i, data_blob in enumerate(loader):
        # 注意: DataLoader 会自动计算 batch 数量，所以这里的 limit check 稍微粗略
        if val_samples > 0 and (i * batch_size) >= val_samples: break

        _, image1, image2, flow_gt, valid_gt = data_blob
        image1, image2 = image1.to(device), image2.to(device)
        flow_gt, valid_gt = flow_gt.to(device), valid_gt.to(device)

        # 1. Edge Extraction (Batch-wise on GPU)
        edge_map = extract_edge_sobel_batch(image1)  # [B, H, W]

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        start = time.time()
        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
        end = time.time()

        if i > 5: elapsed_list.append(end - start)

        flow_pr = padder.unpad(flow_pr)
        assert flow_pr.shape == flow_gt.shape

        # 2. Metrics Calculation (Vectorized)
        epe = (flow_pr - flow_gt).abs()  # [B, 1, H, W] or [B, H, W]
        if epe.dim() == 4: epe = epe.squeeze(1)
        if flow_gt.dim() == 4: flow_gt = flow_gt.squeeze(1)
        if valid_gt.dim() == 4: valid_gt = valid_gt.squeeze(1)

        mask = (valid_gt >= 0.5) & (flow_gt.abs() < 192)

        # 3. Accumulate Pixel Metrics
        if mask.sum() > 0:
            masked_epe = epe[mask]

            # --- Pixel Mean Accumulation ---
            total_epe_sum += masked_epe.sum().item()
            total_d1_sum += (masked_epe > 3.0).float().sum().item()
            total_d3_sum += (masked_epe > 1.0).float().sum().item()
            total_d5_sum += (masked_epe > 0.5).float().sum().item()
            total_valid_pixels += mask.sum().item()

            # --- Image Mean Accumulation ---
            # Need to loop over batch to get per-image mean (vectorized reduction is tricky with variable valid counts per img)
            for b in range(image1.shape[0]):
                b_mask = mask[b]
                if b_mask.sum() > 0:
                    image_epe_list.append(epe[b][b_mask].mean().item())

            # --- Edge vs Flat Analysis ---
            is_edge = (edge_map > edge_threshold) & mask
            is_flat = (edge_map <= edge_threshold) & mask

            if is_edge.sum() > 0:
                total_edge_epe += epe[is_edge].sum().item()
                total_edge_pixels += is_edge.sum().item()

            if is_flat.sum() > 0:
                total_flat_epe += epe[is_flat].sum().item()
                total_flat_pixels += is_flat.sum().item()

            # --- Image-level Edge/Flat EPE ---
            for b in range(image1.shape[0]):
                b_edge = is_edge[b]
                if b_edge.sum() > 0:
                    image_epe_edge_list.append(epe[b][b_edge].mean().item())

                b_flat = is_flat[b]
                if b_flat.sum() > 0:
                    image_epe_flat_list.append(epe[b][b_flat].mean().item())

    # --- Final Aggregation ---
    pixel_epe = total_epe_sum / (total_valid_pixels + 1e-8)
    pixel_d1 = 100 * total_d1_sum / (total_valid_pixels + 1e-8)
    pixel_d3 = 100 * total_d3_sum / (total_valid_pixels + 1e-8)
    pixel_d5 = 100 * total_d5_sum / (total_valid_pixels + 1e-8)

    image_mean_epe = np.mean(image_epe_list) if len(image_epe_list) > 0 else 0
    image_mean_epe_edge = np.mean(image_epe_edge_list) if len(image_epe_edge_list) > 0 else 0
    image_mean_epe_flat = np.mean(image_epe_flat_list) if len(image_epe_flat_list) > 0 else 0

    avg_epe_edge = total_edge_epe / (total_edge_pixels + 1e-8)
    avg_epe_flat = total_flat_epe / (total_flat_pixels + 1e-8)

    avg_time = np.mean(elapsed_list) if elapsed_list else 0
    # FPS = Batch_Size / Avg_Time_Per_Batch
    fps = batch_size / avg_time if avg_time > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"SceneFlow Validation Results ({len(image_epe_list)} images evaluated)")
    print(f"{'=' * 60}")
    print(f"  Pixel-level EPE:")
    print(f"    All:   {pixel_epe:.4f} | Edge: {avg_epe_edge:.4f} | Flat: {avg_epe_flat:.4f}")
    print(f"  Image-level EPE:")
    print(f"    All:   {image_mean_epe:.4f} | Edge: {image_mean_epe_edge:.4f} | Flat: {image_mean_epe_flat:.4f}")
    print(f"  -------------------------------------------")
    print(f"  D-Metrics (Error Rate):")
    print(f"    D1 (>3px): {pixel_d1:.2f}%")
    print(f"    D3 (>1px): {pixel_d3:.2f}%")
    print(f"    D5 (>0.5px): {pixel_d5:.2f}%")
    print(f"  -------------------------------------------")
    print(f"  Inference Speed:  {fps:.2f} FPS (Batch Size {batch_size})")
    print(f"{'=' * 60}\n")

    return {
        'scene-disp-epe': pixel_epe,
        'scene-disp-image-epe': image_mean_epe,
        'scene-disp-epe-edge': avg_epe_edge,
        'scene-disp-epe-flat': avg_epe_flat,
        'scene-disp-image-epe-edge': image_mean_epe_edge,
        'scene-disp-image-epe-flat': image_mean_epe_flat,
        'scene-disp-d1': pixel_d1,
        'scene-disp-d3': pixel_d3,
        'scene-disp-d5': pixel_d5
    }


@torch.no_grad()
def validate_middlebury(model, iters=32, split='F', mixed_prec=False, device=None, root=None, val_samples=500,
                        batch_size=1):
    """ Middlebury Validation """
    if root is None: root = DEFAULT_MIDDLEBURY_ROOT
    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    val_dataset = datasets.Middlebury({}, root=root, split=split)
    # Middlebury varies wildly in size (some are 3000px wide), force BS=1
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    total_valid_pixels = 0
    total_epe_sum = 0
    total_d1_sum = 0
    image_epe_list = []

    for i, data_blob in enumerate(loader):
        if val_samples > 0 and i >= val_samples: break

        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = data_blob
        image1, image2 = image1.to(device), image2.to(device)
        flow_gt, valid_gt = flow_gt.to(device), valid_gt.to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        # Mask loading (Slow part, but hard to batch)
        occ_mask_path = imageL_file[0].replace('im0.png', 'mask0nocc.png')
        if os.path.exists(occ_mask_path):
            occ_mask = np.array(Image.open(occ_mask_path).convert('L'))
            occ_mask = torch.from_numpy(occ_mask).to(device)
            occ_mask_bool = (occ_mask == 255)
        else:
            occ_mask_bool = torch.ones_like(valid_gt.squeeze(), dtype=torch.bool)

        flow_gt = flow_gt.squeeze()
        valid_gt = valid_gt.squeeze()

        epe = (flow_pr - flow_gt).abs()

        mask = (valid_gt >= 0.5) & (flow_gt < 192) & occ_mask_bool

        if mask.sum() > 0:
            epe_val = epe[mask]
            total_epe_sum += epe_val.sum().item()
            total_d1_sum += (epe_val > 2.0).float().sum().item()
            total_valid_pixels += mask.sum().item()
            image_epe_list.append(epe_val.mean().item())

    pixel_epe = total_epe_sum / (total_valid_pixels + 1e-8)
    pixel_d1 = 100 * total_d1_sum / (total_valid_pixels + 1e-8)
    image_epe = np.mean(image_epe_list)

    print(f"\n{'=' * 60}")
    print(f"Middlebury-{split} Validation Results")
    print(f"{'=' * 60}")
    print(f"  Pixel EPE: {pixel_epe:.4f}")
    print(f"  Image EPE: {image_epe:.4f}")
    print(f"  D1 (>2px): {pixel_d1:.2f}%")
    print(f"{'=' * 60}\n")
    return {f'middlebury{split}-epe': pixel_epe, f'middlebury{split}-d1': pixel_d1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--dataset', help="dataset for evaluation", default='sceneflow',
                        choices=["eth3d", "kitti", "sceneflow"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'],
                        help='Choose precision type')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--val_samples', type=int, default=0, help='number of samples to validate (0 for all)')

    # 核心修改：支持 Batch Size
    parser.add_argument('--batch_size', type=int, default=1,
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