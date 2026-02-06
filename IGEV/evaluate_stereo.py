from __future__ import print_function, division
import sys

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import time
import logging
import numpy as np
import torch
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

@torch.no_grad()
def validate_eth3d(model, iters=32, mixed_prec=False, device=None, root=None, val_samples=500):
    """ Peform validation using the ETH3D (train) split """
    if root is None:
        root = DEFAULT_ETH3D_ROOT
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params, root=root)

    total_samples = min(val_samples, len(val_dataset)) if val_samples > 0 else len(val_dataset)

    out_list, epe_list = [], []
    for val_id in range(total_samples):
        (imageL_file, imageR_file, GT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()

        occ_mask = Image.open(GT_file.replace('disp0GT.pfm', 'mask0nocc.png'))
        occ_mask = np.ascontiguousarray(occ_mask).flatten()

        val = (valid_gt.flatten() >= 0.5) & (occ_mask == 255)
        out = (epe_flattened > 1.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print(f"\n{'='*60}")
    print(f"ETH3D Validation Results ({total_samples} samples)")
    print(f"{'='*60}")
    print(f"  EPE: {epe:.4f}")
    print(f"  D1 (>1px): {d1:.2f}%")
    print(f"{'='*60}\n")
    return {'eth3d-epe': epe, 'eth3d-d1': d1}


@torch.no_grad()
def validate_kitti(model, iters=32, mixed_prec=False, device=None, root=None, val_samples=500):
    """ Peform validation using the KITTI-2015 (train) split """
    if root is None:
        root = DEFAULT_KITTI_ROOT
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, root=root, image_set='training')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    total_samples = min(val_samples, len(val_dataset)) if val_samples > 0 else len(val_dataset)

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(total_samples):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end-start)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list) if elapsed_list else 0.0

    print(f"\n{'='*60}")
    print(f"KITTI Validation Results ({total_samples} samples)")
    print(f"{'='*60}")
    print(f"  EPE: {epe:.4f}")
    print(f"  D1 (>3px): {d1:.2f}%")
    if avg_runtime > 0:
        print(f"  FPS: {1/avg_runtime:.2f} ({avg_runtime*1000:.1f}ms/frame)")
    print(f"{'='*60}\n")
    return {'kitti-epe': epe, 'kitti-d1': d1}


def extract_edge_sobel(image):
    """从图像提取边缘 (Sobel), 输入[B,3,H,W]或[3,H,W], 输出[H,W] numpy"""
    if image.dim() == 3:
        image = image.unsqueeze(0)
    # 转灰度
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    if gray.max() > 1.0:
        gray = gray / 255.0
    # Sobel
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    grad_x = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
    grad_y = torch.nn.functional.conv2d(gray, sobel_y, padding=1)
    edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
    # 归一化
    edge = edge.squeeze().cpu()
    edge = edge / (edge.quantile(0.98) + 1e-8)
    edge = torch.clamp(edge, 0, 1)
    return edge


@torch.no_grad()
def validate_sceneflow(model, iters=32, mixed_prec=False, device=None, root=None, val_samples=500):
    """ Peform validation using the Scene Flow (TEST) split """
    if root is None:
        root = DEFAULT_SCENEFLOW_ROOT
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(root=root, dstype='frames_finalpass', things_test=True)

    total_samples = min(val_samples, len(val_dataset)) if val_samples > 0 else len(val_dataset)

    out_list, epe_list, epe_edge_list, epe_flat_list = [], [], [], []
    edge_threshold = 0.3

    for val_id in range(total_samples):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]

        # 提取边缘 (在padding前，用原始image1)
        edge_map = extract_edge_sobel(image1)

        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

        epe = torch.abs(flow_pr - flow_gt).squeeze()
        valid_mask = (valid_gt.squeeze() >= 0.5) & (flow_gt.abs().squeeze() < 192)

        if epe[valid_mask].numel() == 0 or np.isnan(epe[valid_mask].mean().item()):
            continue

        # 边缘/平坦区域mask
        is_edge = (edge_map > edge_threshold) & valid_mask
        is_flat = (edge_map <= edge_threshold) & valid_mask

        out = (epe > 3.0)
        epe_list.append(epe[valid_mask].mean().item())
        out_list.append(out[valid_mask].cpu().numpy())

        if is_edge.sum() > 0:
            epe_edge_list.append(epe[is_edge].mean().item())
        if is_flat.sum() > 0:
            epe_flat_list.append(epe[is_flat].mean().item())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    epe_edge = np.mean(epe_edge_list) if epe_edge_list else 0.0
    epe_flat = np.mean(epe_flat_list) if epe_flat_list else 0.0
    d1 = 100 * np.mean(out_list)

    print(f"\n{'='*60}")
    print(f"SceneFlow Validation Results ({total_samples} samples)")
    print(f"{'='*60}")
    print(f"  EPE:       {epe:.4f}")
    print(f"  EPE_edge:  {epe_edge:.4f}")
    print(f"  EPE_flat:  {epe_flat:.4f}")
    print(f"  Edge/Flat: {epe_edge/epe_flat:.2f}x" if epe_flat > 0 else "  Edge/Flat: N/A")
    print(f"  D1 (>3px): {d1:.2f}%")
    print(f"{'='*60}\n")
    return {'scene-disp-epe': epe, 'scene-disp-epe-edge': epe_edge, 'scene-disp-epe-flat': epe_flat, 'scene-disp-d1': d1}


@torch.no_grad()
def validate_middlebury(model, iters=32, split='F', mixed_prec=False, device=None, root=None, val_samples=500):
    """ Peform validation using the Middlebury-V3 dataset """
    if root is None:
        root = DEFAULT_MIDDLEBURY_ROOT
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, root=root, split=split)

    total_samples = min(val_samples, len(val_dataset)) if val_samples > 0 else len(val_dataset)

    out_list, epe_list = [], []
    for val_id in range(total_samples):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()

        occ_mask = Image.open(imageL_file.replace('im0.png', 'mask0nocc.png')).convert('L')
        occ_mask = np.ascontiguousarray(occ_mask, dtype=np.float32).flatten()

        val = (valid_gt.reshape(-1) >= 0.5) & (flow_gt[0].reshape(-1) < 192) & (occ_mask==255)
        out = (epe_flattened > 2.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print(f"\n{'='*60}")
    print(f"Middlebury-{split} Validation Results ({total_samples} samples)")
    print(f"{'='*60}")
    print(f"  EPE: {epe:.4f}")
    print(f"  D1 (>2px): {d1:.2f}%")
    print(f"{'='*60}\n")
    return {f'middlebury{split}-epe': epe, f'middlebury{split}-d1': d1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/sceneflow/sceneflow.pth')
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='/root/autodl-tmp/stereo/model_cache/sceneflow.pth')
    parser.add_argument('--dataset', help="dataset for evaluation", default='sceneflow',
                        choices=["eth3d", "kitti", "sceneflow"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--val_samples', type=int, default=2000, help='number of samples to validate (0 for all)')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for evaluation (default: 8 for 32GB V100)')

    # Dataset paths
    parser.add_argument('--dataset_root', type=str, default=DEFAULT_DATASET_ROOT, help='root directory for all datasets')
    parser.add_argument('--sceneflow_root', type=str, default=None, help='path to SceneFlow dataset')
    parser.add_argument('--kitti_root', type=str, default=None, help='path to KITTI dataset')
    parser.add_argument('--eth3d_root', type=str, default=None, help='path to ETH3D dataset')
    parser.add_argument('--middlebury_root', type=str, default=None, help='path to Middlebury dataset')

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
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

    # Auto-detect device (GPU if available, otherwise CPU)
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

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")

    if args.dataset == 'eth3d':
        validate_eth3d(model, iters=args.valid_iters, mixed_prec=args.mixed_precision, device=args.device, root=args.eth3d_root, val_samples=args.val_samples)

    elif args.dataset == 'kitti':
        validate_kitti(model, iters=args.valid_iters, mixed_prec=args.mixed_precision, device=args.device, root=args.kitti_root, val_samples=args.val_samples)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, iters=args.valid_iters, split=args.dataset[-1], mixed_prec=args.mixed_precision, device=args.device, root=args.middlebury_root, val_samples=args.val_samples)

    elif args.dataset == 'sceneflow':
        validate_sceneflow(model, iters=args.valid_iters, mixed_prec=args.mixed_precision, device=args.device, root=args.sceneflow_root, val_samples=args.val_samples)
