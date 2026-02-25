from __future__ import print_function, division
import sys
sys.path.append('core')

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from igev_stereo import IGEVStereo, autocast
import stereo_datasets as datasets
import cv2
from utils.utils import InputPadder
from PIL import Image

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def validate_eth3d(model, iters=32, mixed_prec=False, device=None):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
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
        # val = (valid_gt.flatten() >= 0.5)
        out = (epe_flattened > 1.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"ETH3D {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation ETH3D: EPE %f, D1 %f" % (epe, d1))
    return {'eth3d-epe': epe, 'eth3d-d1': d1}


@torch.no_grad()
def validate_kitti(model, iters=32, mixed_prec=False, device=None):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, image_set='training')
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
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
        # val = valid_gt.flatten() >= 0.5

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id+1)%10 == 0:
            logging.info(f"KITTI Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}. Runtime: {format(end-start, '.3f')}s ({format(1/(end-start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    print(f"Validation KITTI: EPE {epe}, D1 {d1}, {format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'kitti-epe': epe, 'kitti-d1': d1}


# @torch.no_grad()
# def validate_sceneflow(model, iters=32, mixed_prec=False, args=None, device=None):
#     """ Peform validation using the Scene Flow (TEST) split """
#     model.eval()
#     if device is None:
#         device = next(model.parameters()).device
#     # 从 args 获取 edge_source，确保验证数据集与训练时一致
#     edge_source = getattr(args, 'edge_source', 'rcf') if args is not None else 'rcf'
#     val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True, edge_source=edge_source)

#     out_list = []
#     epe_sum, epe_count = 0.0, 0
#     epe_edge_sum, epe_edge_count = 0.0, 0
#     epe_flat_sum, epe_flat_count = 0.0, 0
#     for val_id in tqdm(range(len(val_dataset))):
#         data_item = val_dataset[val_id]
#         # 有 GT edge 时数据集返回 6 个元素（供 metrics 用），与 edge_source 无关
#         if len(data_item) == 6:
#             meta, image1, image2, flow_gt, valid_gt, gt_edge = data_item
#         else:
#             meta, image1, image2, flow_gt, valid_gt = data_item
#             gt_edge = None

#         image1 = image1[None].to(device)
#         image2 = image2[None].to(device)
#         if gt_edge is not None:
#             gt_edge = gt_edge[None].to(device)

#         # 模型输入：仅 edge_source=='gt' 时传入 edge；geo/rcf 时模型内部预测
#         left_edge_for_model = gt_edge if (getattr(args, 'edge_source', None) == 'gt' and gt_edge is not None) else None

#         padder = InputPadder(image1.shape, divis_by=32)
#         image1, image2 = padder.pad(image1, image2)

#         with autocast(enabled=mixed_prec):
#             flow_pr = model(image1, image2, iters=iters, test_mode=True, left_edge=left_edge_for_model)
#         flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
#         assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

#         # epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()
#         epe = torch.abs(flow_pr - flow_gt)

#         epe = epe.flatten()
#         val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

#         if(np.isnan(epe[val].mean().item())):
#             continue

#         out = (epe > 3.0)
#         epe_sum += epe[val].sum().item()
#         epe_count += val.sum().item()
#         out_list.append(out[val].cpu().numpy())

#         # 若有 GT edge，分别统计 edge 与 flat 区域的 EPE（始终用 GT edge 做 mask）
#         if gt_edge is not None:
#             edge_flat = gt_edge.cpu().squeeze().flatten()
#             if edge_flat.shape[0] == epe.shape[0]:
#                 edge_val = val & (edge_flat > 0.5)
#                 flat_val = val & (edge_flat <= 0.5)
#                 if edge_val.any():
#                     epe_edge_sum += epe[edge_val].sum().item()
#                     epe_edge_count += edge_val.sum().item()
#                 if flat_val.any():
#                     epe_flat_sum += epe[flat_val].sum().item()
#                     epe_flat_count += flat_val.sum().item()

#     out_list = np.concatenate(out_list)

#     epe = epe_sum / epe_count if epe_count > 0 else 0.0
#     d1 = 100 * np.mean(out_list)
#     # os.makedirs(args.name, exist_ok=True)
#     f = open(f'checkpoints/{args.name}/test.txt', 'a')
#     f.write("Validation Scene Flow: %f, %f\n" % (epe, d1))

#     results = {'scene-disp-epe': epe, 'scene-disp-d1': d1}
#     print("Validation Scene Flow: EPE %f, D1 %f" % (epe, d1))
#     if epe_edge_count > 0:
#         epe_edge = epe_edge_sum / epe_edge_count
#         print("  EPE (edge): %f" % epe_edge)
#         f.write("  EPE (edge): %f\n" % epe_edge)
#         results['scene-disp-epe-edge'] = epe_edge
#     if epe_flat_count > 0:
#         epe_flat = epe_flat_sum / epe_flat_count
#         print("  EPE (flat): %f" % epe_flat)
#         f.write("  EPE (flat): %f\n" % epe_flat)
#         results['scene-disp-epe-flat'] = epe_flat
#     return results
@torch.no_grad()
def validate_sceneflow(model, iters=32, mixed_prec=False, args=None, device=None):
    """ Peform validation using the Scene Flow (TEST) split """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    edge_source = getattr(args, 'edge_source', 'rcf') if args is not None else 'rcf'
    val_dataset = datasets.SceneFlowDatasets(
        dstype='frames_finalpass', things_test=True, edge_source=edge_source
    )

    # 全部用 sum/count，避免 out_list + numpy concat 的 CPU/内存抖动
    epe_sum = torch.tensor(0.0, device=device)
    epe_cnt = torch.tensor(0.0, device=device)

    d1_sum  = torch.tensor(0.0, device=device)
    d1_cnt  = torch.tensor(0.0, device=device)

    epe_edge_sum = torch.tensor(0.0, device=device)
    epe_edge_cnt = torch.tensor(0.0, device=device)

    epe_flat_sum = torch.tensor(0.0, device=device)
    epe_flat_cnt = torch.tensor(0.0, device=device)

    for val_id in tqdm(range(len(val_dataset))):
        data_item = val_dataset[val_id]

        # 有 GT edge 时数据集返回 6 个元素（供 metrics 用），与 edge_source 无关
        if len(data_item) == 6:
            meta, image1, image2, flow_gt, valid_gt, gt_edge = data_item
        else:
            meta, image1, image2, flow_gt, valid_gt = data_item
            gt_edge = None

        # ===== 关键：全部放到 GPU =====
        image1   = image1[None].to(device, non_blocking=True)
        image2   = image2[None].to(device, non_blocking=True)
        flow_gt  = flow_gt.to(device, non_blocking=True)       # shape: [H,W] or [1,H,W]
        valid_gt = valid_gt.to(device, non_blocking=True)      # shape: [H,W] or [1,H,W] depending on dataset

        if gt_edge is not None:
            gt_edge = gt_edge.to(device, non_blocking=True)    # shape: [H,W] (通常)
            gt_edge_b = gt_edge[None]                          # [1,H,W] 给模型用
        else:
            gt_edge_b = None

        # 模型输入：仅 edge_source=='gt' 时传入 edge；geo/rcf 时模型内部预测
        left_edge_for_model = (
            gt_edge_b if (getattr(args, 'edge_source', None) == 'gt' and gt_edge_b is not None)
            else None
        )

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True, left_edge=left_edge_for_model)

        # ===== 关键：不要 .cpu()，保持在 GPU =====
        flow_pr = padder.unpad(flow_pr).squeeze(0)

        # 保持你原来的 assert 语义：比 shape（这里 flow_gt 在 GPU）
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

        # 你原代码：epe = abs(pred - gt)
        epe = (flow_pr - flow_gt).abs().flatten()

        # valid / max_disp mask（全部在 GPU）
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        if val.sum() == 0:
            continue

        epe_val = epe[val]
        if torch.isnan(epe_val.mean()):
            continue

        # EPE sum/count
        epe_sum += epe_val.sum()
        epe_cnt += val.sum()

        # D1: epe > 3
        out = (epe > 3.0)
        d1_sum += out[val].float().sum()
        d1_cnt += val.sum()

        # edge/flat 指标：始终用 GT edge 做 mask
        if gt_edge is not None:
            edge_flat = gt_edge.flatten()
            if edge_flat.numel() == epe.numel():
                edge_val = val & (edge_flat > 0.5)
                flat_val = val & (edge_flat <= 0.5)

                if edge_val.any():
                    epe_edge_sum += epe[edge_val].sum()
                    epe_edge_cnt += edge_val.sum()

                if flat_val.any():
                    epe_flat_sum += epe[flat_val].sum()
                    epe_flat_cnt += flat_val.sum()

    # ===== 只在最后把标量拿回 CPU =====
    epe = (epe_sum / torch.clamp(epe_cnt, min=1)).item()
    d1  = (100.0 * d1_sum / torch.clamp(d1_cnt, min=1)).item()

    # 记录结果
    f = open(f'checkpoints/{args.name}/test.txt', 'a')
    f.write("Validation Scene Flow: %f, %f\n" % (epe, d1))

    results = {'scene-disp-epe': epe, 'scene-disp-d1': d1}
    print("Validation Scene Flow: EPE %f, D1 %f" % (epe, d1))

    if epe_edge_cnt.item() > 0:
        epe_edge = (epe_edge_sum / epe_edge_cnt).item()
        print("  EPE (edge): %f" % epe_edge)
        f.write("  EPE (edge): %f\n" % epe_edge)
        results['scene-disp-epe-edge'] = epe_edge

    if epe_flat_cnt.item() > 0:
        epe_flat = (epe_flat_sum / epe_flat_cnt).item()
        print("  EPE (flat): %f" % epe_flat)
        f.write("  EPE (flat): %f\n" % epe_flat)
        results['scene-disp-epe-flat'] = epe_flat

    f.close()
    return results


@torch.no_grad()
def validate_middlebury(model, iters=32, split='F', mixed_prec=False, device=None):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, split=split)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
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
        logging.info(f"Middlebury Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print(f"Validation Middlebury{split}: EPE {epe}, D1 {d1}")
    return {f'middlebury{split}-epe': epe, f'middlebury{split}-d1': d1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='igev-stereo', help="name your experiment")
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use')
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/sceneflow/sceneflow.pth')
    parser.add_argument('--dataset', help="dataset for evaluation", default='sceneflow', choices=["eth3d", "kitti", "sceneflow"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    parser.add_argument('--edge_source', type=str, default='rcf', choices=['rcf', 'gt', 'geo'],
                        help="edge source: 'rcf' use RCF online prediction, 'gt' use gtedge pre-generated edge, 'geo' use GeoEdgeNet online prediction.")
    parser.add_argument('--edge_model', type=str, default='../RCF-PyTorch/rcf.pth', help='path to the edge model (RCF or GeoEdgeNet checkpoint)')
    parser.add_argument('--edge_context_fusion', action='store_true',
                        help='fuse edge into context features for GRU input')
    parser.add_argument('--edge_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated'],
                        help='edge-context fusion: concat/film/gated')
    parser.add_argument('--edge_floor', type=float, default=0.0,
                        help='全局 edge 下限（0=不限制）')
    parser.add_argument('--edge_context_film_gamma_min', type=float, default=0.0,
                        help='Context FiLM γ 在有边像素上的下限（0=不限制）')
    parser.add_argument('--edge_guided_upsample', action='store_true',
                        help='use edge to guide disparity upsampling for sharper boundaries')
    parser.add_argument('--edge_upsample_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated', 'mlp'])
    parser.add_argument('--edge_guided_disp_head', action='store_true')
    parser.add_argument('--edge_disp_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated', 'mlp'])
    parser.add_argument('--edge_guided_cost_agg', action='store_true',
                        help='inject edge into cost_agg (Hourglass) for better init_disp')
    parser.add_argument('--edge_cost_agg_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated'])
    parser.add_argument('--edge_guided_gwc', action='store_true',
                        help='inject edge into GWC corr_feature_att for boundary-aware initial cost')
    parser.add_argument('--edge_gwc_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated'])
    parser.add_argument('--edge_motion_encoder', action='store_true',
                        help='inject edge into Motion Encoder for boundary-aware motion features')
    parser.add_argument('--edge_motion_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated'])
    parser.add_argument('--edge_guided_refinement', action='store_true',
                        help='edge-guided disparity refinement for sharper boundaries')
    parser.add_argument('--boundary_only_refinement', action='store_true',
                        help='refinement only at boundary regions (mask by edge)')
    parser.add_argument('--edge_refinement_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated'])
    # Edge-aware geo encoding volume (Combined_Geo_Encoding_Volume)
    parser.add_argument('--edge_geo_radius_aware', action='store_true',
                        help='use edge to adaptively shrink sampling radius in geo encoding volume')
    parser.add_argument('--edge_geo_radius_shrink', type=float, default=0.5,
                        help='lambda for shrinking geo sampling radius near edges (0=off, 0.5=moderate)')
    # Feature backbone: Edge-FiLM on x4 feature (left branch)
    parser.add_argument('--feature_edge_x4_film', action='store_true',
                        help='use edge-conditioned FiLM on x4 feature (left image only)')
    parser.add_argument('--feature_edge_x4_film_strength', type=float, default=1.0,
                        help='strength of FiLM modulation on x4 feature')
    args = parser.parse_args()

    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[args.device])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    device = torch.device(args.device)
    model.to(device)
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")

    if args.dataset == 'eth3d':
        validate_eth3d(model, iters=args.valid_iters, mixed_prec=args.mixed_precision, device=device)

    elif args.dataset == 'kitti':
        validate_kitti(model, iters=args.valid_iters, mixed_prec=args.mixed_precision, device=device)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, iters=args.valid_iters, split=args.dataset[-1], mixed_prec=args.mixed_precision, device=device)

    elif args.dataset == 'sceneflow':
        validate_sceneflow(model, iters=args.valid_iters, mixed_prec=args.mixed_precision, args=args, device=device)
