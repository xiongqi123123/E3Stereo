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
def validate_eth3d(model, iters=32, mixed_prec=False):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, imageR_file, GT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

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
def validate_kitti(model, iters=32, mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, image_set='training')
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

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


@torch.no_grad()
def validate_sceneflow(model, iters=32, mixed_prec=False, args=None):
    """ Peform validation using the Scene Flow (TEST) split """
    model.eval()
    # 从 args 获取 edge_source，确保验证数据集与训练时一致
    edge_source = getattr(args, 'edge_source', 'rcf') if args is not None else 'rcf'
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True, edge_source=edge_source)

    out_list, epe_list = [], []
    epe_edge_list, epe_flat_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        data_item = val_dataset[val_id]
        # 数据可能包含 edge（如果 edge_source='gt'）
        if len(data_item) == 6:
            meta, image1, image2, flow_gt, valid_gt, left_edge = data_item
        else:
            meta, image1, image2, flow_gt, valid_gt = data_item
            left_edge = None

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        
        # 如果从数据集获取了 edge，需要添加 batch 维度并移到 GPU
        if left_edge is not None:
            left_edge = left_edge[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True, left_edge=left_edge)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

        # epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()
        epe = torch.abs(flow_pr - flow_gt)

        epe = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        if(np.isnan(epe[val].mean().item())):
            continue

        out = (epe > 3.0)
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

        # 若有 GT edge，分别统计 edge 与 flat 区域的 EPE
        if left_edge is not None:
            edge_flat = left_edge.cpu().squeeze().flatten()
            if edge_flat.shape[0] == epe.shape[0]:
                edge_val = val & (edge_flat > 0.5)
                flat_val = val & (edge_flat <= 0.5)
                if edge_val.any():
                    epe_edge_list.append(epe[edge_val].mean().item())
                if flat_val.any():
                    epe_flat_list.append(epe[flat_val].mean().item())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)
    if args is not None and getattr(args, 'edge_context_fusion', False):
        f = open(f'test/{getattr(args, "edge_fusion_mode", "film")}-context-{getattr(args, "edge_source", "rcf")}-test.txt', 'a')
    elif args is not None and getattr(args, 'edge_guided_upsample', False):
        f = open(f'test/{getattr(args, "edge_upsample_fusion_mode", "film")}-upsample-{getattr(args, "edge_source", "rcf")}-test.txt', 'a')
    elif args is not None and getattr(args, 'edge_guided_disp_head', False):
        f = open(f'test/{getattr(args, "edge_disp_fusion_mode", "film")}-disp-head-{getattr(args, "edge_source", "rcf")}-test.txt', 'a')
    elif args is not None and getattr(args, 'edge_guided_cost_agg', False):
        f = open(f'test/{getattr(args, "edge_cost_agg_fusion_mode", "film")}-cost-agg-{getattr(args, "edge_source", "rcf")}-test.txt', 'a')
    elif args is not None and getattr(args, 'edge_guided_gwc', False):
        f = open(f'test/{getattr(args, "edge_gwc_fusion_mode", "film")}-gwc-{getattr(args, "edge_source", "rcf")}-test.txt', 'a')
    elif args is not None and getattr(args, 'edge_motion_encoder', False):
        f = open(f'test/{getattr(args, "edge_motion_fusion_mode", "film")}-motion-encoder-{getattr(args, "edge_source", "rcf")}-test.txt', 'a')
    elif args is not None and getattr(args, 'boundary_only_refinement', False):
        f = open(f'test/{getattr(args, "edge_refinement_fusion_mode", "film")}-boundary-only-{getattr(args, "edge_source", "rcf")}-test.txt', 'a')
    elif args is not None and getattr(args, 'edge_guided_refinement', False):
        f = open(f'test/{getattr(args, "edge_refinement_fusion_mode", "film")}-refinement-{getattr(args, "edge_source", "rcf")}-test.txt', 'a')
    else:
        f = open('test/test.txt', 'a')
    f.write("Validation Scene Flow: %f, %f\n" % (epe, d1))

    results = {'scene-disp-epe': epe, 'scene-disp-d1': d1}
    print("Validation Scene Flow: EPE %f, D1 %f" % (epe, d1))
    if len(epe_edge_list) > 0:
        epe_edge = np.mean(epe_edge_list)
        print("  EPE (edge): %f" % epe_edge)
        results['scene-disp-epe-edge'] = epe_edge
    if len(epe_flat_list) > 0:
        epe_flat = np.mean(epe_flat_list)
        print("  EPE (flat): %f" % epe_flat)
        results['scene-disp-epe-flat'] = epe_flat
    return results


@torch.no_grad()
def validate_middlebury(model, iters=32, split='F', mixed_prec=False):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, split=split)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

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
    parser.add_argument('--edge_source', type=str, default='rcf', choices=['rcf', 'gt'],
                        help="edge source: 'rcf' use RCF online prediction, 'gt' use gtedge pre-generated edge.")
    parser.add_argument('--edge_model', type=str, default='../RCF-PyTorch/rcf.pth', help='path to the edge model')
    parser.add_argument('--edge_context_fusion', action='store_true',
                        help='fuse edge into context features for GRU input')
    parser.add_argument('--edge_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated'],
                        help='edge-context fusion: concat/film/gated')
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
    args = parser.parse_args()

    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")

    if args.dataset == 'eth3d':
        validate_eth3d(model, iters=args.valid_iters, mixed_prec=args.mixed_precision)

    elif args.dataset == 'kitti':
        validate_kitti(model, iters=args.valid_iters, mixed_prec=args.mixed_precision)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, iters=args.valid_iters, split=args.dataset[-1], mixed_prec=args.mixed_precision)

    elif args.dataset == 'sceneflow':
        validate_sceneflow(model, iters=args.valid_iters, mixed_prec=args.mixed_precision, args=args)
