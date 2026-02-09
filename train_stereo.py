import os
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from core.igev_stereo import IGEVStereo
from evaluate_stereo import *
import core.stereo_datasets as datasets
import torch.nn.functional as F
import datetime
try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


def sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, loss_gamma=0.9, max_disp=192,
                  edge=None):
    """ Loss function defined over sequence of disp predictions """

    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    mag = torch.sum(disp_gt**2, dim=1).sqrt()
    valid_mask = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert valid_mask.shape == disp_gt.shape, [valid_mask.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid_mask.bool()]).any()

    disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[valid_mask.bool()], disp_gt[valid_mask.bool()], size_average=True)
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        assert i_loss.shape == valid_mask.shape, [i_loss.shape, valid_mask.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss[valid_mask.bool()].mean()

    epe_map = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt().unsqueeze(1)  # [B, 1, H, W]
    epe_flat_all = epe_map.view(-1)[valid_mask.view(-1)]

    # 先初始化 metrics，把 edge/flat 的 key 固定加入，保证日志里总会有这两个字段
    metrics = {
        'epe': epe_flat_all.mean().item(),
        '1px': (epe_flat_all < 1).float().mean().item(),
        '3px': (epe_flat_all < 3).float().mean().item(),
        '5px': (epe_flat_all < 5).float().mean().item(),
        'epe_edge': 0.0,  # 默认 0，如果当前 batch 没有 edge 像素就保持为 0
        'epe_flat': 0.0,  # 默认 0，后面会用真实 flat 区域的 EPE 覆盖
    }

    # 若有 edge，分别统计 edge 区域和 flat 区域的 EPE
    if edge is not None:
        if edge.shape[-2:] != valid_mask.shape[-2:]:
            edge = F.interpolate(edge.float(), size=valid_mask.shape[-2:], mode='bilinear', align_corners=False)

        edge_mask = (edge > 0.5) & valid_mask
        flat_mask = (~(edge > 0.5)) & valid_mask

        if edge_mask.any():
            epe_edge = epe_map[edge_mask].view(-1)
            metrics['epe_edge'] = epe_edge.mean().item()
        # 一般 flat 区域总是存在，这里仍然做一下保护
        if flat_mask.any():
            epe_flat = epe_map[flat_mask].view(-1)
            metrics['epe_flat'] = epe_flat.mean().item()

    return disp_loss, metrics

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler

class Logger:
    SUM_FREQ = 100
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=args.logdir)

    def _print_training_status(self):
        keys = sorted(self.running_loss.keys())
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in keys]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = " ".join(f"{k}={v:.4f}" for k, v in zip(keys, metrics_data))
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str}{metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=args.logdir)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=args.logdir)

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    model = nn.DataParallel(IGEVStereo(args))
    print("Parameter Count: %d" % count_parameters(model))

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    logger = Logger(model, scheduler)

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")
    model.cuda()
    model.train()
    model.module.freeze_bn() # We keep BatchNorm frozen

    validation_frequency = 10000

    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    global_batch_num = 0
    while should_keep_training:

        for i_batch, (meta, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            # data_blob 现在包含: image1, image2, disp_gt, valid, edge
            if len(data_blob) == 5:
                image1, image2, disp_gt, valid, left_edge = [x.cuda() if x is not None else None for x in data_blob]
            else:
                # 兼容旧版本（没有 edge）
                image1, image2, disp_gt, valid = [x.cuda() for x in data_blob]
                left_edge = None

            assert model.training
            disp_init_pred, disp_preds, left_edge_out = model(image1, image2, iters=args.train_iters, left_edge=left_edge)
            assert model.training
            # edge_for_metrics: 用 dataset 原始 edge (0/1) 做 epe_edge/epe_flat 统计，避免模型内部 edge_scale 缩放后 edge>0.5 的 mask 为空
            # 当 dataset 无 edge (如 edge_source=rcf) 时退化为 left_edge_out
            edge_for_metrics = left_edge if left_edge is not None else left_edge_out

            loss, metrics = sequence_loss(
                disp_preds, disp_init_pred, disp_gt, valid, max_disp=args.max_disp,
                edge=edge_for_metrics,
            )
            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            logger.push(metrics)

            if total_steps % validation_frequency == validation_frequency - 1:
                save_path = Path(args.logdir + '/%d_%s.pth' % (total_steps + 1, args.name))
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save(model.state_dict(), save_path)
                if 'sceneflow' in args.train_datasets:
                    results = validate_sceneflow(model.module, iters=args.valid_iters, args=args)
                elif 'kitti' in args.train_datasets:
                    results = validate_kitti(model.module, iters=args.valid_iters)
                else: 
                    raise Exception('Unknown validation dataset.')
                logger.write_dict(results)
                model.train()
                model.module.freeze_bn()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    print("FINISHED TRAINING")
    logger.close()
    PATH = args.logdir + '/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='igev-stereo', help="name your experiment")
    parser.add_argument('--restore_ckpt', default=None, help="load the weights from a specific checkpoint")
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float16', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--logdir', default='./checkpoints/sceneflow', help='the directory to save logs and checkpoints')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=200000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 736], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=22, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32, help='number of disp-field updates during validation forward pass')

    # Architecure choices
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4], help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.2, 0.4], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    
    # Edge augmentation (需配合 edge_model 使用)
    parser.add_argument('--edge_source', type=str, default=None, choices=['rcf', 'gt'],
                        help="edge source: 'rcf' use RCF online prediction, 'gt' use gtedge pre-generated edge.")
    parser.add_argument('--edge_model', type=str, default=None,
                        help='path to the edge model (当 edge_source=rcf 且开启任意 edge_* 时必需)')
    parser.add_argument('--edge_context_fusion', action='store_true',
                        help='fuse edge into context features for GRU input')
    parser.add_argument('--edge_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated'],
                        help='edge-context fusion: concat(hard), film(soft), gated(soft)')
    parser.add_argument('--edge_guided_upsample', action='store_true',
                        help='use edge to guide disparity upsampling for sharper boundaries')
    parser.add_argument('--edge_upsample_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated', 'mlp'],
                        help='edge-guided upsampling fusion: concat/film/gated/mlp')
    parser.add_argument('--edge_guided_disp_head', action='store_true',
                        help='use edge to guide delta_disp prediction in GRU update')
    parser.add_argument('--edge_disp_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated', 'mlp'],
                        help='edge-disp fusion: concat/film/gated/mlp')
    parser.add_argument('--edge_guided_cost_agg', action='store_true',
                        help='inject edge into cost_agg (Hourglass) for better init_disp at boundaries')
    parser.add_argument('--edge_cost_agg_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated'],
                        help='edge-cost_agg fusion: concat/film/gated')
    parser.add_argument('--edge_guided_gwc', action='store_true',
                        help='inject edge into GWC corr_feature_att for boundary-aware initial cost')
    parser.add_argument('--edge_gwc_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated'],
                        help='edge-GWC fusion: concat/film/gated')
    parser.add_argument('--edge_motion_encoder', action='store_true',
                        help='inject edge into Motion Encoder for boundary-aware motion features')
    parser.add_argument('--edge_motion_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated'],
                        help='edge-motion encoder fusion: concat/film/gated')
    parser.add_argument('--edge_guided_refinement', action='store_true',
                        help='edge-guided disparity refinement post-processing for sharper boundaries')
    parser.add_argument('--boundary_only_refinement', action='store_true',
                        help='refinement only at boundary regions (mask by edge), lower compute in flat areas')
    parser.add_argument('--edge_refinement_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated'],
                        help='edge-refinement fusion: concat/film/gated')
    args = parser.parse_args()

    torch.manual_seed(666)
    np.random.seed(666)

    Path(args.logdir).mkdir(exist_ok=True, parents=True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        filename=f"{args.logdir}/train{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    train(args)
