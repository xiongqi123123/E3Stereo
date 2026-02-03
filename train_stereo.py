# import os
# import argparse
# import logging
# import numpy as np
# from pathlib import Path
# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
# import torch
# import torch.nn as nn
# from evaluate_stereo import *
# import torch.optim as optim
# # from core.igev_stereo import IGEVStereo
# # from core.igev_d1_gate import IGEVStereo
# # from core.igev_d1_spat import IGEVStereo
# from core.igev_d2_agg import IGEVStereo

# import core.stereo_datasets as datasets
# import torch.nn.functional as F
# try:
#     from torch.cuda.amp import GradScaler
# except:
#     class GradScaler:
#         def __init__(self):
#             pass
#         def scale(self, loss):
#             return loss
#         def unscale_(self, optimizer):
#             pass
#         def step(self, optimizer):
#             optimizer.step()
#         def update(self):
#             pass


# def sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, edge=None, loss_gamma=0.9, max_disp=192, edge_smooth_weight=0.05):
#     """ Loss function defined over sequence of disp predictions """

#     n_predictions = len(disp_preds)
#     assert n_predictions >= 1
#     disp_loss = 0.0
#     mag = torch.sum(disp_gt**2, dim=1).sqrt()
#     valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
#     assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
#     assert not torch.isinf(disp_gt[valid.bool()]).any()


#     disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[valid.bool()], disp_gt[valid.bool()], size_average=True)

#     for i in range(n_predictions):
#         adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
#         i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
#         i_loss = (disp_preds[i] - disp_gt).abs()
#         assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
#         disp_loss += i_weight * i_loss[valid.bool()].mean()
    
#     if edge is not None:
#         # 只对最后一个预测加（最稳、也最像 EdgeStereo）
#         disp_loss = disp_loss + edge_smooth_weight * edge_aware_smoothness_loss(disp_preds[-1], edge, valid)

#     epe = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt()
#     epe = epe.view(-1)[valid.view(-1)]

#     metrics = {
#         'epe': epe.mean().item(),
#         '1px': (epe < 1).float().mean().item(),
#         '3px': (epe < 3).float().mean().item(),
#         '5px': (epe < 5).float().mean().item(),
#         'edge_loss': edge_smooth_weight * edge_aware_smoothness_loss(disp_preds[-1], edge, valid).item(),
#         'disp_loss': disp_loss.item(),
#     }
#     return disp_loss, metrics

# def gradient_x(img):
#     return img[:, :, :, 1:] - img[:, :, :, :-1]

# def gradient_y(img):
#     return img[:, :, 1:, :] - img[:, :, :-1, :]

# def edge_aware_smoothness_loss(disp, edge, valid=None, eps=1e-3):
#     """
#     disp:  [B,1,H,W]  (use final upsampled disparity)
#     edge:  [B,1,H,W]  in [0,1]
#     valid: [B,1,H,W]  optional mask
#     """
#     # disparity gradients
#     dx = gradient_x(disp).abs()
#     dy = gradient_y(disp).abs()

#     # edge weights: non-edge -> weight ~1, edge -> weight -> 0
#     # 方案A（最少超参）：w = 1 - edge
#     edge_x = edge[:, :, :, :-1]
#     edge_y = edge[:, :, :-1, :]
#     wx = (1.0 - edge_x).clamp(0.0, 1.0)
#     wy = (1.0 - edge_y).clamp(0.0, 1.0)

#     loss = (wx * dx).mean() + (wy * dy).mean()

#     if valid is not None:
#         # 对齐 valid 尺寸
#         valid_x = valid[:, :, :, :-1]
#         valid_y = valid[:, :, :-1, :]
#         loss = ((wx * dx)[valid_x.bool()].mean() + (wy * dy)[valid_y.bool()].mean())

#     return loss


# def fetch_optimizer(args, model):
#     """ Create the optimizer and learning rate scheduler """
#     optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

#     scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
#             pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
#     return optimizer, scheduler

# class Logger:
#     SUM_FREQ = 100
#     def __init__(self, model, scheduler):
#         self.model = model
#         self.scheduler = scheduler
#         self.total_steps = 0
#         self.running_loss = {}
#         self.writer = SummaryWriter(log_dir=args.logdir)

#     def _print_training_status(self):
#         metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
#         training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
#         metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
#         # print the training status
#         logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

#         if self.writer is None:
#             self.writer = SummaryWriter(log_dir=args.logdir)

#         for k in self.running_loss:
#             self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
#             self.running_loss[k] = 0.0

#     def push(self, metrics):
#         self.total_steps += 1

#         for key in metrics:
#             if key not in self.running_loss:
#                 self.running_loss[key] = 0.0

#             self.running_loss[key] += metrics[key]

#         if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
#             self._print_training_status()
#             self.running_loss = {}

#     def write_dict(self, results):
#         if self.writer is None:
#             self.writer = SummaryWriter(log_dir=args.logdir)

#         for key in results:
#             self.writer.add_scalar(key, results[key], self.total_steps)

#     def close(self):
#         self.writer.close()


# def train(args):

#     model = nn.DataParallel(IGEVStereo(args))
#     print("Parameter Count: %d" % count_parameters(model))

#     train_loader = datasets.fetch_dataloader(args)
#     optimizer, scheduler = fetch_optimizer(args, model)
#     total_steps = 0
#     logger = Logger(model, scheduler)

#     if args.restore_ckpt is not None:
#         assert args.restore_ckpt.endswith(".pth")
#         logging.info("Loading checkpoint...")
#         checkpoint = torch.load(args.restore_ckpt)
#         model.load_state_dict(checkpoint, strict=True)
#         logging.info(f"Done loading checkpoint")
#     model.cuda()
#     model.train()
#     model.module.freeze_bn() # We keep BatchNorm frozen

#     validation_frequency = 10000

#     scaler = GradScaler(enabled=args.mixed_precision)

#     should_keep_training = True
#     global_batch_num = 0
#     while should_keep_training:

#         for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
#             optimizer.zero_grad()
#             image1, image2, disp_gt, valid = [x.cuda() for x in data_blob]

#             ####### Change: EdgeLoss #######
#             assert model.training
#             # disp_init_pred, disp_preds = model(image1, image2, iters=args.train_iters)
#             disp_init_pred, disp_preds, left_edge_raw = model(image1, image2, iters=args.train_iters)
#             assert model.training

#             # loss, metrics = sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, max_disp=args.max_disp)
#             loss, metrics = sequence_loss(disp_preds, disp_init_pred, disp_gt, valid,
#                               edge=left_edge_raw, max_disp=args.max_disp)
#             ####### Change: EdgeLoss #######
#             logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
#             logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
#             global_batch_num += 1
#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

#             scaler.step(optimizer)
#             scheduler.step()
#             scaler.update()
#             logger.push(metrics)

#             if total_steps % validation_frequency == validation_frequency - 1:
#                 save_path = Path(args.logdir + '/%d_%s.pth' % (total_steps + 1, args.name))
#                 logging.info(f"Saving file {save_path.absolute()}")
#                 torch.save(model.state_dict(), save_path)
#                 if 'sceneflow' in args.train_datasets:
#                     results = validate_sceneflow(model.module, iters=args.valid_iters)
#                 elif 'kitti' in args.train_datasets:
#                     results = validate_kitti(model.module, iters=args.valid_iters)
#                 else: 
#                     raise Exception('Unknown validation dataset.')
#                 logger.write_dict(results)
#                 model.train()
#                 model.module.freeze_bn()

#             total_steps += 1

#             if total_steps > args.num_steps:
#                 should_keep_training = False
#                 break

#     print("FINISHED TRAINING")
#     logger.close()
#     PATH = args.logdir + '/%s.pth' % args.name
#     torch.save(model.state_dict(), PATH)

#     return PATH


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--name', default='igev-stereo', help="name your experiment")
#     parser.add_argument('--restore_ckpt', default=None, help="load the weights from a specific checkpoint")
#     parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
#     parser.add_argument('--precision_dtype', default='float16', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')
#     parser.add_argument('--logdir', default='./checkpoints/sceneflow', help='the directory to save logs and checkpoints')

#     # Training parameters
#     parser.add_argument('--batch_size', type=int, default=8, help="batch size used during training.")
#     parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
#     parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
#     parser.add_argument('--num_steps', type=int, default=200000, help="length of training schedule.")
#     parser.add_argument('--image_size', type=int, nargs='+', default=[320, 736], help="size of the random image crops used during training.")
#     parser.add_argument('--train_iters', type=int, default=22, help="number of updates to the disparity field in each forward pass.")
#     parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

#     # Validation parameters
#     parser.add_argument('--valid_iters', type=int, default=32, help='number of disp-field updates during validation forward pass')

#     # Architecure choices
#     parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
#     parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
#     parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
#     parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
#     parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
#     parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

#     # Data augmentation
#     parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
#     parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4], help='color saturation')
#     parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
#     parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.2, 0.4], help='re-scale the images randomly')
#     parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    
#     # Edge augmentation
#     parser.add_argument('--edge_model', type=str, default='../RCF-PyTorch/rcf.pth', help='path to the edge model')

#     args = parser.parse_args()

#     torch.manual_seed(666)
#     np.random.seed(666)
    
#     logging.basicConfig(level=logging.INFO,
#                         format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

#     Path(args.logdir).mkdir(exist_ok=True, parents=True)

#     train(args)

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


def sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, loss_gamma=0.9, max_disp=192):
    """ Loss function defined over sequence of disp predictions """

    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    mag = torch.sum(disp_gt**2, dim=1).sqrt()
    valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()


    disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[valid.bool()], disp_gt[valid.bool()], size_average=True)
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
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
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

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

        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            image1, image2, disp_gt, valid = [x.cuda() for x in data_blob]

            assert model.training
            disp_init_pred, disp_preds = model(image1, image2, iters=args.train_iters)
            assert model.training

            loss, metrics = sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, max_disp=args.max_disp)
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
                    results = validate_sceneflow(model.module, iters=args.valid_iters)
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
    parser.add_argument('--edge_model', type=str, default='../RCF-PyTorch/rcf.pth', help='path to the edge model (required when edge_context_fusion or edge_guided_upsample)')
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
    args = parser.parse_args()

    torch.manual_seed(666)
    np.random.seed(666)
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path(args.logdir).mkdir(exist_ok=True, parents=True)

    train(args)
