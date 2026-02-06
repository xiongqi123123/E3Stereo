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
from igev_stereo import IGEVStereo
from evaluate_stereo import *
import stereo_datasets as datasets
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


def extract_edge_from_image(image):
    """从 RGB 图像提取边缘强度图，用于统计 edge/flat EPE"""
    # 转换为灰度图 [B, 1, H, W]
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    if gray.max() > 1.0:
        gray = gray / 255.0

    # Sobel 算子
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=image.dtype, device=image.device).view(1, 1, 3, 3)

    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    edge_strength = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    # 归一化
    B = edge_strength.shape[0]
    edge_normalized = torch.zeros_like(edge_strength)
    for b in range(B):
        e = edge_strength[b]
        upper = torch.quantile(e.flatten(), 0.98)
        e_clipped = torch.clamp(e, 0, upper)
        edge_normalized[b] = e_clipped / (upper + 1e-8)

    return edge_normalized


def sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, edge_map=None,
                  loss_gamma=0.9, max_disp=192, compute_metrics=False):
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

    # 只在需要时计算详细 metrics
    if compute_metrics:
        epe = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt()
        epe_flat = epe.view(-1)[valid.view(-1)]

        # 计算 edge/flat EPE
        epe_edge_val = epe_flat.mean().item()
        epe_flat_val = epe_flat.mean().item()

        if edge_map is not None:
            edge_threshold = 0.3
            edge_mask = (edge_map > edge_threshold).squeeze(1)  # [B, H, W]
            flat_mask = (edge_map <= edge_threshold).squeeze(1)

            valid_sq = valid.squeeze(1)  # [B, H, W]
            is_edge = edge_mask & valid_sq
            is_flat = flat_mask & valid_sq

            if is_edge.sum() > 0:
                epe_edge_val = epe[is_edge].mean().item()
            if is_flat.sum() > 0:
                epe_flat_val = epe[is_flat].mean().item()

        d1 = (epe_flat > 3).float().mean().item()  # D1: >3px error rate
        metrics = {
            'epe': epe_flat.mean().item(),
            'epe_edge': epe_edge_val,
            'epe_flat': epe_flat_val,
            '1px': (epe_flat < 1).float().mean().item(),
            '3px': (epe_flat < 3).float().mean().item(),
            '5px': (epe_flat < 5).float().mean().item(),
            'd1': d1,
        }
        return disp_loss, metrics
    else:
        return disp_loss, None


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler


class Logger:
    SUM_FREQ = 100
    def __init__(self, model, scheduler, logdir):
        self.model = model
        self.scheduler = scheduler
        self.logdir = logdir
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=logdir)
        self.last_train_metrics = {}  # 保存最近的train metrics用于validation时打印

    def _print_training_status(self):
        avg_metrics = {k: self.running_loss[k]/Logger.SUM_FREQ for k in self.running_loss}
        lr = self.scheduler.get_last_lr()[0]

        # 清晰的格式化输出
        print("\n" + "="*80)
        print(f"[IGEV Baseline] Step: {self.total_steps+1:>6d} | LR: {lr:.2e}")
        print("-"*80)
        print(f"  EPE: {avg_metrics.get('epe', 0):.4f} | "
              f"Edge EPE: {avg_metrics.get('epe_edge', 0):.4f} | "
              f"Flat EPE: {avg_metrics.get('epe_flat', 0):.4f}")
        print(f"  1px: {avg_metrics.get('1px', 0)*100:.2f}% | "
              f"3px: {avg_metrics.get('3px', 0)*100:.2f}% | "
              f"5px: {avg_metrics.get('5px', 0)*100:.2f}%")
        if avg_metrics.get('epe_flat', 0) > 0:
            ratio = avg_metrics.get('epe_edge', 0) / avg_metrics.get('epe_flat', 1)
            print(f"  Edge/Flat Ratio: {ratio:.2f}x")
        print("="*80)

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.logdir)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        # 保存最近的train metrics用于validation时打印
        self.last_train_metrics = metrics.copy()

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.logdir)

        # 为validation结果添加 "val/" 前缀，便于TensorBoard区分
        for key in results:
            self.writer.add_scalar(f"val/{key}", results[key], self.total_steps)

    def print_train_val_comparison(self, val_results):
        """打印Train和Val的对比结果"""
        print("\n" + "="*80)
        print(f"[IGEV Baseline] Step: {self.total_steps:>6d} - Train vs Val Comparison")
        print("="*80)

        # 获取train metrics
        train_loss = self.last_train_metrics.get('loss', 0)
        train_epe = self.last_train_metrics.get('epe', 0)
        train_d1 = self.last_train_metrics.get('d1', 0) * 100  # 转为百分比

        # 获取val metrics (根据dataset不同key不同)
        val_epe = val_results.get('scene-disp-epe', val_results.get('kitti-epe', 0))
        val_d1 = val_results.get('scene-disp-d1', val_results.get('kitti-d1', 0))

        print(f"{'Metric':<15} {'Train':>12} {'Val':>12}")
        print("-"*40)
        print(f"{'Loss':<15} {train_loss:>12.4f} {'N/A':>12}")
        print(f"{'EPE':<15} {train_epe:>12.4f} {val_epe:>12.4f}")
        print(f"{'D1 (>3px)':<15} {train_d1:>11.2f}% {val_d1:>11.2f}%")

        # 如果有edge/flat EPE也打印出来
        if 'epe_edge' in self.last_train_metrics:
            train_epe_edge = self.last_train_metrics.get('epe_edge', 0)
            train_epe_flat = self.last_train_metrics.get('epe_flat', 0)
            val_epe_edge = val_results.get('scene-disp-epe-edge', 0)
            val_epe_flat = val_results.get('scene-disp-epe-flat', 0)
            print(f"{'EPE_edge':<15} {train_epe_edge:>12.4f} {val_epe_edge:>12.4f}")
            print(f"{'EPE_flat':<15} {train_epe_flat:>12.4f} {val_epe_flat:>12.4f}")

        print("="*80 + "\n")

    def close(self):
        self.writer.close()


def train(args):

    model = nn.DataParallel(IGEVStereo(args))
    print("Parameter Count: %d" % count_parameters(model))

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    logger = Logger(model, scheduler, args.logdir)

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")
    model.to(args.device)
    model.train()
    model.module.freeze_bn() # We keep BatchNorm frozen

    scaler = GradScaler(enabled=args.mixed_precision)

    total_steps = 0
    global_batch_num = 0
    best_epe = float('inf')  # 跟踪最优EPE

    # 创建checkpoint输出目录
    ckpt_output_dir = Path(args.logdir).parent / 'ckpt_output'
    ckpt_output_dir.mkdir(exist_ok=True, parents=True)
    logging.info(f"Best checkpoints will be saved to: {ckpt_output_dir}")

    # 使用 tqdm 显示总训练步数进度
    pbar = tqdm(total=args.num_steps, desc="Training", dynamic_ncols=True)

    while total_steps < args.num_steps:
        for i_batch, (_, *data_blob) in enumerate(train_loader):
            if total_steps >= args.num_steps:
                break
            optimizer.zero_grad()
            image1, image2, disp_gt, valid = [x.to(args.device) for x in data_blob]

            # 提取边缘用于统计 edge/flat EPE（不影响 loss，仅用于对比）
            edge_map = extract_edge_from_image(image1)

            assert model.training
            disp_init_pred, disp_preds = model(image1, image2, iters=args.train_iters)
            assert model.training

            # 判断是否需要计算详细metrics（每LOG_FREQ步计算一次）
            should_log = (total_steps + 1) % Logger.SUM_FREQ == 0

            loss, metrics = sequence_loss(disp_preds, disp_init_pred, disp_gt, valid,
                                          edge_map=edge_map, max_disp=args.max_disp,
                                          compute_metrics=should_log)

            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({'Loss': f"{loss.item():.3f}"})

            # 只在计算了metrics时更新logger
            if metrics is not None:
                metrics['loss'] = loss.item()
                logger.push(metrics)

            if total_steps > 0 and total_steps % args.valid_freq == 0:
                # 定期保存checkpoint
                save_path = Path(args.logdir + '/%d_%s.pth' % (total_steps + 1, args.name))
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save(model.state_dict(), save_path)

                # Validation
                if 'sceneflow' in args.train_datasets:
                    results = validate_sceneflow(model.module, iters=args.valid_iters, root=args.sceneflow_root, val_samples=args.val_samples)
                elif 'kitti' in args.train_datasets:
                    results = validate_kitti(model.module, iters=args.valid_iters, root=args.kitti_root, val_samples=args.val_samples)
                else:
                    raise Exception('Unknown validation dataset.')
                logger.write_dict(results)

                # 打印Train vs Val对比
                logger.print_train_val_comparison(results)

                # 根据EPE保存最优checkpoint
                current_epe = results.get('scene-disp-epe', results.get('kitti-epe', float('inf')))
                if current_epe < best_epe:
                    best_epe = current_epe
                    best_ckpt_path = ckpt_output_dir / f'{args.name}_best_epe{current_epe:.4f}_step{total_steps}.pth'
                    torch.save(model.state_dict(), best_ckpt_path)
                    logging.info(f"✓ New best EPE: {current_epe:.4f}, saved to {best_ckpt_path}")

                model.train()
                model.module.freeze_bn()

            total_steps += 1

    pbar.close()
    print("FINISHED TRAINING")
    logger.close()
    PATH = args.logdir + '/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='igev-stereo', help="name your experiment")
    parser.add_argument('--restore_ckpt',
                        # default='/root/autodl-tmp/stereo/model_cache/sceneflow.pth',
                        default=None,
                        help="load the weights from a specific checkpoint")
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float16', choices=['float16', 'bfloat16', 'float32'],
                        help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--logdir', default='/root/autodl-tmp/stereo/logs/baseline_igev', help='the directory to save logs and checkpoints')

    # Training parameters - 默认快速测试配置
    parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=25000, help="length of training schedule.")
    # 标准 SceneFlow 训练尺寸是 [320, 736]。
    # 如果显存吃紧，可以折中改为 [320, 640] 或 [288, 576]，但 256x448 确实太小了。
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 512],
                        help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=22, help="不要低于 16 ield in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # Validation parameters
    parser.add_argument('--valid_freq', type=int, default=500, help='validation frequency (steps)')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of disp-field updates during validation forward pass')
    parser.add_argument('--val_samples', type=int, default=100, help='number of samples to validate (0 for all)')

    # Architecure choices
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--max_disp', type=int, default=128, help="max disp of geometry encoding volume")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4], help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.2, 0.4], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')

    # Edge augmentation
    parser.add_argument('--edge_model', type=str, default='../model_cache/RCF-PyTorch/rcf.pth', help='path to the edge model')

    # Dataset paths
    parser.add_argument('--dataset_root', type=str, default='/root/autodl-tmp/stereo/dataset_cache', help='root directory for all datasets')
    parser.add_argument('--sceneflow_root', type=str, default=None, help='path to SceneFlow dataset (default: dataset_root/SceneFlow)')
    parser.add_argument('--kitti_root', type=str, default=None, help='path to KITTI dataset (default: dataset_root/KITTI)')
    parser.add_argument('--middlebury_root', type=str, default=None, help='path to Middlebury dataset (default: dataset_root/Middlebury/MiddEval3)')
    parser.add_argument('--eth3d_root', type=str, default=None, help='path to ETH3D dataset (default: dataset_root/ETH3D)')

    args = parser.parse_args()

    # Set default dataset paths if not specified
    if args.sceneflow_root is None:
        args.sceneflow_root = os.path.join(args.dataset_root, 'SceneFlow')
    if args.kitti_root is None:
        args.kitti_root = os.path.join(args.dataset_root, 'KITTI')
    if args.middlebury_root is None:
        args.middlebury_root = os.path.join(args.dataset_root, 'Middlebury/MiddEval3')
    if args.eth3d_root is None:
        args.eth3d_root = os.path.join(args.dataset_root, 'ETH3D')

    torch.manual_seed(666)
    np.random.seed(666)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path(args.logdir).mkdir(exist_ok=True, parents=True)

    # Auto-detect device (GPU if available, otherwise CPU)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {args.device}")

    train(args)
