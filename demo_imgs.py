import sys
sys.path.append('core')
DEVICE = 'cuda'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from igev_stereo import IGEVStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import cv2

def load_image(imfile):
    img = np.array(Image.open(imfile).convert('RGB')).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def collect_pairs_from_folder(root_folder, left_name=None, right_name=None, auto_detect=True):
    """
    递归扫描文件夹，按原图结构收集 left/right 对。
    支持两种结构：
      1) 同目录：im0.png + im1.png 或 left.png + right.png
      2) left/right 子目录（SceneFlow）：left/0006.png + right/0006.png
    返回 [(left_path, right_path, rel_dir, out_filename), ...]
    使用 os.walk(followlinks=True) 以正确遍历含 symlink 的目录（如 SceneFlow TEST）。
    """
    root = Path(root_folder).resolve()
    pairs = []
    seen = set()

    def add_pair(left_p, right_p, rel_d, out_fn=None):
        key = (str(left_p), str(right_p))
        if key not in seen:
            seen.add(key)
            pairs.append((str(left_p), str(right_p), str(rel_d), out_fn))

    if not auto_detect and left_name and right_name:
        # 显式指定：同目录模式
        for dirpath, _, filenames in os.walk(root, followlinks=True):
            if left_name in filenames and right_name in filenames:
                left_path = Path(dirpath) / left_name
                right_path = Path(dirpath) / right_name
                rel_dir = Path(dirpath).relative_to(root)
                add_pair(left_path, right_path, rel_dir, out_fn=None)
    else:
        # 自动检索：先尝试 left/right 子目录（SceneFlow）
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
        # 若无 left/right 子目录，尝试同目录模式
        if not pairs:
            for pattern in [('im0.png', 'im1.png'), ('left.png', 'right.png'), ('im0.ppm', 'im1.ppm')]:
                for dirpath, _, filenames in os.walk(root, followlinks=True):
                    if pattern[0] in filenames and pattern[1] in filenames:
                        left_path = Path(dirpath) / pattern[0]
                        right_path = Path(dirpath) / pattern[1]
                        rel_dir = Path(dirpath).relative_to(root)
                        add_pair(left_path, right_path, rel_dir, out_fn=None)
                if pairs:
                    break

    return sorted(pairs, key=lambda x: x[0])


def demo(args):
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    ckpt = torch.load(args.restore_ckpt, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt, strict=False)

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    # 文件夹模式：按原图结构推理并保存
    if args.folder:
        pairs = collect_pairs_from_folder(
            args.folder,
            left_name=getattr(args, 'left_name', None) if not args.auto else None,
            right_name=getattr(args, 'right_name', None) if not args.auto else None,
            auto_detect=args.auto,
        )
        if not pairs:
            print(f"No left/right pairs found in {args.folder}")
            return
        print(f"Found {len(pairs)} pairs in {args.folder}. Saving to {output_directory}/ (preserving structure)")

        out_name_default = (args.output_name or 'disp.png').replace('.png', '') + '.png'
        with torch.no_grad():
            for (imfile1, imfile2, rel_dir, out_filename) in tqdm(pairs, desc="Infer"):
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)

                padder = InputPadder(image1.shape, divis_by=32)
                image1, image2 = padder.pad(image1, image2)

                disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
                disp = disp.cpu().numpy()
                disp = padder.unpad(disp)

                out_dir = output_directory / rel_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                # left/right 子目录时用原图名（如 0006.png）；同目录时用 output_name
                out_name = (out_filename or out_name_default)
                if not out_name.lower().endswith('.png'):
                    out_name += '.png'
                plt.imsave(out_dir / out_name, disp.squeeze(), cmap='jet')
                if args.save_numpy:
                    np.save(out_dir / Path(out_name).with_suffix('.npy'), disp.squeeze())
        return

    # 原有 glob 模式
    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = disp.cpu().numpy()
            disp = padder.unpad(disp)
            file_stem = imfile1.split('/')[-2]
            filename = os.path.join(output_directory, f"{file_stem}.png")
            plt.imsave(output_directory / f"{file_stem}.png", disp.squeeze(), cmap='jet')
            if args.save_numpy:
                np.save(output_directory / f"{file_stem}.npy", disp.squeeze())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='/home/qi.xiong/StereoMatching/IGEV-Improve/EStereo/checkpoints/estero-20W-shared-edgeaware-0.5-3.0-0.5-refinement0.1/200000_estero-20W-shared-edgeaware-0.5-3.0-0.5-refinement0.1.pth')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('--folder', help="推理整个文件夹，按原图目录结构保存（递归扫描）")
    parser.add_argument('--auto', action='store_true', default=True, help="自动检索 left/right 或 im0/im1 结构（默认开启）")
    parser.add_argument('--no-auto', dest='auto', action='store_false', help="关闭自动检索，使用 --left_name/--right_name")
    parser.add_argument('--left_name', default='im0.png', help="非 auto 模式下左图文件名")
    parser.add_argument('--right_name', default='im1.png', help="非 auto 模式下右图文件名")
    parser.add_argument('--output_name', default='disp.png', help="同目录模式下的输出文件名（left/right 子目录时用原图名）")
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames (glob)", default="./demo-imgs/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames (glob)", default="./demo-imgs/*/im1.png")

    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/Middlebury/trainingH/*/im0.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/Middlebury/trainingH/*/im1.png")
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/ETH3D/two_view_training/*/im0.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/ETH3D/two_view_training/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="./demo-output/")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume"),
    parser.add_argument('--edge_source', type=str, default='shared', choices=['rcf', 'gt', 'geo', 'shared']),
    parser.add_argument('--edge_init_from_geo', type=str, default=None,
                        help='当 edge_source=shared 时，从此 GeoEdgeNet 加载 edge_head+edge_refine 权重（backbone 不加载，用 stereo 的）')
    parser.add_argument('--edge_model', type=str, default='../RCF-PyTorch/rcf.pth', help='path to the edge model')
    parser.add_argument('--edge_context_fusion', action='store_true',
                        help='fuse edge into context features for GRU input')
    parser.add_argument('--edge_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated'])
    parser.add_argument('--edge_floor', type=float, default=0.0)
    parser.add_argument('--edge_context_film_gamma_min', type=float, default=0.0)
    parser.add_argument('--edge_guided_upsample', action='store_true',
                        help='use edge to guide disparity upsampling for sharper boundaries')
    parser.add_argument('--edge_upsample_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated', 'mlp'])
    parser.add_argument('--edge_guided_disp_head', action='store_true')
    parser.add_argument('--edge_disp_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated', 'mlp'])
    parser.add_argument('--edge_guided_cost_agg', action='store_true')
    parser.add_argument('--edge_cost_agg_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated'])
    parser.add_argument('--edge_guided_gwc', action='store_true')
    parser.add_argument('--edge_gwc_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated'])
    parser.add_argument('--edge_motion_encoder', action='store_true')
    parser.add_argument('--edge_motion_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated'])
    parser.add_argument('--edge_guided_refinement', action='store_true')
    parser.add_argument('--boundary_only_refinement', action='store_true')
    parser.add_argument('--edge_refinement_fusion_mode', type=str, default='film',
                        choices=['concat', 'film', 'gated'])
    # Feature backbone: Edge-FiLM on x4 feature (left branch)
    parser.add_argument('--feature_edge_x4_film', action='store_true',
                        help='use edge-conditioned FiLM on x4 feature (left image only)')
    parser.add_argument('--feature_edge_x4_film_strength', type=float, default=1.0,
                        help='strength of FiLM modulation on x4 feature')
    # Edge-aware geo encoding volume (Combined_Geo_Encoding_Volume)
    parser.add_argument('--edge_geo_radius_aware', action='store_true',
                        help='use edge to adaptively shrink sampling radius in geo encoding volume')
    parser.add_argument('--edge_geo_radius_shrink', type=float, default=0.5,
                        help='lambda for shrinking geo sampling radius near edges (0=off, 0.5=moderate)')
    args = parser.parse_args()

    Path(args.output_directory).mkdir(exist_ok=True, parents=True)

    demo(args)
