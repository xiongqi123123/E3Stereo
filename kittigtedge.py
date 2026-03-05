# 预生成 KITTI GT edge 脚本。KITTI 视差为半稀疏（无效区 disp=0），
# 需先对无效区插值再取梯度，否则有效/无效边界会产生伪边缘。
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from core.utils import frame_utils
from gtedge import disp_to_edge


def inpaint_disp(disp, valid, radius=32):
    """对无效区做 inpainting，避免梯度在有效/无效边界产生伪边缘。"""
    disp = np.asarray(disp, np.float32)
    invalid = ~valid
    if not np.any(invalid):
        return disp
    mask = (invalid).astype(np.uint8) * 255
    disp_inpainted = cv2.inpaint(disp, mask, radius, cv2.INPAINT_TELEA)
    return disp_inpainted


def precompute_edges(dataset_root, split='training', disp_subdir='disp_occ', inpaint=True):
    """
    Args:
        dataset_root: 如 .../kitti_2012 或 .../kitti_2015
        split: 'training' 或 'testing'
        disp_subdir: KITTI 2012 用 'disp_occ'，KITTI 2015 用 'disp_occ_0'
        inpaint: 是否先对无效区插值再取 edge（推荐 True）
    """
    disp_dir = Path(dataset_root) / split / disp_subdir
    edge_dir = Path(dataset_root) / split / 'gtedge'
    edge_dir.mkdir(exist_ok=True)

    disp_files = sorted(disp_dir.glob('*_10.png'))
    for disp_path in tqdm(disp_files):
        disp, valid = frame_utils.readDispKITTI(str(disp_path))
        if inpaint:
            disp = inpaint_disp(disp, valid)
        edge = disp_to_edge(disp, grad_thresh=2.0, mode="laplacian")
        # 可选：在原始无效区将 edge 置 0，只保留有真实 disp 的区域的边缘
        if inpaint:
            edge[~valid] = 0
        edge_path = edge_dir / disp_path.name
        cv2.imwrite(str(edge_path), edge)

if __name__ == '__main__':
    # KITTI 2012: disp_occ
    precompute_edges(
        '/home/qi.xiong/StereoMatching/IGEV-Improve/data/kitti/kitti_2012',
        split='training', disp_subdir='disp_occ', inpaint=True,
    )
    # KITTI 2015: disp_occ_0
    precompute_edges(
        '/home/qi.xiong/StereoMatching/IGEV-Improve/data/kitti/kitti_2015',
        split='training', disp_subdir='disp_occ_0', inpaint=True,
    )