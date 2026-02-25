"""
几何边缘训练数据集：从 SceneFlow 加载左图 + GT 几何边缘（由 disparity 梯度生成）。
"""
import os
import os.path as osp
import random
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import cv2
from glob import glob

from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor


class SceneFlowEdgeDataset(data.Dataset):
    """
    SceneFlow 几何边缘数据集：左图 + GT edge（gtedge 目录）。
    数据增强与 stereo 一致，对 img 和 edge 同步施加空间变换。
    """
    def __init__(
        self,
        root="/home/qi.xiong/StereoMatching/IGEV-Improve/data/sceneflow",
        dstype="frames_finalpass",
        aug_params=None,
        split="TRAIN",
    ):
        self.root = root
        self.dstype = dstype
        self.split = split
        self.augmentor = None
        self.img_pad = aug_params.pop("img_pad", None) if aug_params else None
        self.fixed_size = aug_params.pop("fixed_size", None) if aug_params else None

        if aug_params and "crop_size" in aug_params and self.fixed_size is None:
            self.augmentor = FlowAugmentor(**aug_params)

        self.image_list = []
        self.edge_list = []
        self._collect_files()

    def _collect_files(self):
        """收集 (left_image, edge) 路径对"""
        # 与 SceneFlowDatasets 一致的路径逻辑
        left_images = sorted(
            glob(osp.join(self.root, self.dstype, self.split, "*/*/left/*.png"))
        )
        left_images += sorted(
            glob(osp.join(self.root, self.dstype, self.split, "*/left/*.png"))
        )
        left_images += sorted(
            glob(osp.join(self.root, self.dstype, self.split, "*/*/*/left/*.png"))
        )
        left_images = sorted(set(left_images))

        for left_path in left_images:
            # edge: frames_finalpass -> gtedge，路径结构一致
            edge_path = left_path.replace(self.dstype, "gtedge")
            if osp.exists(edge_path):
                self.image_list.append(left_path)
                self.edge_list.append(edge_path)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        edge_path = self.edge_list[index]

        img = np.array(frame_utils.read_gen(img_path)).astype(np.uint8)
        if len(img.shape) == 2:
            img = np.tile(img[..., None], (1, 1, 3))
        else:
            img = img[..., :3]

        edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        if edge is None:
            edge = np.zeros(img.shape[:2], dtype=np.uint8)
        edge = edge.astype(np.float32) / 255.0  # [0, 1]

        # 确保尺寸一致
        if edge.shape != img.shape[:2]:
            edge = cv2.resize(edge, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Eval 模式：固定尺寸，无增强
        if self.fixed_size is not None:
            h, w = self.fixed_size[0], self.fixed_size[1]
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            edge = cv2.resize(edge, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            # 数据增强：需要 img + edge 同步变换，使用假的 flow（仅用于 augmentor 的 crop）
            flow = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float32)
            if self.augmentor is not None:
                img, _, flow, edge = self.augmentor(img, img.copy(), flow, edge)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        edge = torch.from_numpy(edge).unsqueeze(0).float()  # [1, H, W]

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img = F.pad(img, [padW] * 2 + [padH] * 2)
            edge = F.pad(edge, [padW] * 2 + [padH] * 2)

        return img, edge


def fetch_edge_dataloader(args):
    """创建几何边缘训练的 DataLoader"""
    aug_params = {
        "crop_size": args.image_size,
        "min_scale": args.spatial_scale[0],
        "max_scale": args.spatial_scale[1],
        "do_flip": getattr(args, "edge_do_flip", "hf"),  # 'hf': 仅水平翻转，不交换左右
        "yjitter": not getattr(args, "noyjitter", False),
    }
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma

    dataset = SceneFlowEdgeDataset(
        root=getattr(args, "data_root", "/home/qi.xiong/StereoMatching/IGEV-Improve/data/sceneflow"),
        dstype="frames_finalpass",
        aug_params=aug_params,
        split="TRAIN",
    )

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 6)) - 2
    num_workers = max(1, num_workers)

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def fetch_edge_eval_dataloader(args, max_samples=None):
    """创建无增强的 eval DataLoader，用于 ODS/OIS 评估。"""
    # 无 crop，使用 fixed_size 得到固定分辨率
    image_size = getattr(args, "image_size", [320, 736])
    if isinstance(image_size, (list, tuple)):
        fixed_size = (image_size[0], image_size[1])
    else:
        fixed_size = (image_size, image_size)

    aug_params = {"fixed_size": fixed_size}

    dataset = SceneFlowEdgeDataset(
        root=getattr(args, "data_root", "/home/qi.xiong/StereoMatching/IGEV-Improve/data/sceneflow"),
        dstype="frames_finalpass",
        aug_params=aug_params,
        split="TRAIN",
    )

    if max_samples is not None and len(dataset) > max_samples:
        dataset = data.Subset(dataset, range(max_samples))

    loader = data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return loader
