import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import re
import copy
import math
import random
from pathlib import Path
from glob import glob
import os.path as osp

from PIL import Image
import frame_utils


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None]


def load_disp(dispfile):
    if dispfile.endswith('.pfm'):
        disp, scale = readPFM(dispfile)
        disp = disp.copy()
    else:
        disp = np.array(Image.open(dispfile))

    disp = torch.from_numpy(disp).float()
    return disp[None]


class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None):
        self.aug_params = aug_params
        self.sparse = sparse
        self.img_list = []
        self.disparity_list = []

        if reader is None:
            self.reader = frame_utils.read_gen
        else:
            self.reader = reader

    def __getitem__(self, index):
        flow = None
        disp = None

        img1 = np.array(Image.open(self.img_list[index][0])).astype(np.uint8)
        img2 = np.array(Image.open(self.img_list[index][1])).astype(np.uint8)

        if self.disparity_list[index].endswith('.pfm'):
            disp, scale = readPFM(self.disparity_list[index])
            disp = disp.astype(np.float32)
        else:
            disp = np.array(Image.open(self.disparity_list[index])).astype(np.float32) / 256.0

        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.aug_params is not None:
            if self.sparse:
                img1, img2, disp, valid = self.augment_sparse(img1, img2, disp, self.aug_params)
            else:
                img1, img2, disp = self.augment_dense(img1, img2, disp, self.aug_params)
                valid = (disp < 512) & (disp > 0)  # 标准有效性检查
        else:
            # No augmentation: create valid mask from disparity
            valid = (disp < 512) & (disp > 0)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        disp = torch.from_numpy(disp)
        valid = torch.from_numpy(valid).float()

        # Always add channel dimension for consistency
        disp = disp[None]  # [H, W] -> [1, H, W]
        valid = valid[None]  # [H, W] -> [1, H, W]

        # Return file paths for validation, index for training
        img1_path, img2_path = self.img_list[index]
        disp_path = self.disparity_list[index]

        return (img1_path, img2_path, disp_path), img1, img2, disp, valid

    def augment_dense(self, img1, img2, disp, aug_params):
        """
        密集视差增强 (用于 SceneFlow)
        """
        h, w = img1.shape[:2]
        crop_h, crop_w = aug_params['crop_size']

        # ================= [FIX START] 自动 Padding 逻辑 =================
        # 如果图片比裁剪尺寸小，先进行 Padding，防止 random_crop 报错或输出尺寸不一致
        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)
        if pad_h > 0 or pad_w > 0:
            img1 = np.pad(img1, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            img2 = np.pad(img2, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            disp = np.pad(disp, ((0, pad_h), (0, pad_w)), mode='constant')
            # 更新宽高
            h, w = img1.shape[:2]
        # ================= [FIX END] =====================================

        # Random Crop
        y1 = random.randint(0, h - crop_h)
        x1 = random.randint(0, w - crop_w)

        img1 = img1[y1:y1 + crop_h, x1:x1 + crop_w]
        img2 = img2[y1:y1 + crop_h, x1:x1 + crop_w]
        disp = disp[y1:y1 + crop_h, x1:x1 + crop_w]

        # 颜色增强等其他操作...
        if random.random() < 0.5 and aug_params.get('do_flip', False):
            img1 = np.flipud(img1)
            img2 = np.flipud(img2)
            disp = np.flipud(disp)

        return img1, img2, disp

    def augment_sparse(self, img1, img2, disp, aug_params):
        """
        稀疏视差增强 (用于 KITTI, ETH3D, Middlebury)
        """
        h, w = img1.shape[:2]
        crop_h, crop_w = aug_params['crop_size']

        # ================= [FIX START] 自动 Padding 逻辑 =================
        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)
        if pad_h > 0 or pad_w > 0:
            img1 = np.pad(img1, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            img2 = np.pad(img2, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            disp = np.pad(disp, ((0, pad_h), (0, pad_w)), mode='constant')

        # 更新宽高
        h, w = img1.shape[:2]
        # ================= [FIX END] =====================================

        y1 = random.randint(0, h - crop_h)
        x1 = random.randint(0, w - crop_w)

        img1 = img1[y1:y1 + crop_h, x1:x1 + crop_w]
        img2 = img2[y1:y1 + crop_h, x1:x1 + crop_w]
        disp = disp[y1:y1 + crop_h, x1:x1 + crop_w]

        # Valid mask (GT > 0)
        valid = (disp > 0)

        return img1, img2, disp, valid

    def __len__(self):
        return len(self.img_list)


class SceneFlowDatasets(StereoDataset):
    def __init__(self, aug_params=None, root='/data/SceneFlow', dstype='frames_finalpass', things_test=False):
        super(SceneFlowDatasets, self).__init__(aug_params)
        self.root = root
        self.dstype = dstype

        if things_test:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa("TRAIN")
            self._add_driving("TRAIN")

    def _add_things(self, split='TRAIN'):
        """ 加载 FlyingThings3D """
        original_length = len(self.img_list)
        root = osp.join(self.root, 'FlyingThings3D')
        left_images = sorted(glob(osp.join(root, self.dstype, split, '*/*/left/*.png')))
        right_images = [im.replace('left', 'right') for im in left_images]
        disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.img_list.append((img1, img2))
            self.disparity_list.append(disp)
        logging.info(f"Added {len(self.img_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self, split='TRAIN'):
        """ 加载 Monkaa """
        original_length = len(self.img_list)
        root = osp.join(self.root, 'Monkaa')
        left_images = sorted(glob(osp.join(root, self.dstype, '*/*/left/*.png')))
        right_images = [im.replace('left', 'right') for im in left_images]
        disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.img_list.append((img1, img2))
            self.disparity_list.append(disp)
        logging.info(f"Added {len(self.img_list) - original_length} from Monkaa {self.dstype}")

    def _add_driving(self, split='TRAIN'):
        """ 加载 Driving """
        original_length = len(self.img_list)
        root = osp.join(self.root, 'Driving')
        left_images = sorted(glob(osp.join(root, self.dstype, '*/*/*/left/*.png')))
        right_images = [im.replace('left', 'right') for im in left_images]
        disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.img_list.append((img1, img2))
            self.disparity_list.append(disp)
        logging.info(f"Added {len(self.img_list) - original_length} from Driving {self.dstype}")


class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='/data/ETH3D', split='training'):
        super(ETH3D, self).__init__(aug_params, sparse=True)

        image1_list = sorted(glob(osp.join(root, 'two_view_training', '*', 'im0.png')))
        image2_list = sorted(glob(osp.join(root, 'two_view_training', '*', 'im1.png')))
        disp_list = sorted(glob(osp.join(root, 'two_view_training_gt', '*', 'disp0GT.pfm')))

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.img_list.append((img1, img2))
            self.disparity_list.append(disp)


class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='/data/Middlebury', split='training'):
        super(Middlebury, self).__init__(aug_params, sparse=True)
        # 支持 Middlebury 2014 标准结构
        image1_list = sorted(glob(osp.join(root, 'trainingH', '*', 'im0.png')))
        image2_list = sorted(glob(osp.join(root, 'trainingH', '*', 'im1.png')))
        disp_list = sorted(glob(osp.join(root, 'trainingH', '*', 'disp0GT.pfm')))

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.img_list.append((img1, img2))
            self.disparity_list.append(disp)


class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='/data/KITTI', split='training'):
        super(KITTI, self).__init__(aug_params, sparse=True)

        # KITTI 2015
        root15 = osp.join(root, 'KITTI_2015/training')
        image1_list = sorted(glob(osp.join(root15, 'image_2/*_10.png')))
        image2_list = sorted(glob(osp.join(root15, 'image_3/*_10.png')))
        disp_list = sorted(glob(osp.join(root15, 'disp_occ_0/*_10.png')))

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.img_list.append((img1, img2))
            self.disparity_list.append(disp)

        # KITTI 2012
        root12 = osp.join(root, 'KITTI_2012/training')
        image1_list = sorted(glob(osp.join(root12, 'colored_0/*_10.png')))
        image2_list = sorted(glob(osp.join(root12, 'colored_1/*_10.png')))
        disp_list = sorted(glob(osp.join(root12, 'disp_occ/*_10.png')))

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.img_list.append((img1, img2))
            self.disparity_list.append(disp)


def fetch_dataloader(args):
    """ 创建 DataLoader """

    # 定义数据增强参数 (包含 crop size)
    aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1],
                  'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    train_dataset = None

    # 根据参数混合数据集
    for dataset_name in args.train_datasets:
        if 'sceneflow' in dataset_name:
            new_dataset = SceneFlowDatasets(aug_params, root=args.sceneflow_root, dstype='frames_finalpass')
        elif 'kitti' in dataset_name:
            # 兼容 kitti, kitti15, kitti12 写法
            new_dataset = KITTI(aug_params, root=args.kitti_root)  # KITTI 类内部已经处理了 12+15 的加载
        elif 'middlebury' in dataset_name:
            new_dataset = Middlebury(aug_params, root=args.middlebury_root)
        elif 'eth3d' in dataset_name:
            new_dataset = ETH3D(aug_params, root=args.eth3d_root)

        if train_dataset is None:
            train_dataset = new_dataset
        else:
            train_dataset = data.ConcatDataset([train_dataset, new_dataset])

    logging.info(f"Training with {len(train_dataset)} image pairs")

    # 核心：num_workers 可以根据CPU情况调整，通常 4-8
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   pin_memory=True, shuffle=True, num_workers=4, drop_last=True)

    return train_loader