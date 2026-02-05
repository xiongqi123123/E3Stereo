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
import cv2

from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None, edge_source='rcf'):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader        

        self.edge_source = edge_source  # 'rcf' or 'gt'
        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        disp = self.disparity_reader(self.disparity_list[index])
        
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 512

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        disp = np.array(disp).astype(np.float32)

        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        # 加载 GT edge（如果需要）
        edge = None
        if self.edge_source == 'gt':
            disp_path = self.disparity_list[index]
            edge_path = disp_path.replace('/disparity/', '/gtedge/').replace('.pfm', '.png')
            if osp.exists(edge_path):
                edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
                if edge is not None:
                    edge = edge.astype(np.float32) / 255.0  # [H, W] in [0, 1]
                    # 确保 edge 和 img1 尺寸一致（数据增强前）
                    h_img, w_img = img1.shape[:2]
                    h_edge, w_edge = edge.shape[:2]
                    if (h_edge, w_edge) != (h_img, w_img):
                        edge = cv2.resize(edge, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
                else:
                    logging.warning(f"Failed to read edge map: {edge_path}, using None")
            else:
                logging.warning(f"Edge map not found: {edge_path}, using None")

        if self.augmentor is not None:
            if self.sparse:
                # 传递 edge 给 SparseFlowAugmentor，让它同步应用空间变换
                if edge is not None:
                    img1, img2, flow, valid, edge = self.augmentor(img1, img2, flow, valid, edge)
                else:
                    img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                # 传递 edge 给 FlowAugmentor，让它同步应用空间变换（scale/flip/crop）
                if edge is not None:
                    img1, img2, flow, edge = self.augmentor(img1, img2, flow, edge)
                else:
                    img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 512) & (flow[1].abs() < 512)

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)
            if edge is not None:
                edge = F.pad(edge, [padW]*2 + [padH]*2)

        flow = flow[:1]
        
        # 将 edge 转换为 tensor
        if edge is not None:
            edge = torch.from_numpy(edge).unsqueeze(0).float()  # [1, H, W]
        
        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float(), edge


    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self
        
    def __len__(self):
        return len(self.image_list)


class SceneFlowDatasets(StereoDataset):
    def __init__(self, aug_params=None, root='/home/qi.xiong/StereoMatching/IGEV-Improve/data/sceneflow/', dstype='frames_finalpass', things_test=False, edge_source='rcf'):
        super(SceneFlowDatasets, self).__init__(aug_params, edge_source=edge_source)
        self.root = root
        self.dstype = dstype

        if things_test:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa("TRAIN")
            self._add_driving("TRAIN")

    def _add_things(self, split='TRAIN'):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        # root = osp.join(self.root, 'FlyingThings3D')
        root = self.root
        left_images = sorted( glob(osp.join(root, self.dstype, split, '*/*/left/*.png')) )
        right_images = [ im.replace('left', 'right') for im in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        state = np.random.get_state()
        np.random.seed(1000)
        # val_idxs = set(np.random.permutation(len(left_images))[:100])
        val_idxs = set(np.random.permutation(len(left_images)))
        np.random.set_state(state)

        for idx, (img1, img2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
            if (split == 'TEST' and idx in val_idxs) or split == 'TRAIN':
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self, split="TRAIN"):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = self.root
        left_images = sorted( glob(osp.join(root, self.dstype, split, '*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")


    def _add_driving(self, split="TRAIN"):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = self.root
        left_images = sorted( glob(osp.join(root, self.dstype, split, '*/*/*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")


class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='/data/ETH3D', split='training', edge_source='rcf'):
        super(ETH3D, self).__init__(aug_params, sparse=True, edge_source=edge_source)

        image1_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im0.png')) )
        image2_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im1.png')) )
        disp_list = sorted( glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm')) ) if split == 'training' else [osp.join(root, 'two_view_training_gt/playground_1l/disp0GT.pfm')]*len(image1_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class SintelStereo(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/SintelStereo', edge_source='rcf'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispSintelStereo, edge_source=edge_source)

        image1_list = sorted( glob(osp.join(root, 'training/*_left/*/frame_*.png')) )
        image2_list = sorted( glob(osp.join(root, 'training/*_right/*/frame_*.png')) )
        disp_list = sorted( glob(osp.join(root, 'training/disparities/*/frame_*.png')) ) * 2

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            assert img1.split('/')[-2:] == disp.split('/')[-2:]
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class FallingThings(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/FallingThings', edge_source='rcf'):
        super().__init__(aug_params, reader=frame_utils.readDispFallingThings, edge_source=edge_source)
        assert os.path.exists(root)

        with open(os.path.join(root, 'filenames.txt'), 'r') as f:
            filenames = sorted(f.read().splitlines())

        image1_list = [osp.join(root, e) for e in filenames]
        image2_list = [osp.join(root, e.replace('left.jpg', 'right.jpg')) for e in filenames]
        disp_list = [osp.join(root, e.replace('left.jpg', 'left.depth.png')) for e in filenames]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class TartanAir(StereoDataset):
    def __init__(self, aug_params=None, root='datasets', keywords=[], edge_source='rcf'):
        super().__init__(aug_params, reader=frame_utils.readDispTartanAir, edge_source=edge_source)
        assert os.path.exists(root)

        with open(os.path.join(root, 'tartanair_filenames.txt'), 'r') as f:
            filenames = sorted(list(filter(lambda s: 'seasonsforest_winter/Easy' not in s, f.read().splitlines())))
            for kw in keywords:
                filenames = sorted(list(filter(lambda s: kw in s.lower(), filenames)))

        image1_list = [osp.join(root, e) for e in filenames]
        image2_list = [osp.join(root, e.replace('_left', '_right')) for e in filenames]
        disp_list = [osp.join(root, e.replace('image_left', 'depth_left').replace('left.png', 'left_depth.npy')) for e in filenames]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='/data/KITTI/KITTI_2015', image_set='training', edge_source='rcf'):
        super(KITTI, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI, edge_source=edge_source)
        assert os.path.exists(root)

        root_12 = '/data/KITTI/KITTI_2012'
        image1_list = sorted(glob(os.path.join(root_12, image_set, 'colored_0/*_10.png')))
        image2_list = sorted(glob(os.path.join(root_12, image_set, 'colored_1/*_10.png')))
        disp_list = sorted(glob(os.path.join(root_12, 'training', 'disp_occ/*_10.png'))) if image_set == 'training' else [osp.join(root, 'training/disp_occ/000085_10.png')]*len(image1_list)

        root_15 = '/data/KITTI/KITTI_2015'
        image1_list += sorted(glob(os.path.join(root_15, image_set, 'image_2/*_10.png')))
        image2_list += sorted(glob(os.path.join(root_15, image_set, 'image_3/*_10.png')))
        disp_list += sorted(glob(os.path.join(root_15, 'training', 'disp_occ_0/*_10.png'))) if image_set == 'training' else [osp.join(root, 'training/disp_occ_0/000085_10.png')]*len(image1_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]


class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='/data/Middlebury', split='F', edge_source='rcf'):
        super(Middlebury, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispMiddlebury, edge_source=edge_source)
        assert os.path.exists(root)
        assert split in "FHQ"
        lines = list(map(osp.basename, glob(os.path.join(root, "trainingH/*"))))
        # lines = list(filter(lambda p: any(s in p.split('/') for s in Path(os.path.join(root, "MiddEval3/official_train.txt")).read_text().splitlines()), lines))
        # image1_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/im0.png') for name in lines])
        # image2_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/im1.png') for name in lines])
        # disp_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/disp0GT.pfm') for name in lines])
        image1_list = sorted([os.path.join(root, f'training{split}', f'{name}/im0.png') for name in lines])
        image2_list = sorted([os.path.join(root, f'training{split}', f'{name}/im1.png') for name in lines])
        disp_list = sorted([os.path.join(root, f'training{split}', f'{name}/disp0GT.pfm') for name in lines])

        assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

  
def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    # 获取 edge_source 参数
    edge_source = getattr(args, 'edge_source', 'rcf')

    train_dataset = None
    for dataset_name in args.train_datasets:
        if re.compile("middlebury_.*").fullmatch(dataset_name):
            new_dataset = Middlebury(aug_params, split=dataset_name.replace('middlebury_',''), edge_source=edge_source)
        elif dataset_name == 'sceneflow':
            #clean_dataset = SceneFlowDatasets(aug_params, dstype='frames_cleanpass')
            final_dataset = SceneFlowDatasets(aug_params, dstype='frames_finalpass', edge_source=edge_source)
            #new_dataset = (clean_dataset*4) + (final_dataset*4)
            new_dataset = final_dataset
            logging.info(f"Adding {len(new_dataset)} samples from SceneFlow")
        elif 'kitti' in dataset_name:
            new_dataset = KITTI(aug_params, edge_source=edge_source)
            logging.info(f"Adding {len(new_dataset)} samples from KITTI")
        elif dataset_name == 'sintel_stereo':
            new_dataset = SintelStereo(aug_params, edge_source=edge_source)*140
            logging.info(f"Adding {len(new_dataset)} samples from Sintel Stereo")
        elif dataset_name == 'falling_things':
            new_dataset = FallingThings(aug_params, edge_source=edge_source)*5
            logging.info(f"Adding {len(new_dataset)} samples from FallingThings")
        elif dataset_name.startswith('tartan_air'):
            new_dataset = TartanAir(aug_params, keywords=dataset_name.split('_')[2:], edge_source=edge_source)
            logging.info(f"Adding {len(new_dataset)} samples from Tartain Air")
        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=True, shuffle=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, drop_last=True)

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader

