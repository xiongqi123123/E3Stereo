import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from IGEV.igev_stereo import IGEVStereo
from collections import namedtuple

# 模拟参数配置 (保持不变)
Args = namedtuple('Args', [
    'restore_ckpt', 'hidden_dims', 'corr_levels', 'corr_radius',
    'n_downsample', 'n_gru_layers', 'max_disp', 'mixed_precision', 'precision_dtype'
])


def visualize_features(ckpt_path, image_path, output_path='feature_heatmap_fixed.png'):
    # 1. 基础配置
    args = Args(
        restore_ckpt=ckpt_path,
        hidden_dims=[128] * 3,
        corr_levels=2,
        corr_radius=4,
        n_downsample=2,
        n_gru_layers=3,
        max_disp=192,
        mixed_precision=True,
        precision_dtype='float32'
    )

    # 2. 加载模型
    model = torch.nn.DataParallel(IGEVStereo(args))

    # 增加容错：如果没有 checkpoint，随机初始化也能看大概结构
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=True)
        print("成功加载模型权重。")
    except Exception as e:
        print(f"警告：加载权重失败 ({e})，将使用随机初始化权重进行演示。")

    model = model.module.cuda()
    model.eval()

    # 3. 读取并预处理图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图像 {image_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize 以匹配 IGEV 常见的输入倍数 (32的倍数)，防止 padding 引起的不对齐
    h, w = img.shape[:2]
    new_h = int(np.ceil(h / 32) * 32)
    new_w = int(np.ceil(w / 32) * 32)
    img_resized = cv2.resize(img_rgb, (new_w, new_h))

    image_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()[None].cuda()
    image_tensor = (image_tensor / 255.0) * 2 - 1.0

    # 4. 提取特征 (Hook)
    features = []

    def hook_fn(module, input, output):
        # IGEV cnet 返回结构: (outputs04, outputs08, outputs16)
        # outputs04 是一个 list: [tensor_net, tensor_inp, ...]
        # 修正点：取 output[0][1] 也就是 Context Feature (inp)
        # output[0][0] 是 Hidden State，通常是初始化的 0 或特定分布，不包含纹理
        feature_group = output[0]
        if isinstance(feature_group, list) and len(feature_group) > 1:
            context_feat = feature_group[1]  # 取 Context 分支
            features.append(context_feat.detach().cpu())
        else:
            print("警告：特征层结构不符合预期，尝试取第一个元素")
            features.append(feature_group[0].detach().cpu())

    handle = model.cnet.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model.cnet(image_tensor)

    handle.remove()

    if not features:
        print("错误：未提取到特征。")
        return

    # [C, H, W]
    feat = features[0][0]
    print(f"提取特征维度: {feat.shape}")

    # 5. 处理特征图 (核心修正)
    # 使用 L2 范数聚合通道，反应特征强度
    heatmap = torch.norm(feat, dim=0).numpy()

    # 鲁棒归一化：截断极端值 (Outliers)
    # 很多时候特征图中会有极少数特别大的值，导致整张图变黑
    low_bound = np.percentile(heatmap, 2)
    high_bound = np.percentile(heatmap, 98)
    heatmap = np.clip(heatmap, low_bound, high_bound)

    # 归一化到 0-1
    heatmap = (heatmap - low_bound) / (high_bound - low_bound + 1e-8)

    # 6. 生成可视化
    # 还原回原始图像大小
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # 叠加
    overlap = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    # 7. 保存与显示
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Context Feature (Fixed)")
    plt.imshow(heatmap_resized, cmap='jet')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(cv2.cvtColor(overlap, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"修复后的特征可视化已保存至: {output_path}")


if __name__ == '__main__':
    # 修改以下路径
    CKPT_PATH = '/root/autodl-tmp/stereo/model_cache/sceneflow.pth'
    # IMG_PATH = '/root/autodl-tmp/stereo/dataset_cache/KITTI/KITTI_2012/training/image_0/000192_10.png'
    # IMG_PATH = '/root/autodl-tmp/stereo/dataset_cache/KITTI/KITTI_2012/training/disp_occ/000192_10.png'
    IMG_PATH = '/root/autodl-tmp/stereo/dataset_cache/SceneFlow/FlyingThings3D/frames_finalpass/TRAIN/A/0130/left/0015.png'
    visualize_features(CKPT_PATH, IMG_PATH)