import argparse
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from IGEV.igev_stereo import IGEVStereo
from IGEV.utils import InputPadder


def normalize_feat(feat):
    """特征图归一化用于可视化"""
    # feat: [C, H, W] -> 聚合 -> [H, W]
    heatmap = torch.mean(torch.abs(feat), dim=0).cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


def inspect_sample(args):
    # 1. 加载单个模型
    print(f"正在加载模型: {args.ckpt}")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"模型文件不存在: {args.ckpt}")

    model = torch.nn.DataParallel(IGEVStereo(args))
    model.load_state_dict(torch.load(args.ckpt), strict=False)
    model = model.module.cuda().eval()
    print("模型加载完成！")

    # 2. 读取图片 (手动指定一张图片路径)
    left_path = args.left_img
    right_path = args.right_img

    print(f"读取左图: {left_path}")
    print(f"读取右图: {right_path}")

    if not os.path.exists(left_path):
        raise FileNotFoundError(f"左图不存在: {left_path}")
    if not os.path.exists(right_path):
        raise FileNotFoundError(f"右图不存在: {right_path}")

    image1 = cv2.imread(left_path)
    image2 = cv2.imread(right_path)

    if image1 is None:
        raise ValueError(f"无法读取左图: {left_path}")
    if image2 is None:
        raise ValueError(f"无法读取右图: {right_path}")

    print(f"图像尺寸: {image1.shape}")

    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()[None].cuda()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()[None].cuda()

    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)

    # 3. 注册 HOOKs (关键步骤)
    # 我们想看 Context Network 的输出
    features = {}

    def get_activation(name):
        def hook(model, input, output):
            # output[0] 是 hidden state, output[1] 是 context feature
            features[name] = output[0][1].detach()

        return hook

    # Hook 住 cnet
    model.cnet.register_forward_hook(get_activation('context_feat'))

    # 4. 运行模型并收集中间迭代结果
    print("正在运行模型推理...")
    disp_iters = []
    with torch.no_grad():
        # 这里我们需要稍微修改一下 forward 调用，或者直接调用 model 的内部组件
        # 为了简单，我们让模型正常跑，但是因为 IGEV forward 里会返回 (init_disp, disp_preds)
        # 所以我们需要正确解包这个元组
        init_disp, disp_preds = model(image1, image2, iters=32, test_mode=False)  # test_mode=False 会返回 (init_disp, disp_preds_list)

    print(f"模型完成推理，共 {len(disp_preds)} 次迭代")

    # 5. 可视化布局
    print("正在生成可视化...")
    plt.figure(figsize=(16, 12))

    # 提取模型名称（从路径中）
    model_name = os.path.basename(args.ckpt).replace('.pth', '')
    img_name = os.path.basename(args.left_img)
    plt.suptitle(f"Model Inspection: {model_name} | Image: {img_name}", fontsize=16)

    # (1) 原始图像
    img_vis = image1[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    plt.subplot(3, 3, 1)
    plt.title("Input Left Image")
    plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # (2) Context Feature Heatmap
    if 'context_feat' in features:
        c_feat = features['context_feat'][0]  # [128, H/4, W/4]
        c_heatmap = normalize_feat(c_feat)
        plt.subplot(3, 3, 2)
        plt.title("Context Feature (1/4 Res)")
        plt.imshow(c_heatmap, cmap='jet')
        plt.axis('off')

    # (3) 迭代过程展示
    # 挑选几个关键迭代步数：Init, Iter 5, Iter 15, Iter 32
    indices = [0, 5, 15, 31]
    plot_idx = 4
    for i in indices:
        if i < len(disp_preds):
            disp = disp_preds[i]
            disp = padder.unpad(disp).squeeze().cpu().numpy()  # 使用squeeze()去掉所有维度为1的维度

            plt.subplot(3, 3, plot_idx)
            plt.title(f"Disparity Iteration {i + 1}")
            plt.imshow(disp, cmap='magma', vmin=0, vmax=192)
            plt.axis('off')
            plot_idx += 1

    # (4) 最终预测的边缘响应 (Gradient of Disparity)
    # 这能帮你看到模型预测出来的边缘是不是锐利
    final_disp = disp_preds[-1]
    final_disp_np = padder.unpad(final_disp).squeeze().cpu().numpy()  # 使用squeeze()去掉所有维度为1的维度
    dy, dx = np.gradient(final_disp_np)
    edge_response = np.sqrt(dx ** 2 + dy ** 2)

    plt.subplot(3, 3, 3)
    plt.title("Pred Disp Gradients (Edges)")
    plt.imshow(np.clip(edge_response, 0, 5), cmap='gray_r')  # 反色，黑线为边
    plt.axis('off')

    # 保存到更清晰的路径
    save_dir = 'inspection_output'
    os.makedirs(save_dir, exist_ok=True)

    # 使用模型名和图像名构建保存文件名
    save_filename = f"{model_name}_{os.path.splitext(img_name)[0]}.png"
    save_path = os.path.join(save_dir, save_filename)

    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"\n深度可视化图已保存至: {save_path}")
    print(f"请查看 {save_dir} 目录")

    # 如果不在服务器环境，可以显示图像
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 模型选择 - 可以切换不同的checkpoint
    parser.add_argument('--ckpt', default='../logs/baseline_igev/1001_igev-stereo.pth')
    # parser.add_argument('--ckpt', default='../logs/ours_edge_weight/1001_igev-stereo.pth')  # Our方法
    # parser.add_argument('--ckpt', default='../model_cache/sceneflow.pth')  # 预训练模型

    # 测试样本路径 - 提供多个备选样本
    # 样本1: Monkaa - family_x2
    # parser.add_argument('--left_img',
    #     default='../dataset_cache/SceneFlow/Monkaa/frames_finalpass/family_x2/left/0160.png',
    #     help="Path to a left image")
    # parser.add_argument('--right_img',
    #     default='../dataset_cache/SceneFlow/Monkaa/frames_finalpass/family_x2/right/0160.png',
    #     help="Path to a right image")

    # 备选样本2: Monkaa - eating_camera2_x2
    # parser.add_argument('--left_img', default='../dataset_cache/SceneFlow/Monkaa/frames_finalpass/eating_camera2_x2/left/0050.png')
    # parser.add_argument('--right_img', default='../dataset_cache/SceneFlow/Monkaa/frames_finalpass/eating_camera2_x2/right/0050.png')

    # 备选样本3: Monkaa - treeflight_x2
    # parser.add_argument('--left_img', default='../dataset_cache/SceneFlow/Monkaa/frames_finalpass/treeflight_x2/left/0100.png')
    # parser.add_argument('--right_img', default='../dataset_cache/SceneFlow/Monkaa/frames_finalpass/treeflight_x2/right/0100.png')

    # 备选样本4: FlyingThings3D (需要调整路径到具体的TEST样本)
    # parser.add_argument('--left_img', default='../dataset_cache/SceneFlow/FlyingThings3D/frames_finalpass/TEST/A/0000/left/0006.png')
    # parser.add_argument('--right_img', default='../dataset_cache/SceneFlow/FlyingThings3D/frames_finalpass/TEST/A/0000/right/0006.png')

    # 备选样本5: Driving - different scenes
    parser.add_argument('--left_img', default='../dataset_cache/SceneFlow/Driving/frames_finalpass/35mm_focallength/scene_backwards/fast/left/0001.png')
    parser.add_argument('--right_img', default='../dataset_cache/SceneFlow/Driving/frames_finalpass/35mm_focallength/scene_backwards/fast/right/0001.png')
    # IGEV Default Params
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3)
    parser.add_argument('--corr_levels', type=int, default=2)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--n_gru_layers', type=int, default=3)
    parser.add_argument('--max_disp', type=int, default=192)
    parser.add_argument('--mixed_precision', action='store_true', default=True)
    parser.add_argument('--precision_dtype', default='float16')

    args = parser.parse_args()
    inspect_sample(args)