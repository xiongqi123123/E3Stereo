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
    if len(feat.shape) == 3:  # [C, H, W]
        heatmap = torch.mean(torch.abs(feat), dim=0).cpu().numpy()
    else:  # [H, W]
        heatmap = feat.cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


def compute_error_map(disp_pred, disp_gt):
    """计算视差误差图"""
    error = torch.abs(disp_pred - disp_gt)
    return error


def inspect_sample(args):
    # 1. 加载模型
    print(f"正在加载模型: {args.ckpt}")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"模型文件不存在: {args.ckpt}")

    model = torch.nn.DataParallel(IGEVStereo(args))
    model.load_state_dict(torch.load(args.ckpt), strict=False)
    model = model.module.cuda().eval()
    print("模型加载完成！")

    # 2. 读取图片
    left_path = args.left_img
    right_path = args.right_img
    gt_path = args.gt_disp

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

    image1_t = torch.from_numpy(image1).permute(2, 0, 1).float()[None].cuda()
    image2_t = torch.from_numpy(image2).permute(2, 0, 1).float()[None].cuda()

    padder = InputPadder(image1_t.shape, divis_by=32)
    image1_t, image2_t = padder.pad(image1_t, image2_t)

    # 加载GT视差
    disp_gt = None
    if gt_path and os.path.exists(gt_path):
        print(f"读取GT视差: {gt_path}")
        from IGEV.frame_utils import readPFM
        disp_gt = readPFM(gt_path)
        disp_gt = torch.from_numpy(disp_gt.copy()).unsqueeze(0).unsqueeze(0).cuda()
        disp_gt, _ = padder.pad(disp_gt, disp_gt)
    else:
        print("未提供GT视差，将跳过error map可视化")

    # 3. 手动运行模型的forward，记录中间结果
    print("正在运行模型推理...")

    features = {}

    with torch.no_grad():
        image1_norm = (2 * (image1_t / 255.0) - 1.0).contiguous()
        image2_norm = (2 * (image2_t / 255.0) - 1.0).contiguous()

        # Feature extraction
        features_left = model.feature(image1_norm)
        features_right = model.feature(image2_norm)

        # Store feature network output
        features['feature_left'] = features_left[0].detach()

        # Stem processing
        stem_2x = model.stem_2(image1_norm)
        stem_4x = model.stem_4(stem_2x)
        stem_2y = model.stem_2(image2_norm)
        stem_4y = model.stem_4(stem_2y)
        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        # Matching features
        match_left = model.desc(model.conv(features_left[0]))
        match_right = model.desc(model.conv(features_right[0]))

        # Build cost volume
        from IGEV.submodule import build_gwc_volume, disparity_regression
        gwc_volume = build_gwc_volume(match_left, match_right, args.max_disp // 4, 8)
        gwc_volume = model.corr_stem(gwc_volume)
        gwc_volume = model.corr_feature_att(gwc_volume, features_left[0])

        # Cost aggregation
        geo_encoding_volume = model.cost_agg(gwc_volume, features_left)
        features['geo_encoding'] = geo_encoding_volume.detach()

        # Initial disparity
        prob = torch.softmax(model.classifier(geo_encoding_volume).squeeze(1), dim=1)
        init_disp = disparity_regression(prob, args.max_disp // 4)
        features['init_disp'] = init_disp.detach()

        # Context network
        cnet_list = model.cnet(image1_norm, num_layers=args.n_gru_layers)
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        features['context_hidden'] = net_list[0].detach()
        features['context_input'] = inp_list[0].detach()

        inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in
                    zip(inp_list, model.context_zqr_convs)]

        # GRU iterations
        from IGEV.geometry import Combined_Geo_Encoding_Volume
        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(match_left.float(), match_right.float(), geo_encoding_volume.float(),
                           radius=args.corr_radius, num_levels=args.corr_levels)

        b, c, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1, 1, w, 1).repeat(b, h, 1, 1)
        disp = init_disp

        # 只保存需要可视化的迭代步骤
        save_iters = [0, 5, 10, 15, 20, 25, args.n_iters - 1]
        disp_preds = {}
        geo_feats = {}

        # 迭代更新
        for itr in range(args.n_iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords)

            if itr == 0:
                features['geo_feat_input'] = geo_feat.detach()

            net_list, mask_feat_4, delta_disp = model.update_block(net_list, inp_list, geo_feat, disp,
                                                                  iter16=args.n_gru_layers == 3,
                                                                  iter08=args.n_gru_layers >= 2)

            disp = disp + delta_disp

            # 只在需要的步骤保存结果
            if itr in save_iters:
                disp_up = model.upsample_disp(disp, mask_feat_4, stem_2x)
                disp_preds[itr] = disp_up.detach()
                geo_feats[itr] = geo_feat.detach()

        # 释放不需要的大对象
        del geo_encoding_volume, gwc_volume, match_left, match_right, prob
        torch.cuda.empty_cache()

    print(f"模型完成推理，保存了 {len(disp_preds)} 个关键迭代步骤")

    # 4. 可视化
    print("正在生成可视化...")

    # 创建大图布局
    fig = plt.figure(figsize=(24, 18))
    model_name = os.path.basename(args.ckpt).replace('.pth', '')
    img_name = os.path.basename(args.left_img)
    fig.suptitle(f"IGEV Internal Visualization | Model: {model_name} | Image: {img_name}", fontsize=18)

    plot_idx = 1

    # Row 1: Input & Networks Output
    # (1) Input Image
    img_vis = image1_t[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    plt.subplot(5, 6, plot_idx)
    plt.title("(1) Input Left Image")
    plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plot_idx += 1

    # (2) Feature Network Output
    feat_left = features['feature_left'][0]  # [C, H/4, W/4]
    feat_heatmap = normalize_feat(feat_left)
    plt.subplot(5, 6, plot_idx)
    plt.title("(2) Feature Network Output\n(1/4 Res)")
    plt.imshow(feat_heatmap, cmap='jet')
    plt.axis('off')
    plot_idx += 1

    # (3) Context Network - Hidden
    ctx_hidden = features['context_hidden'][0]  # [128, H/4, W/4]
    ctx_heatmap = normalize_feat(ctx_hidden)
    plt.subplot(5, 6, plot_idx)
    plt.title("(3) Context Network\nHidden State")
    plt.imshow(ctx_heatmap, cmap='jet')
    plt.axis('off')
    plot_idx += 1

    # (4) Context Network - Input
    ctx_input = features['context_input'][0]  # [128, H/4, W/4]
    ctx_inp_heatmap = normalize_feat(ctx_input)
    plt.subplot(5, 6, plot_idx)
    plt.title("(4) Context Network\nInput Feature")
    plt.imshow(ctx_inp_heatmap, cmap='jet')
    plt.axis('off')
    plot_idx += 1

    # (5) Geo Encoding Volume
    geo_enc = features['geo_encoding'][0]  # [8, D, H, W]
    geo_enc_mean = torch.mean(geo_enc, dim=(0, 1)).cpu().numpy()  # [H, W]
    geo_enc_mean = (geo_enc_mean - geo_enc_mean.min()) / (geo_enc_mean.max() - geo_enc_mean.min() + 1e-8)
    plt.subplot(5, 6, plot_idx)
    plt.title("(5) Geo Encoding Volume\n(Cost Aggregation Output)")
    plt.imshow(geo_enc_mean, cmap='viridis')
    plt.axis('off')
    plot_idx += 1

    # (6) Initial Disparity
    init_disp_vis = features['init_disp'][0, 0].cpu().numpy()
    plt.subplot(5, 6, plot_idx)
    plt.title("(6) Initial Disparity\n(Before GRU)")
    im = plt.imshow(init_disp_vis, cmap='magma', vmin=0, vmax=args.max_disp // 4)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis('off')
    plot_idx += 1

    # Row 2: CGEV Input & Initial Error
    # (7) CGEV Output (Geo Feat) - Input to ConvGRU
    geo_feat_inp = features['geo_feat_input'][0]  # [C, H, W]
    geo_feat_heatmap = normalize_feat(geo_feat_inp)
    plt.subplot(5, 6, plot_idx)
    plt.title("(7) CGEV Output\n(Input to ConvGRU)")
    plt.imshow(geo_feat_heatmap, cmap='plasma')
    plt.axis('off')
    plot_idx += 1

    # (8) Initial Disparity Upsampled
    if disp_gt is not None:
        init_disp_up = torch.nn.functional.interpolate(
            features['init_disp'].unsqueeze(1) * 4,
            size=disp_gt.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        plt.subplot(5, 6, plot_idx)
        plt.title("(8) Initial Disp\n(Upsampled)")
        im = plt.imshow(init_disp_up[0, 0].cpu().numpy(), cmap='magma', vmin=0, vmax=192)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.axis('off')
        plot_idx += 1

        # (9) Initial Error Map
        init_error = compute_error_map(init_disp_up, disp_gt)[0, 0].cpu().numpy()
        plt.subplot(5, 6, plot_idx)
        plt.title("(9) Initial Disp\nError Map")
        im = plt.imshow(init_error, cmap='hot', vmin=0, vmax=5)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.axis('off')
        plot_idx += 1
    else:
        plot_idx += 2

    # Row 3-5: GRU Iteration Results
    # 从保存的迭代步骤中选择展示
    for idx in sorted(disp_preds.keys()):
        if plot_idx > 30:
            break

        disp_pred = disp_preds[idx]
        disp_pred_np = padder.unpad(disp_pred).squeeze().cpu().numpy()

        # Disparity Prediction
        plt.subplot(5, 6, plot_idx)
        plt.title(f"Iter {idx+1}: Disparity")
        im = plt.imshow(disp_pred_np, cmap='magma', vmin=0, vmax=192)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.axis('off')
        plot_idx += 1

        # Error Map
        if disp_gt is not None:
            error_map = compute_error_map(disp_pred, disp_gt)[0, 0].cpu().numpy()
            error_map_unpad = padder.unpad(torch.from_numpy(error_map).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
            plt.subplot(5, 6, plot_idx)
            plt.title(f"Iter {idx+1}: Error Map")
            im = plt.imshow(error_map_unpad, cmap='hot', vmin=0, vmax=5)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.axis('off')
        else:
            plt.subplot(5, 6, plot_idx)
            plt.title(f"Iter {idx+1}: Error\n(No GT)")
            plt.axis('off')
        plot_idx += 1

    # 保存
    save_dir = 'inspection_output'
    os.makedirs(save_dir, exist_ok=True)
    save_filename = f"{model_name}_{os.path.splitext(img_name)[0]}_detailed.png"
    save_path = os.path.join(save_dir, save_filename)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"\n可视化图已保存至: {save_path}")
    print(f"请查看 {save_dir} 目录")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 模型checkpoint
    parser.add_argument('--ckpt',
                        default='../model_cache/sceneflow.pth')

    # 测试图像路径
    parser.add_argument('--left_img',
                        default='../dataset_cache/SceneFlow/Driving/frames_finalpass/35mm_focallength/scene_backwards/fast/left/0001.png')
    parser.add_argument('--right_img',
                        default='../dataset_cache/SceneFlow/Driving/frames_finalpass/35mm_focallength/scene_backwards/fast/right/0001.png')

    # GT视差路径（可选）
    parser.add_argument('--gt_disp',
                        default='../dataset_cache/SceneFlow/Driving/disparity/35mm_focallength/scene_backwards/fast/left/0001.pfm',
                        help="Ground truth disparity path (optional)")

    # IGEV 模型参数
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3)
    parser.add_argument('--corr_levels', type=int, default=2)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--n_gru_layers', type=int, default=3)
    parser.add_argument('--max_disp', type=int, default=192)
    parser.add_argument('--mixed_precision', action='store_true', default=True)
    parser.add_argument('--precision_dtype', default='bfloat16')
    parser.add_argument('--n_iters', type=int, default=32, help="Number of GRU iterations")

    args = parser.parse_args()
    inspect_sample(args)
