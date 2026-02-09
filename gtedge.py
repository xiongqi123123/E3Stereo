import os
import os.path as osp
from glob import glob
import argparse

import cv2
import numpy as np
from tqdm import tqdm


def read_pfm(file):
    """Minimal PFM reader compatible with SceneFlow disparity (.pfm)."""
    import re

    with open(file, "rb") as f:
        header = f.readline().rstrip().decode("ascii")
        if header not in ("PF", "Pf"):
            raise ValueError("Not a PFM file: %s" % file)

        dim_line = f.readline().decode("ascii")
        while dim_line.startswith("#"):
            dim_line = f.readline().decode("ascii")
        width, height = map(int, re.findall(r"\d+", dim_line))

        scale = float(f.readline().rstrip().decode("ascii"))
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")
        data = np.reshape(data, (height, width, 3) if header == "PF" else (height, width))
        data = np.flipud(data)
    return data


def _grad_mag(disp, mode="sobel", blur_ksize=5, blur_sigma=1.0):
    """
    计算 disparity 的梯度幅值:
    - mode == "sobel"      : 直接对原始 disp 做 Sobel
    - mode == "blur_sobel" : 先高斯模糊，再做 Sobel（抑制细碎高频，如树叶）
    """
    disp = np.asarray(disp, np.float32)

    # 无效值置零，避免产生伪边缘
    invalid = ~np.isfinite(disp)
    disp[invalid] = 0.0

    # 可选：先做高斯模糊，去掉高频噪声/细碎纹理
    if mode in ("blur_sobel", "blur_laplacian"):
        # 高斯模糊，仅保留大尺度结构
        k = int(blur_ksize)
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1  # kernel size 必须为奇数
        disp = cv2.GaussianBlur(disp, (k, k), sigmaX=float(blur_sigma), sigmaY=float(blur_sigma))

    if mode in ("sobel", "blur_sobel"):
        # 一阶梯度（Slope）：平面斜坡会是常数非零 → 容易整块发白
        gx = cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(disp, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
    elif mode in ("laplacian", "blur_laplacian", "laplacian_close"):
        # 二阶梯度（Curvature）：平面斜坡二阶导接近0，只在曲率/跳变处响应
        lap = cv2.Laplacian(disp, cv2.CV_32F, ksize=3)
        mag = np.abs(lap)
    else:
        raise ValueError(f"Unknown mode for _grad_mag: {mode}")

    return mag


def disp_to_edge(
    disp,
    grad_thresh=0.5,
    mode="sobel",
    blur_ksize=5,
    blur_sigma=1.0,
    canny_low=None,
    canny_high=None,
):
    """
    根据 disparity 的梯度生成几何边缘:
    - mode="sobel"              : 直接 Sobel，一阶绝对梯度
    - mode="sobel_rel"          : Sobel + 相对梯度阈值：|∇d| / (|d|+eps)，对大平面更友好
    - mode="blur_sobel"         : 先模糊，再 Sobel，只保留大尺度几何边
    - mode="laplacian"          : 直接二阶导（Laplacian），对平面斜坡更不敏感
    - mode="blur_laplacian"     : 先模糊，再 Laplacian，更关注大尺度曲率/跳变
    - mode="laplacian_close"    : 直接二阶导 + 形态学闭运算（先膨胀再腐蚀），可加粗并填补空洞
    - mode="canny"              : 对 disparity 做归一化 + Canny 边缘检测
    - 根据梯度幅值阈值 grad_thresh 生成二值边缘
    """
    disp_arr = np.asarray(disp, np.float32)

    # 专门的 Canny 分支（不走 _grad_mag）
    if mode == "canny":
        # 处理无效值
        d = np.copy(disp_arr)
        invalid = ~np.isfinite(d)
        d[invalid] = 0.0

        # 基于分位数做鲁棒归一化到 [0, 255]
        vmin, vmax = np.percentile(d, 1), np.percentile(d, 99)
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
        d = np.clip(d, vmin, vmax)
        d = (d - vmin) / (vmax - vmin) * 255.0
        d8 = d.astype(np.uint8)

        # 可选模糊，减少噪声
        if blur_ksize and blur_ksize > 1:
            k = int(blur_ksize)
            if k % 2 == 0:
                k += 1
            d8 = cv2.GaussianBlur(d8, (k, k), sigmaX=float(blur_sigma), sigmaY=float(blur_sigma))

        # Canny 阈值：如果没显式给，就用一个相对保守的默认
        if canny_low is None and canny_high is None:
            low = 50
            high = 150
        else:
            low = canny_low if canny_low is not None else 50
            high = canny_high if canny_high is not None else low * 3.0

        edge = cv2.Canny(d8, threshold1=float(low), threshold2=float(high))
        return edge

    # 映射到基础梯度模式
    base_mode = mode
    if mode == "laplacian_close":
        base_mode = "laplacian"
    if mode == "sobel_rel":
        base_mode = "sobel"

    mag = _grad_mag(disp_arr, mode=base_mode, blur_ksize=blur_ksize, blur_sigma=blur_sigma)

    # 相对梯度阈值：|∇d| / (|d| + eps)
    if mode == "sobel_rel":
        denom = np.abs(disp_arr)
        eps = 1e-3
        denom[denom < eps] = eps
        mag = mag / denom
    edge = (mag >= grad_thresh).astype(np.uint8)

    if mode == "laplacian_close":
        # 形态学闭运算：先膨胀再腐蚀，填补缝隙、加粗细边
        k = int(blur_ksize)
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edge = cv2.dilate(edge, kernel, iterations=1)
        edge = cv2.erode(edge, kernel, iterations=1)

    edge = (edge > 0).astype(np.uint8) * 255
    return edge


def process_sceneflow_disp(
    root_disp="/home/qi.xiong/StereoMatching/IGEV-Improve/data/sceneflow/disparity",
    root_edge="/home/qi.xiong/StereoMatching/IGEV-Improve/data/sceneflow/ggtedge",
    split_list=("TRAIN", "TEST"),
    grad_thresh=2.5,
    mode="sobel",
    blur_ksize=5,
    blur_sigma=1.0,
    canny_low=None,
    canny_high=None,
    one_per_folder=False,
    max_files_per_split=None,
):
    """
    遍历 SceneFlow disparity 目录，生成 GT Edge Map。

    目录关系（和你现在的数据完全一致）：
        /data/sceneflow/
            disparity/      # 视差 .pfm
                TRAIN/...
                TEST/...
            frames_finalpass/  # RGB 图像
                TRAIN/...
                TEST/...
            gtedge/         # 我们要生成的 GT 边缘（本脚本输出）
                TRAIN/...
                TEST/...
    root_disp 对应上面的 disparity，root_edge 对应 gtedge。
    """

    for split in split_list:
        # 例如：.../disparity/TRAIN/**.pfm
        pattern = osp.join(root_disp, split, "**", "*.pfm")
        all_files = sorted(glob(pattern, recursive=True))

        # 选择要处理的文件列表
        if one_per_folder:
            # 每个目录只取一张，用于快速可视化/验证
            disp_files = []
            seen_dirs = set()
            for p in all_files:
                ddir = osp.dirname(p)
                if ddir in seen_dirs:
                    continue
                seen_dirs.add(ddir)
                disp_files.append(p)
        else:
            disp_files = list(all_files)

        # 进一步限制每个 split 的最大处理数量
        if max_files_per_split is not None and max_files_per_split > 0:
            disp_files = disp_files[: max_files_per_split]

        print(
            f"[gtedge] Split={split}, found {len(all_files)} disparity files, "
            f"selected {len(disp_files)} for processing "
            f"(one_per_folder={one_per_folder}, max_files_per_split={max_files_per_split})"
        )

        for disp_path in tqdm(disp_files, desc=f"GT Edge {split}", ncols=80):
            # 读取 disparity
            disp = read_pfm(disp_path)
            if disp.ndim == 3:
                # 有些 PFM 可能是三通道，这里取第一通道
                disp = disp[..., 0]

            edge = disp_to_edge(
                disp,
                grad_thresh=grad_thresh,
                mode=mode,
                blur_ksize=blur_ksize,
                blur_sigma=blur_sigma,
                canny_low=canny_low,
                canny_high=canny_high,
            )

            # 生成对应的输出路径：把 root_disp 替换成 root_edge，并把后缀换成 .png
            # 例如：
            #   disp_path = .../disparity/TRAIN/A/xxx/0000.pfm
            #   rel_path = TRAIN/A/xxx/0000.pfm
            #   edge_path = .../gtedge/TRAIN/A/xxx/0000.png
            rel_path = osp.relpath(disp_path, root_disp)
            edge_rel = rel_path.replace(".pfm", ".png")
            edge_path = osp.join(root_edge, edge_rel)

            os.makedirs(osp.dirname(edge_path), exist_ok=True)
            cv2.imwrite(edge_path, edge)


def process_single_dir(
    disp_dir,
    root_disp="/home/qi.xiong/StereoMatching/IGEV-Improve/data/sceneflow/disparity",
    root_edge="/home/qi.xiong/StereoMatching/IGEV-Improve/data/sceneflow/ggtedge",
    grad_thresh=2.5,
    mode="sobel",
    blur_ksize=5,
    blur_sigma=1.0,
    canny_low=None,
    canny_high=None,
):
    """
    仅处理指定的 disparity 子目录（例如：
        /home/.../data/sceneflow/disparity/TRAIN/15mm_focallength/scene_backwards/fast/left
    ），方便快速调试可视化。
    """
    disp_dir = osp.abspath(disp_dir)
    pattern = osp.join(disp_dir, "*.pfm")
    disp_files = sorted(glob(pattern))

    print(f"[gtedge] Single dir mode: {disp_dir}, found {len(disp_files)} disparity files")

    for disp_path in tqdm(disp_files, desc="GT Edge single dir", ncols=80):
        disp = read_pfm(disp_path)
        if disp.ndim == 3:
            disp = disp[..., 0]

        edge = disp_to_edge(
            disp,
            grad_thresh=grad_thresh,
            mode=mode,
            blur_ksize=blur_ksize,
            blur_sigma=blur_sigma,
            canny_low=canny_low,
            canny_high=canny_high,
        )

        rel_path = osp.relpath(disp_path, root_disp)
        edge_rel = rel_path.replace(".pfm", ".png")
        edge_path = osp.join(root_edge, edge_rel)

        os.makedirs(osp.dirname(edge_path), exist_ok=True)
        cv2.imwrite(edge_path, edge)


if __name__ == "__main__":
    """
    使用示例：
        # 1) 直接 Sobel + 阈值 2.5（和之前版本等价）
        python gtedge.py --grad_thresh 2.5 --mode sobel

        # 2) 先高斯模糊再求梯度，只保留大尺度几何边
        python gtedge.py --grad_thresh 2.5 --mode blur_sobel --blur_ksize 5 --blur_sigma 1.0
    """

    parser = argparse.ArgumentParser(description="Generate GT geometric edge maps from SceneFlow disparity.")
    parser.add_argument("--grad_thresh", type=float, default=3.0, help="梯度幅值阈值，越大边越稀疏（单位：像素差）。")
    parser.add_argument(
        "--mode",
        type=str,
        default="sobel",
        choices=["sobel", "sobel_rel", "blur_sobel", "laplacian", "blur_laplacian", "laplacian_close", "canny"],
        help=(
            "edge 计算模式："
            "sobel（原始一阶绝对梯度）, "
            "sobel_rel（一阶相对梯度 |∇d|/(|d|+eps)，对大平面更友好）, "
            "blur_sobel（先模糊再一阶）, "
            "laplacian（原始二阶）, "
            "blur_laplacian（先模糊再二阶，对平面斜坡最不敏感）, "
            "laplacian_close（二阶 + 闭运算：先膨胀再腐蚀，加粗并连通细边）, "
            "canny（在 disparity 上做归一化后的 Canny）"
        ),
    )
    parser.add_argument(
        "--blur_ksize",
        type=int,
        default=5,
        help="高斯模糊核大小（用于 blur_* 模式和 canny 的预平滑，需为奇数）。",
    )
    parser.add_argument(
        "--blur_sigma",
        type=float,
        default=1.0,
        help="高斯模糊 sigma（用于 blur_* 模式和 canny 的预平滑）。",
    )
    parser.add_argument(
        "--canny_low",
        type=float,
        default=None,
        help="Canny 低阈值（0-255），留空则使用默认50。",
    )
    parser.add_argument(
        "--canny_high",
        type=float,
        default=None,
        help="Canny 高阈值（0-255），留空则使用 max(canny_low*3,150)。",
    )
    parser.add_argument(
        "--one_per_folder",
        action="store_true",
        help="每个 disparity 子目录只处理一张，用于快速可视化/验证。",
    )
    parser.add_argument(
        "--max_files_per_split",
        type=int,
        default=0,
        help="每个 split 最多处理多少张（按路径排序截断），默认50；设为<=0表示不截断。",
    )
    parser.add_argument(
        "--disp_dir",
        type=str,
        default=None,
        help="仅处理指定 disparity 子目录（例如 TRAIN/15mm_focallength/.../left），用于快速调试。",
    )

    args = parser.parse_args()

    if args.disp_dir is not None:
        process_single_dir(
            disp_dir=args.disp_dir,
            grad_thresh=args.grad_thresh,
            mode=args.mode,
            blur_ksize=args.blur_ksize,
            blur_sigma=args.blur_sigma,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
        )
    else:
        process_sceneflow_disp(
            grad_thresh=args.grad_thresh,
            mode=args.mode,
            blur_ksize=args.blur_ksize,
            blur_sigma=args.blur_sigma,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            one_per_folder=args.one_per_folder,
            max_files_per_split=args.max_files_per_split,
        )

