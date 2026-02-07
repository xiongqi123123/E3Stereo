import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import re


def read_pfm(file):
    """读取 PFM 格式的视差图 (SceneFlow 数据集常用)"""
    file = open(file, 'rb')
    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode('utf-8').rstrip())
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


def print_random_patch(disp, edge_map, n=5, find_edge=False):
    """
    随机选取一个 N*N 的区域，打印 GT disparity 和 edge weight map 的数值
    如果 find_edge=True，则选取边缘较强的区域
    """
    h, w = disp.shape
    np.set_printoptions(precision=2, suppress=True, linewidth=120)

    if find_edge:
        # 找到边缘值较大的区域
        from scipy.ndimage import uniform_filter
        edge_avg = uniform_filter(edge_map, size=n, mode='constant')
        # 排除边界
        edge_avg[:n, :] = 0
        edge_avg[-n:, :] = 0
        edge_avg[:, :n] = 0
        edge_avg[:, -n:] = 0
        # 找到最大值位置
        max_idx = np.unravel_index(np.argmax(edge_avg), edge_avg.shape)
        y, x = max_idx[0] - n // 2, max_idx[1] - n // 2
        label = "高边缘区域"
    else:
        # 随机选取左上角坐标，确保不越界
        y = np.random.randint(0, h - n)
        x = np.random.randint(0, w - n)
        label = "随机选取区域"

    disp_patch = disp[y:y+n, x:x+n]
    edge_patch = edge_map[y:y+n, x:x+n]

    print(f"\n{label}: 左上角坐标 (y={y}, x={x}), 大小 {n}x{n}")
    print("=" * 60)

    print(f"\nGT Disparity ({n}x{n}):")
    print(disp_patch)

    print(f"\nEdge Weight Map ({n}x{n}):")
    print(edge_patch)

    print("=" * 60)

    return y, x, disp_patch, edge_patch


def non_maximum_suppression(grad_mag, grad_x, grad_y):
    """
    非极大值抑制：沿梯度方向只保留局部最大值，生成细线条边缘
    """
    h, w = grad_mag.shape
    nms = np.zeros_like(grad_mag)

    # 计算梯度方向 (弧度)
    angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
    angle[angle < 0] += 180  # 转换到 0-180 度

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # 根据梯度方向确定比较的邻居
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                neighbors = [grad_mag[i, j-1], grad_mag[i, j+1]]
            elif 22.5 <= angle[i, j] < 67.5:
                neighbors = [grad_mag[i-1, j+1], grad_mag[i+1, j-1]]
            elif 67.5 <= angle[i, j] < 112.5:
                neighbors = [grad_mag[i-1, j], grad_mag[i+1, j]]
            else:  # 112.5 - 157.5
                neighbors = [grad_mag[i-1, j-1], grad_mag[i+1, j+1]]

            # 只保留局部最大值
            if grad_mag[i, j] >= neighbors[0] and grad_mag[i, j] >= neighbors[1]:
                nms[i, j] = grad_mag[i, j]

    return nms


def compute_edge_map(disp, method='sobel', threshold=None):
    """
    计算边缘图
    method:
        'sobel' - 原始 Sobel 梯度幅值（较粗的边缘）
        'nms' - Sobel + 非极大值抑制（细线条）
        'canny' - Canny 边缘检测（二值细线条）
        'laplacian' - Laplacian 算子
    threshold: 边缘阈值，低于此值的设为0
    """
    # 使用 Sobel 算子计算梯度
    grad_x = cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(disp, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    if method == 'sobel':
        edge_map = grad_mag
    elif method == 'nms':
        edge_map = non_maximum_suppression(grad_mag, grad_x, grad_y)
    elif method == 'canny':
        # Canny 需要 uint8 输入
        disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        edge_map = cv2.Canny(disp_norm, 50, 150).astype(np.float32)
    elif method == 'laplacian':
        edge_map = np.abs(cv2.Laplacian(disp, cv2.CV_32F, ksize=3))
    else:
        raise ValueError(f"Unknown method: {method}")

    # 应用阈值
    if threshold is not None:
        edge_map[edge_map < threshold] = 0

    return edge_map, grad_x, grad_y


def generate_edge_map(disp_path, output_path='edge_visualization.png', patch_size=5,
                      method='sobel', threshold=None, compare_methods=False):
    """
    输入: GT 视差图路径
    输出: 可视化的边缘图，并保存

    method: 'sobel', 'nms', 'canny', 'laplacian'
    threshold: 边缘阈值
    compare_methods: 如果为 True，对比显示所有方法
    """
    # 1. 加载视差图
    if disp_path.endswith('.pfm'):
        disp, _ = read_pfm(disp_path)
    else:
        disp = cv2.imread(disp_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        if "KITTI" in disp_path:
            disp = disp / 256.0

    # 处理无效值
    disp[np.isinf(disp)] = 0

    if compare_methods:
        # 对比所有方法
        methods = ['sobel', 'nms', 'canny', 'laplacian']
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 显示原始视差图
        disp_viz = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_viz, cv2.COLORMAP_MAGMA)
        axes[0, 0].imshow(cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("GT Disparity")
        axes[0, 0].axis('off')

        for idx, m in enumerate(methods):
            edge_map, _, _ = compute_edge_map(disp, method=m)
            max_val = np.percentile(edge_map[edge_map > 0], 95) if edge_map.max() > 0 else 1
            edge_viz = np.clip(edge_map, 0, max_val)
            edge_viz = cv2.normalize(edge_viz, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            edge_color = cv2.applyColorMap(edge_viz, cv2.COLORMAP_JET)

            row, col = (idx + 1) // 3, (idx + 1) % 3
            axes[row, col].imshow(cv2.cvtColor(edge_color, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(f"Method: {m}")
            axes[row, col].axis('off')

        axes[1, 2].axis('off')  # 隐藏多余的子图
        plt.tight_layout()
        plt.savefig(output_path.replace('.png', '_compare.png'))
        plt.show()
        print(f"对比图已保存至: {output_path.replace('.png', '_compare.png')}")
        return

    # 2. 计算边缘图
    edge_map, grad_x, grad_y = compute_edge_map(disp, method=method, threshold=threshold)

    # 3. 可视化
    max_val = np.percentile(edge_map[edge_map > 0], 95) if edge_map.max() > 0 else 1
    edge_viz = np.clip(edge_map, 0, max_val)
    edge_viz = cv2.normalize(edge_viz, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edge_color = cv2.applyColorMap(edge_viz, cv2.COLORMAP_JET)

    # 4. 保存与显示
    disp_viz = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_viz, cv2.COLORMAP_MAGMA)

    combined = np.hstack((disp_color, edge_color))
    cv2.imwrite(output_path, combined)

    print(f"处理完成！方法: {method}")
    print(f"结果已保存至: {output_path}")
    print(f"边缘图最大梯度值: {edge_map.max():.2f}")
    print(f"非零边缘像素数: {np.count_nonzero(edge_map)}")

    # 打印 patch 数值
    print_random_patch(disp, edge_map, n=patch_size, find_edge=False)
    print_random_patch(disp, edge_map, n=patch_size, find_edge=True)

    # matplotlib 预览
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("GT Disparity (Colorized)")
    plt.imshow(cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Edge Map (Method: {method})")
    plt.imshow(cv2.cvtColor(edge_color, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 替换为你实际的 GT 视差图路径
    # test_path = '/root/autodl-tmp/stereo/dataset_cache/SceneFlow/FlyingThings3D/disparity/TRAIN/A/0130/left/0015.pfm'
    test_path = '/root/autodl-tmp/stereo/dataset_cache/SceneFlow/FlyingThings3D/disparity/TRAIN/A/0004/left/0012.pfm'

    if os.path.exists(test_path):
        generate_edge_map(test_path)
    else:
        print(f"请检查路径，未找到文件: {test_path}")