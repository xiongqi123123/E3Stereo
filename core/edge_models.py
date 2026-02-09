"""
几何边缘分支：复用 IGEV 的 Feature 作为 backbone，接 EdgeHead 学习从 RGB 预测 depth-discontinuity 几何边缘。

用于验证：在 SceneFlow 合成数据上，模型能否从单张 RGB 图像学习到几何边缘特征（label 为 disparity 梯度生成的 GT edge）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.extractor import Feature
from core.submodule import BasicConv_IN


class SpatialAttention(nn.Module):
    """
    空间注意力：从多尺度 edge 预测生成空间权重图，在边缘区域增强、背景抑制。
    边缘稀疏，通过空间注意力让网络更关注边界附近。
    """
    def __init__(self, in_channels=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        x: [B, C, H, W] 多尺度 edge 预测的 concat
        return: [B, 1, H, W] 空间权重，边缘区域接近 1，背景接近 0
        """
        return self.conv(x)


class EdgeRefinementModule(nn.Module):
    """
    细边缘锐化模块：在 full-res 上利用 RGB 引导，对模糊的粗边缘进行 refinement。
    针对「细边缘糊成一坨」问题：通过 residual 学习锐化，使预测更细、更清晰。
    """
    def __init__(self, in_channels=4):
        super().__init__()
        # in_channels: 1(edge) + 3(rgb)
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, 1)
        self.norm = nn.InstanceNorm2d(32)

    def forward(self, edge_logits, rgb):
        """
        edge_logits: [B, 1, H, W] 粗边缘预测（已上采样到 full-res）
        rgb: [B, 3, H, W] 原图（用于引导）
        """
        x = torch.cat([edge_logits, rgb], dim=1)
        x = F.leaky_relu(self.norm(self.conv1(x)), 0.1)
        x = F.leaky_relu(self.norm(self.conv2(x)), 0.1)
        delta = self.conv3(x)
        return edge_logits + delta  # residual


class EdgeHead(nn.Module):
    """
    多尺度特征融合的 Edge 预测头。
    输入：Feature 输出的多尺度特征 [x4, x8, x16, x32]
    输出：与输入同分辨率的单通道 edge map (logits)
    use_spatial_attn: 是否使用空间注意力
    """
    def __init__(self, feat_channels=(48, 64, 192, 160), use_spatial_attn=True):
        super().__init__()
        self.use_spatial_attn = use_spatial_attn
        # feat_channels 对应 Feature 输出的 [x4, x8, x16, x32] 通道数
        c4, c8, c16, c32 = feat_channels

        # 从各尺度预测 edge，再上采样融合
        self.edge_4 = nn.Sequential(
            BasicConv_IN(c4, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 1, 1),
        )
        self.edge_8 = nn.Sequential(
            BasicConv_IN(c8, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 1, 1),
        )
        self.edge_16 = nn.Sequential(
            BasicConv_IN(c16, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 1, 1),
        )
        self.edge_32 = nn.Sequential(
            BasicConv_IN(c32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 1, 1),
        )

        # 融合不同尺度的预测；使用可学习 scale 让 e4 更占优，利于细边缘
        self.scale = nn.Parameter(torch.ones(4) * 0.4)  # e4 默认略高
        self.scale.data[0] = 1.2  # e4 细尺度权重更高
        self.spatial_attn = SpatialAttention(in_channels=4) if use_spatial_attn else None
        self.fuse = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, features, target_size=None):
        """
        features: list of [x4, x8, x16, x32]
        target_size: (H, W) 最终输出上采样尺寸，融合始终在 e4 分辨率进行
        """
        e4 = self.edge_4(features[0])
        e8 = self.edge_8(features[1])
        e16 = self.edge_16(features[2])
        e32 = self.edge_32(features[3])

        # 融合时统一到 e4 分辨率（x4 为 1/4 输入）
        h, w = e4.shape[2], e4.shape[3]

        e8_up = F.interpolate(e8, size=(h, w), mode='bilinear', align_corners=False)
        e16_up = F.interpolate(e16, size=(h, w), mode='bilinear', align_corners=False)
        e32_up = F.interpolate(e32, size=(h, w), mode='bilinear', align_corners=False)

        # 融合多尺度，scale 使 e4 更占优，利于细边缘
        scale = F.softmax(self.scale, dim=0)
        fused = torch.cat([
            e4 * scale[0], e8_up * scale[1], e16_up * scale[2], e32_up * scale[3]
        ], dim=1)
        # 空间注意力：边缘区域增强、背景抑制
        if self.spatial_attn is not None:
            attn = self.spatial_attn(fused)
            fused = fused * (1.0 + attn)
        edge_logits = self.fuse(fused)
        return edge_logits


class GeoEdgeNet(nn.Module):
    """
    几何边缘网络：复用 IGEV 的 Feature backbone + EdgeHead。
    输入：RGB 图像 [B, 3, H, W]，归一化到 [-1, 1]
    输出：edge logits [B, 1, H, W]
    use_refinement: 是否使用 EdgeRefinementModule 锐化细边缘
    refine_iters: Refine 迭代次数，1=单次，2/3=迭代锐化（共享同一 Refine 模块）
    use_spatial_attn: 是否使用空间注意力（边缘增强、背景抑制）
    """
    def __init__(self, use_refinement=True, refine_iters=1, use_spatial_attn=True):
        super().__init__()
        self.use_refinement = use_refinement
        self.refine_iters = max(1, int(refine_iters))
        self.backbone = Feature()
        self.edge_head = EdgeHead(
            feat_channels=(48, 64, 192, 160),
            use_spatial_attn=use_spatial_attn,
        )
        if use_refinement:
            self.refine = EdgeRefinementModule(in_channels=4)

    def forward(self, x, target_size=None):
        """
        x: [B, 3, H, W], 值域建议 [-1, 1] 或 [0, 1]
        target_size: (H, W) 输出尺寸，默认与输入一致
        """
        features = self.backbone(x)
        if target_size is None:
            target_size = (x.shape[2], x.shape[3])
        edge_logits = self.edge_head(features, target_size)
        # 上采样到输入分辨率
        if edge_logits.shape[2:] != target_size:
            edge_logits = F.interpolate(
                edge_logits, size=target_size,
                mode='bilinear', align_corners=False
            )
        # 迭代 Refine：coarse -> refine -> refine -> ...（共享同一 Refine）
        if self.use_refinement:
            for _ in range(self.refine_iters):
                edge_logits = self.refine(edge_logits, x)
        return edge_logits
