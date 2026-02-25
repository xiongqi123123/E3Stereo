import torch
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler


class Combined_Geo_Encoding_Volume:
    def __init__(self, init_fmap1, init_fmap2, geo_volume, num_levels=2, radius=4, edge_radius_shrink=0.5):
        self.num_levels = num_levels
        self.radius = radius
        # 当提供 edge 时，用于缩小有效搜索半径的系数 λ，越大表示边缘处窗口越窄。
        # 有效缩放因子 α = 1 - λ * edge，防止过窄会在调用处再做 clamp。
        self.edge_radius_shrink = edge_radius_shrink
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []

        # all pairs correlation
        init_corr = Combined_Geo_Encoding_Volume.corr(init_fmap1, init_fmap2)

        b, h, w, _, w2 = init_corr.shape
        b, c, d, h, w = geo_volume.shape
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d)

        init_corr = init_corr.reshape(b*h*w, 1, 1, w2)
        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)
        for i in range(self.num_levels-1):
            geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2])
            self.geo_volume_pyramid.append(geo_volume)

        for i in range(self.num_levels-1):
            init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2])
            self.init_corr_pyramid.append(init_corr)

    def __call__(self, disp, coords, edge=None):
        r = self.radius
        b, _, h, w = disp.shape
        out_pyramid = []
        # 若提供 edge，则根据 edge 构造 per-pixel 的采样缩放因子 alpha \in [alpha_min, 1]
        # alpha 越小 -> 有效搜索半径越小，边缘附近更“局部”，平坦区域保持原有半径。
        alpha = None
        if edge is not None:
            # edge: [B, 1 or C, H, W]，这里统一为单通道并 clamp 到 [0, 1]
            if edge.dim() == 4 and edge.shape[1] != 1:
                edge = edge.mean(dim=1, keepdim=True)
            if edge.shape[-2:] != (h, w):
                edge = F.interpolate(edge.float(), size=(h, w), mode='bilinear', align_corners=False)
            edge_norm = edge.clamp(0.0, 1.0)
            # 展平成 [B*H*W, 1, 1, 1]，与 dx 广播
            edge_flat = edge_norm.view(b * h * w, 1, 1, 1)
            # α = 1 - λ * edge, 其中 λ = self.edge_radius_shrink
            alpha = 1.0 - self.edge_radius_shrink * edge_flat
            # 防止窗口过窄，限制最小缩放系数
            alpha = alpha.clamp(0.25, 1.0)
        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=disp.device, dtype=disp.dtype)
            dx = dx.view(1, 1, 2*r+1, 1)
            if alpha is not None:
                # 对每个像素的离散 offset 做缩放：dx_scaled = alpha * dx
                dx_scaled = alpha * dx
            else:
                dx_scaled = dx
            x0 = dx_scaled + disp.reshape(b*h*w, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            disp_lvl = torch.cat([x0,y0], dim=-1)
            geo_volume = bilinear_sampler(geo_volume, disp_lvl)
            geo_volume = geo_volume.view(b, h, w, -1)

            init_corr = self.init_corr_pyramid[i]
            init_x0 = coords.reshape(b*h*w, 1, 1, 1)/2**i - disp.reshape(b*h*w, 1, 1, 1) / 2**i + dx_scaled
            init_coords_lvl = torch.cat([init_x0,y0], dim=-1)
            init_corr = bilinear_sampler(init_corr, init_coords_lvl)
            init_corr = init_corr.view(b, h, w, -1)

            out_pyramid.append(geo_volume)
            out_pyramid.append(init_corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    
    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr