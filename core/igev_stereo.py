import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock
from core.extractor import MultiBasicEncoder, Feature
from core.geometry import Combined_Geo_Encoding_Volume
from core.submodule import *
from core.rcf_models import RCF
import time
from tqdm import tqdm

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))



        self.feature_att_8 = FeatureAtt(in_channels*2, 64)
        self.feature_att_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_32 = FeatureAtt(in_channels*6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels*2, 64)

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv

class IGEVStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # Edge 来源: 'rcf' 使用 RCF 分支在线预测; 'gt' 使用外部读取的 gt edge (如 gtedge.py 预生成)
        self.edge_source = getattr(args, 'edge_source', 'rcf')

        context_dims = args.hidden_dims

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn="batch", downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        # ================== Edge Context Fusion ====================##
        # 多模态融合：将 Edge 与 Context 融合，支持 concat / film / gated 三种模式
        # concat: Hard 融合，直接拼接后 1x1 conv
        # film:   Soft 融合，FiLM (Feature-wise Linear Modulation)，edge 生成 γ,β 调制 context
        # gated:  Soft 融合，门控机制，学习 edge 与 context 的自适应权重
        self.edge_context_fusion = getattr(args, 'edge_context_fusion', False)
        self.edge_fusion_mode = getattr(args, 'edge_fusion_mode', 'film')
        if self.edge_context_fusion:
            if self.edge_fusion_mode == 'concat':
                self.edge_fusion_conv = nn.ModuleList([
                    nn.Conv2d(context_dims[i] + 1, context_dims[i], kernel_size=1)
                    for i in range(self.args.n_gru_layers)
                ])
            elif self.edge_fusion_mode == 'film':
                # FiLM: out = (1 + γ) * context + β, γ/β 由 edge 生成，实现条件调制
                self.edge_film_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(1, context_dims[i] // 4, 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(context_dims[i] // 4, context_dims[i] * 2, 1),  # γ 和 β 各 C 维
                    )
                    for i in range(self.args.n_gru_layers)
                ])
            elif self.edge_fusion_mode == 'gated':
                # Gated: gate = σ(f(concat)), out = gate * context + (1-gate) * edge_proj
                self.edge_gate_conv = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(context_dims[i] + 1, context_dims[i], 1),
                        nn.Sigmoid(),
                    )
                    for i in range(self.args.n_gru_layers)
                ])
                self.edge_proj_conv = nn.ModuleList([
                    nn.Conv2d(1, context_dims[i], 3, padding=1)
                    for i in range(self.args.n_gru_layers)
                ])
            else:
                raise ValueError(f"edge_fusion_mode must be one of concat/film/gated, got {self.edge_fusion_mode}")
        # ================== End Edge Context Fusion ====================##

        # ================== Context ZQR Convs ====================##
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])
        # ================== End Context ZQR ====================##
        self.feature = Feature()

        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)

        # ================== Edge-Guided Upsampling ====================##
        # 使用 Edge 引导视差上采样，支持多种融合方式：concat / film / gated / mlp
        self.edge_guided_upsample = getattr(args, 'edge_guided_upsample', False)
        self.edge_upsample_fusion_mode = getattr(args, 'edge_upsample_fusion_mode', 'film')
        if self.edge_guided_upsample:
            spx_channels = 64
            if self.edge_upsample_fusion_mode == 'concat':
                self.edge_upsample_fusion = nn.Sequential(
                    nn.Conv2d(spx_channels + 1, spx_channels, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(spx_channels, spx_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            elif self.edge_upsample_fusion_mode == 'film':
                self.edge_upsample_film = nn.Sequential(
                    nn.Conv2d(1, 16, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, spx_channels * 2, 1),
                )
            elif self.edge_upsample_fusion_mode == 'gated':
                self.edge_upsample_gate = nn.Sequential(
                    nn.Conv2d(spx_channels + 1, spx_channels, 1),
                    nn.Sigmoid(),
                )
                self.edge_upsample_proj = nn.Conv2d(1, spx_channels, 3, padding=1)
            elif self.edge_upsample_fusion_mode == 'mlp':
                self.edge_upsample_mlp = nn.Sequential(
                    nn.Conv2d(spx_channels + 1, spx_channels * 2, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(spx_channels * 2, spx_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            else:
                raise ValueError(f"edge_upsample_fusion_mode must be concat/film/gated/mlp, got {self.edge_upsample_fusion_mode}")
        # ================== End Edge-Guided Upsampling ====================##

        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, 96)
        # ================== Edge-Guided GWC ====================##
        # 将 Edge 注入 corr_feature_att 的 feat，使 GWC 的 attention 在边界处更具判别力
        # features[0] 通道数为 96，与 corr_feature_att 的 feat_chan 一致
        self.edge_guided_gwc = getattr(args, 'edge_guided_gwc', False)
        self.edge_gwc_fusion_mode = getattr(args, 'edge_gwc_fusion_mode', 'film')
        gwc_feat_chan = 96
        if self.edge_guided_gwc:
            if self.edge_gwc_fusion_mode == 'concat':
                self.edge_gwc_fusion = nn.Sequential(
                    nn.Conv2d(gwc_feat_chan + 1, gwc_feat_chan, 1),
                    nn.ReLU(inplace=True),
                )
            elif self.edge_gwc_fusion_mode == 'film':
                self.edge_gwc_film = nn.Sequential(
                    nn.Conv2d(1, 24, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(24, gwc_feat_chan * 2, 1),
                )
            elif self.edge_gwc_fusion_mode == 'gated':
                self.edge_gwc_gate = nn.Sequential(
                    nn.Conv2d(gwc_feat_chan + 1, gwc_feat_chan, 1),
                    nn.Sigmoid(),
                )
                self.edge_gwc_proj = nn.Conv2d(1, gwc_feat_chan, 3, padding=1)
            else:
                raise ValueError(f"edge_gwc_fusion_mode must be concat/film/gated, got {self.edge_gwc_fusion_mode}")
        # ================== End Edge-Guided GWC ====================##

        self.cost_agg = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

        # ================== Edge-Guided Cost Aggregation (Hourglass) ====================##
        # 将 Edge 注入 cost_agg 的 FeatureAtt，使 Hourglass 在物体边界处产生更优的 init_disp
        # 融合方式：concat / film / gated，与 features[1/2/3] 的通道 (64,192,160) 对应
        self.edge_guided_cost_agg = getattr(args, 'edge_guided_cost_agg', False)
        self.edge_cost_agg_fusion_mode = getattr(args, 'edge_cost_agg_fusion_mode', 'film')
        if self.edge_guided_cost_agg:
            feat_chans = [64, 192, 160]  # features[1], [2], [3] 通道数
            if self.edge_cost_agg_fusion_mode == 'concat':
                self.edge_cost_agg_fusion = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(feat_chans[i] + 1, feat_chans[i], 1),
                        nn.ReLU(inplace=True),
                    )
                    for i in range(3)
                ])
            elif self.edge_cost_agg_fusion_mode == 'film':
                self.edge_cost_agg_film = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(1, feat_chans[i] // 4, 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(feat_chans[i] // 4, feat_chans[i] * 2, 1),
                    )
                    for i in range(3)
                ])
            elif self.edge_cost_agg_fusion_mode == 'gated':
                self.edge_cost_agg_gate = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(feat_chans[i] + 1, feat_chans[i], 1),
                        nn.Sigmoid(),
                    )
                    for i in range(3)
                ])
                self.edge_cost_agg_proj = nn.ModuleList([
                    nn.Conv2d(1, feat_chans[i], 3, padding=1)
                    for i in range(3)
                ])
            else:
                raise ValueError(f"edge_cost_agg_fusion_mode must be concat/film/gated, got {self.edge_cost_agg_fusion_mode}")
        # ================== End Edge-Guided Cost Aggregation ====================##

        # ================== Edge Branch ====================##
        # 仅当任一 Edge 相关功能启用时加载 Edge 分支，否则与原始 IGEV 完全一致
        self.edge_guided_disp_head = getattr(args, 'edge_guided_disp_head', False)
        self.edge_motion_encoder = getattr(args, 'edge_motion_encoder', False)
        # ================== Edge-Guided Refinement ====================##
        # 后处理：用 Edge 引导视差 refinement，在边界处锐化、平坦区可学习平滑
        # 残差形式 disp_refined = disp + f(disp, edge)，支持 concat / film / gated
        # boundary_only: 仅在边界区域施加 refinement，disp_refined = disp + mask(edge) * residual
        self.edge_guided_refinement = getattr(args, 'edge_guided_refinement', False)
        self.boundary_only_refinement = getattr(args, 'boundary_only_refinement', False)
        self._refinement_enabled = self.edge_guided_refinement or self.boundary_only_refinement
        self.edge_refinement_fusion_mode = getattr(args, 'edge_refinement_fusion_mode', 'film')
        if self._refinement_enabled:
            if self.edge_refinement_fusion_mode == 'concat':
                self.edge_refinement_conv = nn.Sequential(
                    nn.Conv2d(2, 32, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 32, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 1, 3, padding=1),
                )
            elif self.edge_refinement_fusion_mode == 'film':
                self.edge_refinement_film = nn.Sequential(
                    nn.Conv2d(1, 16, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 2, 1),
                )
            elif self.edge_refinement_fusion_mode == 'gated':
                self.edge_refinement_gate = nn.Sequential(
                    nn.Conv2d(2, 32, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 2, 1),
                )
            else:
                raise ValueError(f"edge_refinement_fusion_mode must be concat/film/gated, got {self.edge_refinement_fusion_mode}")
        # ================== End Edge-Guided Refinement ====================##

        self.edge_aware_smoothness = getattr(args, 'edge_aware_smoothness', False)
        self.edge_weight_epe = getattr(args, 'edge_weight_epe_weight', 0.0) > 0
        self._use_edge = (
            self.edge_context_fusion
            or self.edge_guided_upsample
            or self.edge_guided_disp_head
            or self.edge_guided_cost_agg
            or self.edge_guided_gwc
            or self.edge_motion_encoder
            or self._refinement_enabled
            or self.edge_aware_smoothness
            or self.edge_weight_epe
        )
        if self._use_edge:
            if self.edge_source == 'rcf':
                edge_model_path = getattr(args, 'edge_model', None)
                if edge_model_path is None:
                    raise ValueError("edge_model is required when edge_source='rcf' and any edge_* feature is enabled")
                self.edge = RCF()
                self.edge.load_state_dict(torch.load(edge_model_path, map_location='cpu'), strict=False)
                for param in self.edge.parameters():
                    param.requires_grad = False
                self.edge.eval()
            # 学习一个 log-scale，对 edge 强度做自适应缩放；无论 rcf 还是 gt 都共享这一个尺度
            self.edge_log_scale = nn.Parameter(torch.tensor(-1.0))
        # ================== End Edge Branch ====================##

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x, left_edge=None):
        """
        将 1/4 分辨率视差上采样到全分辨率。
        当 edge_guided_upsample 启用时，使用 left_edge 引导上采样权重，在物体边界处保持锐利。
        """
        with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)

            # ================== Edge-Guided Upsampling ====================##
            if self.edge_guided_upsample and left_edge is not None:
                edge_resized = F.interpolate(
                    left_edge,
                    size=xspx.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                if self.edge_upsample_fusion_mode == 'concat':
                    xspx_with_edge = torch.cat([xspx, edge_resized], dim=1)
                    xspx = xspx + self.edge_upsample_fusion(xspx_with_edge)
                elif self.edge_upsample_fusion_mode == 'film':
                    gamma, beta = self.edge_upsample_film(edge_resized).chunk(2, dim=1)
                    xspx = (1 + gamma) * xspx + beta
                elif self.edge_upsample_fusion_mode == 'gated':
                    gate = self.edge_upsample_gate(torch.cat([xspx, edge_resized], dim=1))
                    edge_proj = self.edge_upsample_proj(edge_resized)
                    xspx = gate * xspx + (1 - gate) * edge_proj
                elif self.edge_upsample_fusion_mode == 'mlp':
                    xspx_with_edge = torch.cat([xspx, edge_resized], dim=1)
                    xspx = xspx + self.edge_upsample_mlp(xspx_with_edge)
            # ================== End Edge-Guided Upsampling ====================##

            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp

    def refine_disp(self, disp, left_edge):
        """
        Edge-guided refinement: 用 edge 引导视差后处理，在边界处锐化、平坦区可学习平滑。
        残差形式 disp_refined = disp + f(disp, edge)。
        boundary_only 时：仅在边界施加 refinement，disp_refined = disp + mask(edge) * residual。
        """
        if not self._refinement_enabled or left_edge is None:
            return disp
        edge_resized = F.interpolate(
            left_edge, size=disp.shape[2:], mode='bilinear', align_corners=False
        )
        with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            if self.edge_refinement_fusion_mode == 'concat':
                disp_edge = torch.cat([disp, edge_resized], dim=1)
                residual = self.edge_refinement_conv(disp_edge)
            elif self.edge_refinement_fusion_mode == 'film':
                gamma, beta = self.edge_refinement_film(edge_resized).chunk(2, dim=1)
                residual = gamma * disp + beta
            elif self.edge_refinement_fusion_mode == 'gated':
                disp_edge = torch.cat([disp, edge_resized], dim=1)
                gate, residual = self.edge_refinement_gate(disp_edge).chunk(2, dim=1)
                residual = torch.sigmoid(gate) * residual
            else:
                return disp
            # boundary_only: 仅边界施加 residual，mask = edge (clamp 到 [0,1])
            if self.boundary_only_refinement:
                mask = edge_resized.clamp(0.0, 1.0)
                residual = mask * residual
            return disp + residual

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False, left_edge=None):
        """ Estimate disparity between pair of frames """

        # ================== Edge ====================##
        # left_edge 输入优先级:
        #   1) 当 edge_source='gt' 时, 期望调用者传入 left_edge (来自 gtedge 预生成);
        #   2) 当 edge_source='rcf' 时, 在线通过 RCF 预测.
        if self._use_edge:
            if self.edge_source == 'gt':
                if left_edge is None:
                    raise ValueError("left_edge tensor must be provided when edge_source='gt'")
                left_edge_raw = left_edge
            else:  # 'rcf'
                with torch.no_grad():
                    left_edge_raw = self.edge(image1)[-1]

            # 自适应缩放 edge 强度
            edge_scale = F.softplus(self.edge_log_scale)
            left_edge = left_edge_raw * edge_scale
        else:
            left_edge = None
        # ================== End Edge ====================##

        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            features_left = self.feature(image1)
            features_right = self.feature(image2)
            stem_2x = self.stem_2(image1)
            stem_4x = self.stem_4(stem_2x)
            stem_2y = self.stem_2(image2)
            stem_4y = self.stem_4(stem_2y)
            features_left[0] = torch.cat((features_left[0], stem_4x), 1)
            features_right[0] = torch.cat((features_right[0], stem_4y), 1)

            match_left = self.desc(self.conv(features_left[0]))
            match_right = self.desc(self.conv(features_right[0]))
            gwc_volume = build_gwc_volume(match_left, match_right, self.args.max_disp//4, 8)
            gwc_volume = self.corr_stem(gwc_volume)
            # ================== Edge-Guided GWC ====================##
            # 将 left_edge 注入 features[0]，使 corr_feature_att 的 attention 在边界处更优
            gwc_feat = features_left[0]
            if self.edge_guided_gwc and left_edge is not None:
                edge_resized = F.interpolate(
                    left_edge, size=gwc_feat.shape[2:], mode='bilinear', align_corners=False
                )
                if self.edge_gwc_fusion_mode == 'concat':
                    feat_with_edge = torch.cat([gwc_feat, edge_resized], dim=1)
                    gwc_feat = gwc_feat + self.edge_gwc_fusion(feat_with_edge)
                elif self.edge_gwc_fusion_mode == 'film':
                    gamma, beta = self.edge_gwc_film(edge_resized).chunk(2, dim=1)
                    gwc_feat = (1 + gamma) * gwc_feat + beta
                elif self.edge_gwc_fusion_mode == 'gated':
                    gate = self.edge_gwc_gate(torch.cat([gwc_feat, edge_resized], dim=1))
                    edge_proj = self.edge_gwc_proj(edge_resized)
                    gwc_feat = gate * gwc_feat + (1 - gate) * edge_proj
            gwc_volume = self.corr_feature_att(gwc_volume, gwc_feat)
            # ================== End Edge-Guided GWC ====================##
            # ================== Edge-Guided Cost Aggregation ====================##
            # 将 left_edge 注入 features[1/2/3]，使 Hourglass 的 FeatureAtt 在边界处产生更优的 init_disp
            cost_agg_features = list(features_left)
            if self.edge_guided_cost_agg and left_edge is not None:
                for i in range(1, 4):  # features[1], [2], [3] -> fusion modules [0], [1], [2]
                    idx = i - 1
                    edge_resized = F.interpolate(
                        left_edge, size=features_left[i].shape[2:],
                        mode='bilinear', align_corners=False
                    )
                    feat = features_left[i]
                    if self.edge_cost_agg_fusion_mode == 'concat':
                        feat_with_edge = torch.cat([feat, edge_resized], dim=1)
                        cost_agg_features[i] = self.edge_cost_agg_fusion[idx](feat_with_edge)
                    elif self.edge_cost_agg_fusion_mode == 'film':
                        gamma, beta = self.edge_cost_agg_film[idx](edge_resized).chunk(2, dim=1)
                        cost_agg_features[i] = (1 + gamma) * feat + beta
                    elif self.edge_cost_agg_fusion_mode == 'gated':
                        gate = self.edge_cost_agg_gate[idx](torch.cat([feat, edge_resized], dim=1))
                        edge_proj = self.edge_cost_agg_proj[idx](edge_resized)
                        cost_agg_features[i] = gate * feat + (1 - gate) * edge_proj
            geo_encoding_volume = self.cost_agg(gwc_volume, cost_agg_features)
            # ================== End Edge-Guided Cost Aggregation ====================##

            # Init disp from geometry encoding volume
            prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
            init_disp = disparity_regression(prob, self.args.max_disp//4)
            
            del prob, gwc_volume

            if not test_mode:
                xspx = self.spx_4(features_left[0])
                xspx = self.spx_2(xspx, stem_2x)
                # ================== Edge-Guided Upsampling (init_disp) ====================##
                if self.edge_guided_upsample and left_edge is not None:
                    edge_resized = F.interpolate(
                        left_edge, size=xspx.shape[2:], mode='bilinear', align_corners=False
                    )
                    if self.edge_upsample_fusion_mode == 'concat':
                        xspx_with_edge = torch.cat([xspx, edge_resized], dim=1)
                        xspx = xspx + self.edge_upsample_fusion(xspx_with_edge)
                    elif self.edge_upsample_fusion_mode == 'film':
                        gamma, beta = self.edge_upsample_film(edge_resized).chunk(2, dim=1)
                        xspx = (1 + gamma) * xspx + beta
                    elif self.edge_upsample_fusion_mode == 'gated':
                        gate = self.edge_upsample_gate(torch.cat([xspx, edge_resized], dim=1))
                        edge_proj = self.edge_upsample_proj(edge_resized)
                        xspx = gate * xspx + (1 - gate) * edge_proj
                    elif self.edge_upsample_fusion_mode == 'mlp':
                        xspx_with_edge = torch.cat([xspx, edge_resized], dim=1)
                        xspx = xspx + self.edge_upsample_mlp(xspx_with_edge)
                # ================== End Edge-Guided Upsampling (init_disp) ====================##
                spx_pred = self.spx(xspx)
                spx_pred = F.softmax(spx_pred, 1)

            cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]

            # ================== Edge Context Fusion ====================##
            if self.edge_context_fusion and left_edge is not None:
                fused_inp_list = []
                for i in range(self.args.n_gru_layers):
                    edge_resized = F.interpolate(
                        left_edge,
                        size=inp_list[i].shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                    ctx = inp_list[i]
                    if self.edge_fusion_mode == 'concat':
                        inp_with_edge = torch.cat([ctx, edge_resized], dim=1)
                        fused_inp = self.edge_fusion_conv[i](inp_with_edge)
                    elif self.edge_fusion_mode == 'film':
                        # FiLM: out = (1 + γ) * context + β
                        film_params = self.edge_film_proj[i](edge_resized)
                        gamma, beta = film_params.chunk(2, dim=1)
                        fused_inp = (1 + gamma) * ctx + beta
                    elif self.edge_fusion_mode == 'gated':
                        # Gated: out = gate * context + (1 - gate) * edge_proj
                        gate = self.edge_gate_conv[i](torch.cat([ctx, edge_resized], dim=1))
                        edge_proj = self.edge_proj_conv[i](edge_resized)
                        fused_inp = gate * ctx + (1 - gate) * edge_proj
                    fused_inp_list.append(fused_inp)
                inp_list = fused_inp_list
            # ================== End Edge Context Fusion ====================##

            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i, conv in zip(inp_list, self.context_zqr_convs)]


        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(match_left.float(), match_right.float(), geo_encoding_volume.float(), radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        b, c, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1,1,w,1).repeat(b, h, 1, 1)
        disp = init_disp
        disp_preds = []

        # GRUs iterations to update disparity
        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords)
            with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
                net_list, mask_feat_4, delta_disp = self.update_block(
                    net_list, inp_list, geo_feat, disp,
                    edge=left_edge if (self.edge_guided_disp_head or self.edge_motion_encoder) else None,
                    iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2
                )

            disp = disp + delta_disp
            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            disp_up = self.upsample_disp(
                disp, mask_feat_4, stem_2x,
                left_edge=left_edge if self.edge_guided_upsample else None
            )
            disp_up = self.refine_disp(disp_up, left_edge)
            disp_preds.append(disp_up)

        if test_mode:
            return disp_preds[-1]

        init_disp = context_upsample(init_disp*4., spx_pred.float()).unsqueeze(1)
        init_disp = self.refine_disp(init_disp, left_edge)
        return init_disp, disp_preds, left_edge
