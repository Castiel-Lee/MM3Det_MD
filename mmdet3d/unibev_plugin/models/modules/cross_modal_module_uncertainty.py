import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.runner import auto_fp16
from mmcv.cnn.bricks.registry import ATTENTION
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import math


class CrossModalComplementaryUncertaintyModule(nn.Module):
    """
    跨模态互补模块：采用Deformable Attention机制实现图像和LiDAR特征的互补增强
    
    核心思想：
    1. 使用Deformable Attention替代标准Multi-Head Attention
    2. 图像特征作为Query，LiDAR特征作为Key/Value进行cross-attention
    3. LiDAR特征作为Query，图像特征作为Key/Value进行cross-attention  
    4. 通过残差连接实现模态互补
    """
    
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=1,
                 num_points=4,
                 dropout=0.1,
                 norm_cfg=dict(type='LN'),
                 use_ffn=True,
                 ffn_ratio=4,
                 bev_h=200,
                 bev_w=200,
                 identity_dropout=False,
                 identity_keep_prob=0.9,
                 unc_head=False,
                 **kwargs):
        super().__init__()
        
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.use_ffn = use_ffn
        self.bev_h = bev_h
        self.bev_w = bev_w
        
        # Deformable Cross Attention: 图像->LiDAR
        self.img_to_pts_attention = DeformableCrossAttention_Conf(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout
        )
        
        # Deformable Cross Attention: LiDAR->图像  
        self.pts_to_img_attention = DeformableCrossAttention_Conf(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout
        )
        
        # 层归一化
        self.norm1_img = build_norm_layer(norm_cfg, embed_dims)[1]
        self.norm1_pts = build_norm_layer(norm_cfg, embed_dims)[1]
        
        if self.use_ffn:
            self.norm2_img = build_norm_layer(norm_cfg, embed_dims)[1]
            self.norm2_pts = build_norm_layer(norm_cfg, embed_dims)[1]
            
            self.ffn_img = FFN(embed_dims, embed_dims * ffn_ratio, dropout)
            self.ffn_pts = FFN(embed_dims, embed_dims * ffn_ratio, dropout)
        
        # 预先计算并缓存BEV位置编码
        self._bev_pos_cache = None
        self._cached_device = None
        self._cached_dtype = None
        
        self.identity_dropout = identity_dropout
        self.identity_keep_prob = identity_keep_prob

        self.unc_head = unc_head
        if self.unc_head:
            # 互补后的不确定性：基于 cross-fused 特征来估计
            self.unc_head_img = nn.Sequential(nn.Conv2d(embed_dims, 1, 1))
            self.unc_head_pts = nn.Sequential(nn.Conv2d(embed_dims, 1, 1))
        
        
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    @staticmethod
    def get_reference_points(H, W, bs=1, device='cuda', dtype=torch.float):
        """生成BEV网格的参考点"""
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)  # [bs, H*W, 1, 2]
        return ref_2d
    
    @auto_fp16(apply_to=('img_feat', 'pts_feat'))
    def forward(self, img_feat, pts_feat, u_img, u_pts, c_img, c_pts):
        """
        前向传播
        
        Args:
            img_feat (Tensor): 图像BEV特征 [B, H*W, C] 或 [B, C, H, W]
            pts_feat (Tensor): 点云BEV特征 [B, H*W, C] 或 [B, C, H, W]
            u_img (Tensor): 图像不确定性 [B, 1, H, W]
            u_pts (Tensor): 点云不确定性 [B, 1, H, W]
            
        Returns:
            tuple: (enhanced_img_feat, enhanced_pts_feat)
        """
        # 特征形状处理
        img_feat = self._reshape_feat(img_feat)  # [B, H*W, C]
        pts_feat = self._reshape_feat(pts_feat)  # [B, H*W, C]
        
        bs = img_feat.shape[0]
        device = img_feat.device
        dtype = img_feat.dtype
        
        # 获取BEV位置编码（内部计算和缓存）
        bev_pos = self._get_bev_pos_encoding(bs, device, dtype)
        
        # 生成参考点和空间形状
        reference_points = self.get_reference_points(self.bev_h, self.bev_w, bs, device, dtype)
        spatial_shapes = torch.tensor([[self.bev_h, self.bev_w]], device=device, dtype=torch.long)
        level_start_index = torch.tensor([0], device=device, dtype=torch.long)
        
        # 添加位置编码到特征
        # bev_pos: [H*W, bs, C] -> [bs, H*W, C]
        bev_pos_feat = bev_pos.permute(1, 0, 2)
        img_feat_with_pos = img_feat + bev_pos_feat
        pts_feat_with_pos = pts_feat + bev_pos_feat
        
        # 互为Query的Deformable Cross-Attention
        img_enhanced = self.img_to_pts_attention(
            query=img_feat_with_pos,
            key=pts_feat,
            value=pts_feat,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            bev_pos=bev_pos,
            value_confidence=c_pts,
        )
        
        pts_enhanced = self.pts_to_img_attention(
            query=pts_feat_with_pos,
            key=img_feat,
            value=img_feat,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            bev_pos=bev_pos,
            value_confidence=c_img,
        )

        # 残差连接 + 层归一化
        
        if self.identity_dropout and self.training:
            img_enhanced = self.norm1_img(img_enhanced + self._identity_dropout(img_feat, self.identity_keep_prob))
            pts_enhanced = self.norm1_pts(pts_enhanced + self._identity_dropout(pts_feat, self.identity_keep_prob))
        else:
            img_enhanced = self.norm1_img(img_enhanced + img_feat)
            pts_enhanced = self.norm1_pts(pts_enhanced + pts_feat)
        
        # FFN增强
        if self.use_ffn:
            img_enhanced = self.norm2_img(self.ffn_img(img_enhanced) + img_enhanced)
            pts_enhanced = self.norm2_pts(self.ffn_pts(pts_enhanced) + pts_enhanced)
        
        sigma2_img = sigma2_pts = None
        if self.unc_head:
            # softplus 保证正；数值更稳
            # print('Debug: unc_head')
            sigma2_img = F.softplus(self.unc_head_img(img_enhanced))
            sigma2_pts = F.softplus(self.unc_head_pts(pts_enhanced))

            return img_enhanced, pts_enhanced, sigma2_img, sigma2_pts
        
        return img_enhanced, pts_enhanced
    
    def _reshape_feat(self, feat):
        """将特征reshape为[B, H*W, C]格式"""
        if feat.dim() == 4:  # [B, C, H, W]
            feat = feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
        return feat
    
    def _generate_bev_pos_encoding(self, H, W, embed_dims, device, dtype):
        """生成sinusoidal BEV位置编码"""
        y_embed = torch.arange(H, dtype=dtype, device=device).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, dtype=dtype, device=device).unsqueeze(0).repeat(H, 1)
        
        # 归一化到[0, 1]
        y_embed = y_embed / H
        x_embed = x_embed / W
        
        # 生成sinusoidal编码
        dim_t = torch.arange(embed_dims // 4, dtype=dtype, device=device)
        dim_t = 10000 ** (2 * (dim_t // 2) / (embed_dims // 4))
        
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)
        
        pos = torch.cat([pos_y, pos_x], dim=-1)  # [H, W, embed_dims//2]
        
        # 如果维度不够，补零
        if pos.shape[-1] < embed_dims:
            pos = torch.cat([pos, torch.zeros(H, W, embed_dims - pos.shape[-1], 
                                            device=device, dtype=dtype)], dim=-1)
        
        # reshape为 [H*W, 1, embed_dims]
        pos = pos.reshape(H * W, embed_dims).unsqueeze(1)  # [H*W, 1, embed_dims]
        
        return pos
    
    def _get_bev_pos_encoding(self, bs, device, dtype):
        """获取BEV位置编码（支持缓存机制）"""
        # 检查是否需要重新计算
        if (self._bev_pos_cache is None or 
            self._cached_device != device or 
            self._cached_dtype != dtype):
            
            # 重新计算并缓存
            self._bev_pos_cache = self._generate_bev_pos_encoding(
                self.bev_h, self.bev_w, self.embed_dims, device, dtype)
            self._cached_device = device
            self._cached_dtype = dtype
        
        # 扩展到当前batch size
        # _bev_pos_cache: [H*W, 1, C] -> [H*W, bs, C]
        bev_pos = self._bev_pos_cache.expand(-1, bs, -1)
        return bev_pos
    
    def _identity_dropout(self, identity: torch.Tensor, keep: float):
        """对 identity 做轻量 Dropout（训练期），保持期望幅度不变。
        identity: [B, H*W, C]"""
        if (not self.training) or keep >= 1.0:
            return identity
        B, N, _ = identity.shape
        m = (torch.rand(B, N, 1, device=identity.device, dtype=identity.dtype) < keep).float() / keep
        return identity * m


@ATTENTION.register_module()
class DeformableCrossAttention_Conf(nn.Module):
    """
    基于Deformable Attention的跨模态注意力模块
    参考TemporalCrossAttention实现，适配跨模态场景
    """
    
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=1,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 **kwargs):
        super().__init__()
        
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.im2col_step = im2col_step
        
        self.head_dims = embed_dims // num_heads
        
        # Check if head_dims is power of 2 (for efficiency)
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                return False
            return (n & (n - 1) == 0) and n != 0
        
        if not _is_power_of_2(self.head_dims):
            warnings.warn(
                "You'd better set embed_dims to make the dimension of each "
                "attention head a power of 2 which is more efficient in CUDA implementation.")
        
        # Deformable attention参数
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        """初始化权重，参考TemporalCrossAttention"""
        # 初始化sampling offsets
        nn.init.constant_(self.sampling_offsets.weight, 0)
        
        # 初始化偏移的bias，使用圆形模式
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.view(-1))
        
        # 初始化attention weights
        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)
        
        # 初始化projection layers
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)
    
    def forward(self, query, key, value, reference_points, spatial_shapes, 
                level_start_index, bev_pos=None, value_confidence=None, **kwargs):
        """
        前向传播
        
        Args:
            query (Tensor): [B, N_q, C]
            key (Tensor): [B, N_k, C] 
            value (Tensor): [B, N_k, C]
            reference_points (Tensor): [B, N_q, num_levels, 2]
            spatial_shapes (Tensor): [num_levels, 2]
            level_start_index (Tensor): [num_levels]
            bev_pos (Tensor): [H*W, bs, C]
        """
        bs, num_query, _ = query.shape
        _, num_value, _ = value.shape
        
        identity = query
        
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        
        # Value projection
        value = self.value_proj(value)
        value = value.reshape(bs, num_value, self.num_heads, self.head_dims)
        
        # 生成sampling offsets和attention weights
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points)
        
        # 计算sampling locations
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + \
                sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + \
                sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be 2 or 4, '
                f'but get {reference_points.shape[-1]} instead.')
        
        if (value_confidence is not None):
            # sampling_locations: [B, Nq, Nh, L, P, 2] in [0,1]
            # print('Debug: using value_confidence in DeformableCrossAttention_Conf')
            grid = sampling_locations * 2 - 1
            B, Nq, Nh, L, P, _ = sampling_locations.shape
            grid = (sampling_locations * 2.0 - 1.0).reshape(B, Nq * Nh * L, P, 2)
            # sample confidence at sampling points
            Wm = F.grid_sample(
                value_confidence, grid,
                mode='bilinear', padding_mode='zeros', align_corners=False
            )
            # 还原成 [B,1,Nq,Nh,L,P]，与 attention_weights 对齐
            Wm = Wm.view(B, 1, Nq, Nh, L, P)
            aw = attention_weights * Wm
            den = aw.sum(dim=(3,4), keepdim=True).clamp_min(1e-6)
            attention_weights = aw / den
        
        # 执行deformable attention
        if torch.cuda.is_available() and value.is_cuda:
            # 使用CUDA实现
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
                
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            # 使用PyTorch实现
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        
        # Output projection
        output = self.output_proj(output)
        
        return self.dropout(output) + identity
    
class FFN(nn.Module):
    """前馈网络模块"""
    
    def __init__(self, embed_dims, feedforward_channels, dropout=0.1):
        super().__init__()
        
        self.fc1 = nn.Linear(embed_dims, feedforward_channels)
        self.fc2 = nn.Linear(feedforward_channels, embed_dims)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))