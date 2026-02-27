import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import build_positional_encoding, build_transformer_layer_sequence
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import auto_fp16
import math

class LayerNorm2d(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class LongTermPredictor(BaseModule):
    """Implements long-term temporal predictor for BEV features.
    
    This predictor uses historical frames to predict current frame features,
    suitable for temporal prediction training with current frame as supervision.
    
    Args:
        in_channels (int): Input feature channels. Defaults: 256.
        out_channels (int): Output feature channels. Defaults: 256.
        embed_dims (int): The feature dimension. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs. Defaults: 512.
        num_history (int): Number of historical frames to use. Defaults: 4.
        reduction (int): Dimension reduction factor for efficiency. Defaults: 4.
        with_query (bool): Whether to use learnable queries. Defaults: True.
        with_sin_embedding (bool): Whether to use sinusoidal time embedding. Defaults: False.
        decoder (dict): Configuration for the transformer decoder.
        bev_h (int): Height of BEV grid. Defaults: 200.
        bev_w (int): Width of BEV grid. Defaults: 200.
        positional_encoding_time (dict): Time positional encoding config.
        positional_encoding_spatial (dict): Spatial positional encoding config.
        loss_weight (float): Weight for temporal prediction loss. Defaults: 1.0.
    """
    
    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 embed_dims=256,
                 feedforward_channels=512,
                 num_history=4,
                 reduction=4,
                 with_query=True,
                 with_sin_embedding=False,
                 decoder=None,
                 bev_h=200,
                 bev_w=200,
                 positional_encoding_time=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 positional_encoding_spatial=dict(
                     type='SinePositionalEncoding',
                     num_feats=32,
                     normalize=True),
                 loss_weight=1.0,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        
        # Update positional encoding dimensions
        positional_encoding_time['num_feats'] = embed_dims // 2
        positional_encoding_spatial['num_feats'] = embed_dims // 2 // reduction
        
        # Store basic parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_history = num_history
        self.reduction = reduction
        self.with_query = with_query
        self.with_sin_embedding = with_sin_embedding
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.loss_weight = loss_weight
        
        # Build transformer decoder
        self.decoder = build_transformer_layer_sequence(decoder)
        
        # Build positional encodings
        self.positional_encoding_spatial = build_positional_encoding(
            positional_encoding_spatial)
        
        # Time embedding components
        if self.with_sin_embedding:
            self.frame_embeds = build_positional_encoding(positional_encoding_time)
        else:
            # Learnable frame embeddings for each historical time step
            self.frame_embeds = nn.Parameter(torch.Tensor(
                self.num_history, self.embed_dims // self.reduction))
        
        # Query embeddings for current frame prediction
        if self.with_query:
            self.queries = nn.Embedding(
                bev_h * bev_w, self.embed_dims // self.reduction)
        
        # Feature projection layers
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dims, embed_dims // self.reduction),
            nn.LayerNorm(embed_dims // self.reduction)
        )
        
        # Output projection to match target dimensions
        self.output_proj = nn.Sequential(
            nn.Conv2d(embed_dims // self.reduction, out_channels, 
                     kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channels)
        )
        
        # Input dimension adaptation if needed
        if self.in_channels != self.embed_dims:
            self.input_adapter = nn.Sequential(
                nn.Conv2d(in_channels, embed_dims, kernel_size=1, bias=False),
                LayerNorm2d(embed_dims)
            )
        
        # Loss function for temporal prediction
        self.criterion = nn.MSELoss()
    
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Initialize transformer layers
        for m in self.modules():
            if 'TemporalCrossAttention' in m.__class__.__name__:
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        
        # Initialize frame embeddings if not using sinusoidal
        if not self.with_sin_embedding:
            nn.init.normal_(self.frame_embeds)
    
    @auto_fp16(apply_to=('historical_feats', 'current_feat'))
    def forward(self, historical_feats, current_feat=None, return_loss=True, **kwargs):
        """
        Forward pass for temporal prediction.
        
        Args:
            historical_feats (list[Tensor] or Tensor): Historical BEV features.
                If list: [T-1, T-2, T-3, ..., T-num_history]
                If Tensor: (bs, num_history, C, H, W)
            current_feat (Tensor, optional): Current frame GT for training.
                Shape: (bs, C, H, W). Required when return_loss=True.
            return_loss (bool): Whether to compute loss. Defaults: True.
        
        Returns:
            dict: Contains 'pred_current' and optionally 'loss_temporal'.
        """
        # Handle input format
        if isinstance(historical_feats, list):
            assert len(historical_feats) == self.num_history, \
                f"Expected {self.num_history} historical frames, got {len(historical_feats)}"
            historical_feats = torch.stack(historical_feats, dim=1)  # (bs, T, C, H, W)
        
        bs, num_frames, C, bev_h, bev_w = historical_feats.shape
        assert num_frames == self.num_history, \
            f"Expected {self.num_history} frames, got {num_frames}"
        
        dtype = historical_feats.dtype
        device = historical_feats.device
        
        # Generate spatial positional encoding
        bev_mask = torch.zeros((bs, bev_h, bev_w), device=device).to(dtype)
        bev_pos = self.positional_encoding_spatial(bev_mask).to(dtype)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)  # (H*W, bs, C//reduction)
        
        # Adapt input dimensions if necessary
        if self.in_channels != self.embed_dims:
            historical_feats = historical_feats.reshape(bs * num_frames, C, bev_h, bev_w)
            historical_feats = self.input_adapter(historical_feats)
            C = self.embed_dims
            historical_feats = historical_feats.reshape(bs, num_frames, C, bev_h, bev_w)
        
        # Process temporal features with embeddings
        temporal_values = []
        
        if self.with_sin_embedding:
            # Generate sinusoidal time embeddings for historical frames
            # Use negative time steps: [-1, -2, -3, ..., -num_history]
            time_steps = torch.arange(-1, -(self.num_history + 1), -1, device=device).float()
            time_mask = time_steps.unsqueeze(0).unsqueeze(2).repeat(bs, 1, 1)  # (bs, T, 1)
            time_pos = self.frame_embeds(time_mask).to(dtype).squeeze(-1)
            time_pos = time_pos.permute(2, 0, 1)  # (embed_dims//2, bs, T)
            
            for i in range(num_frames):
                temporal_values.append(historical_feats[:, i])  # (bs, C, H, W)
        else:
            # Use learnable frame embeddings
            for i in range(num_frames):
                frame_embed = self.frame_embeds[i].reshape(1, -1, 1, 1).to(dtype)
                temporal_values.append(historical_feats[:, i])  # Don't add embedding here, add later
        
        # Stack and reshape for transformer
        # (bs, T, C, H, W) -> (H*W, bs*T, C//reduction)
        stacked_values = torch.stack(temporal_values, dim=1)  # (bs, T, C, H, W)
        stacked_values = stacked_values.reshape(bs * num_frames, C, bev_h * bev_w)
        stacked_values = stacked_values.permute(2, 0, 1)  # (H*W, bs*T, C)
        
        # Apply input projection for dimension reduction
        stacked_values = self.input_proj(stacked_values)  # (H*W, bs*T, C//reduction)
        
        # Add temporal embeddings
        if self.with_sin_embedding:
            # Expand time embeddings to spatial dimensions
            time_embeds = time_pos.unsqueeze(0).repeat(bev_h * bev_w, 1, 1)
            time_embeds = time_embeds.reshape(bev_h * bev_w, bs * num_frames, -1)
            stacked_values = stacked_values + time_embeds
        else:
            # Add learnable frame embeddings
            frame_embeds = self.frame_embeds.unsqueeze(0).repeat(bev_h * bev_w, 1, 1)
            frame_embeds = frame_embeds.unsqueeze(1).repeat(1, bs, 1, 1)
            frame_embeds = frame_embeds.reshape(bev_h * bev_w, bs * num_frames, -1)
            stacked_values = stacked_values + frame_embeds
        
        # Prepare queries for current frame prediction
        if self.with_query:
            # Use learnable queries
            bev_queries = self.queries.weight.to(dtype).unsqueeze(1).repeat(1, bs, 1)
            # Shape: (H*W, bs, C//reduction)
        else:
            # Use most recent frame (T-1) as query
            recent_frame = stacked_values[:, :bs, :]  # (H*W, bs, C//reduction)
            bev_queries = recent_frame
        
        # Add current time embedding to queries (time = 0)
        if self.with_sin_embedding:
            current_time = torch.zeros((bs, 1, 1), device=device).to(dtype)
            current_time_pos = self.frame_embeds(current_time).to(dtype).squeeze(-1)
            current_time_pos = current_time_pos.permute(2, 0, 1)  # (embed_dims//2, bs, 1)
            current_time_embed = current_time_pos.squeeze(-1).unsqueeze(0).repeat(bev_h * bev_w, 1, 1)
            bev_queries = bev_queries + current_time_embed
        
        # Apply transformer decoder to predict current frame
        pred_current = self.decoder(
            bev_queries,           # queries for current frame: (H*W, bs, C//reduction)
            stacked_values,        # keys from historical frames: (H*W, bs*T, C//reduction) 
            stacked_values,        # values from historical frames: (H*W, bs*T, C//reduction)
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            prev_bev=stacked_values,
            **kwargs
        )
        
        # Reshape to spatial format
        pred_current = pred_current.reshape(bs, bev_h, bev_w, self.embed_dims // self.reduction)
        pred_current = pred_current.permute(0, 3, 1, 2)  # (bs, C//reduction, H, W)
        
        # Apply output projection
        pred_current = self.output_proj(pred_current)  # (bs, out_channels, H, W)

        pred_current_flat = pred_current.permute(0, 2, 3, 1).contiguous().view(bs, bev_h * bev_w, self.out_channels)
        
        # Prepare return dictionary
        results = {'pred_current': pred_current, 'pred_current_flat': pred_current_flat}
        
        # Compute loss if ground truth is provided
        if return_loss and self.training:
            assert current_feat is not None, "current_feat must be provided for loss computation"
            loss_temporal = self.criterion(pred_current, current_feat) * self.loss_weight
            results['loss_temporal'] = loss_temporal
        return results
    
    def predict(self, historical_feats, **kwargs):
        """
        Inference mode: predict current frame from historical frames.
        
        Args:
            historical_feats (list[Tensor] or Tensor): Historical BEV features.
        
        Returns:
            Tensor: Predicted current frame features.
        """
        with torch.no_grad():
            results = self.forward(historical_feats, current_feat=None, 
                                 return_loss=False, **kwargs)
            return results['pred_current_flat']
    
    def compute_temporal_loss(self, pred_current, gt_current):
        """
        Compute temporal prediction loss.
        
        Args:
            pred_current (Tensor): Predicted current frame.
            gt_current (Tensor): Ground truth current frame.
        
        Returns:
            Tensor: Temporal prediction loss.
        """
        return self.criterion(pred_current, gt_current) * self.loss_weight
    
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class TemporalDecoder(TransformerLayerSequence):
    def __init__(self, *args, **kwargs):
        super(TemporalDecoder, self).__init__(*args, **kwargs)

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='2d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        if dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = bev_query
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        prev_bev = prev_bev.permute(1, 0, 2)

        bs, len_bev, num_bev_level, _ = ref_2d.shape
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=ref_2d,
                ref_3d=None,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=None,
                level_start_index=None,
                reference_points_cam=None,
                bev_mask=None,
                prev_bev=prev_bev,
                **kwargs)
            bev_query = output
        return output


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class LongTermPredictorRes(BaseModule):
    """Implements long-term temporal predictor with residual connection for BEV features.
    
    This predictor uses historical frames to predict current frame features,
    suitable for temporal prediction training with current frame as supervision.
    
    Args:
        in_channels (int): Input feature channels. Defaults: 256.
        out_channels (int): Output feature channels. Defaults: 256.
        embed_dims (int): The feature dimension. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs. Defaults: 512.
        num_history (int): Number of historical frames to use. Defaults: 4.
        reduction (int): Dimension reduction factor for efficiency. Defaults: 4.
        with_query (bool): Whether to use learnable queries. Defaults: True.
        with_sin_embedding (bool): Whether to use sinusoidal time embedding. Defaults: False.
        decoder (dict): Configuration for the transformer decoder.
        bev_h (int): Height of BEV grid. Defaults: 200.
        bev_w (int): Width of BEV grid. Defaults: 200.
        positional_encoding_time (dict): Time positional encoding config.
        positional_encoding_spatial (dict): Spatial positional encoding config.
        loss_weight (float): Weight for temporal prediction loss. Defaults: 1.0.
    """
    
    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 embed_dims=256,
                 feedforward_channels=512,
                 num_history=4,
                 reduction=4,
                 with_query=True,
                 with_sin_embedding=False,
                 decoder=None,
                 bev_h=200,
                 bev_w=200,
                 positional_encoding_time=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 positional_encoding_spatial=dict(
                     type='SinePositionalEncoding',
                     num_feats=32,
                     normalize=True),
                 loss_weight=1.0,
                 init_cfg=None,
                 with_residual=True,           # 新增：是否使用残差连接
                 residual_weight=1.0,          # 新增：残差权重
                 res_post_proc=True,            # 新增：是否在输出层使用归一化
                 **kwargs):
        super().__init__(init_cfg)
        
        # Update positional encoding dimensions
        positional_encoding_time['num_feats'] = embed_dims // 2
        positional_encoding_spatial['num_feats'] = embed_dims // 2 // reduction
        
        # Store basic parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_history = num_history
        self.reduction = reduction
        self.with_query = with_query
        self.with_sin_embedding = with_sin_embedding
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.loss_weight = loss_weight
        
        self.with_residual = with_residual
        self.residual_weight = residual_weight
        
        # Build transformer decoder
        self.decoder = build_transformer_layer_sequence(decoder)
        
        # Build positional encodings
        self.positional_encoding_spatial = build_positional_encoding(
            positional_encoding_spatial)
        
        # Time embedding components
        if self.with_sin_embedding:
            self.frame_embeds = build_positional_encoding(positional_encoding_time)
        else:
            # Learnable frame embeddings for each historical time step
            self.frame_embeds = nn.Parameter(torch.Tensor(
                self.num_history, self.embed_dims // self.reduction))
        
        # Query embeddings for current frame prediction
        if self.with_query:
            self.queries = nn.Embedding(
                bev_h * bev_w, self.embed_dims // self.reduction)
        
        # Feature projection layers
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dims, embed_dims // self.reduction),
            nn.LayerNorm(embed_dims // self.reduction)
        )
        
        # Output projection to match target dimensions
        self.output_proj = nn.Sequential(
            nn.Conv2d(embed_dims // self.reduction, out_channels, 
                     kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channels)
        )
        
        # Input dimension adaptation if needed
        if self.in_channels != self.embed_dims:
            self.input_adapter = nn.Sequential(
                nn.Conv2d(in_channels, embed_dims, kernel_size=1, bias=False),
                LayerNorm2d(embed_dims)
            )
            
        if self.with_residual and self.in_channels != self.out_channels:
            self.residual_adapter = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(out_channels)
            )
        
        if res_post_proc:
            self.output_post_process = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(out_channels),
            )
        
        # Loss function for temporal prediction
        self.criterion = nn.MSELoss()
    
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Initialize transformer layers
        for m in self.modules():
            if 'TemporalCrossAttention' in m.__class__.__name__:
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        
        # Initialize frame embeddings if not using sinusoidal
        if not self.with_sin_embedding:
            nn.init.normal_(self.frame_embeds)
    
    @auto_fp16(apply_to=('historical_feats', 'current_feat'))
    def forward(self, historical_feats, current_feat=None, return_loss=True, **kwargs):
        """
        Forward pass for temporal prediction.
        
        Args:
            historical_feats (list[Tensor] or Tensor): Historical BEV features.
                If list: [T-1, T-2, T-3, ..., T-num_history]
                If Tensor: (bs, num_history, C, H, W)
            current_feat (Tensor, optional): Current frame GT for training.
                Shape: (bs, C, H, W). Required when return_loss=True.
            return_loss (bool): Whether to compute loss. Defaults: True.
        
        Returns:
            dict: Contains 'pred_current' and optionally 'loss_temporal'.
        """
        # Handle input format
        if isinstance(historical_feats, list):
            assert len(historical_feats) == self.num_history, \
                f"Expected {self.num_history} historical frames, got {len(historical_feats)}"
            historical_feats = torch.stack(historical_feats, dim=1)  # (bs, T, C, H, W)
        
        bs, num_frames, C, bev_h, bev_w = historical_feats.shape
        assert num_frames == self.num_history, \
            f"Expected {self.num_history} frames, got {num_frames}"
        
        dtype = historical_feats.dtype
        device = historical_feats.device
        
        historical_feats_org = historical_feats.clone()  # 保存原始特征用于残差连接
        
        # Generate spatial positional encoding
        bev_mask = torch.zeros((bs, bev_h, bev_w), device=device).to(dtype)
        bev_pos = self.positional_encoding_spatial(bev_mask).to(dtype)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)  # (H*W, bs, C//reduction)
        
        # Adapt input dimensions if necessary
        if self.in_channels != self.embed_dims:
            historical_feats = historical_feats.reshape(bs * num_frames, C, bev_h, bev_w)
            historical_feats = self.input_adapter(historical_feats)
            C = self.embed_dims
            historical_feats = historical_feats.reshape(bs, num_frames, C, bev_h, bev_w)
        
        # Process temporal features with embeddings
        temporal_values = []
        
        if self.with_sin_embedding:
            # Generate sinusoidal time embeddings for historical frames
            # Use negative time steps: [-1, -2, -3, ..., -num_history]
            time_steps = torch.arange(-1, -(self.num_history + 1), -1, device=device).float()
            time_mask = time_steps.unsqueeze(0).unsqueeze(2).repeat(bs, 1, 1)  # (bs, T, 1)
            time_pos = self.frame_embeds(time_mask).to(dtype).squeeze(-1)
            time_pos = time_pos.permute(2, 0, 1)  # (embed_dims//2, bs, T)
            
            for i in range(num_frames):
                temporal_values.append(historical_feats[:, i])  # (bs, C, H, W)
        else:
            # Use learnable frame embeddings
            for i in range(num_frames):
                frame_embed = self.frame_embeds[i].reshape(1, -1, 1, 1).to(dtype)
                temporal_values.append(historical_feats[:, i])  # Don't add embedding here, add later
        
        # Stack and reshape for transformer
        # (bs, T, C, H, W) -> (H*W, bs*T, C//reduction)
        stacked_values = torch.stack(temporal_values, dim=1)  # (bs, T, C, H, W)
        stacked_values = stacked_values.reshape(bs * num_frames, C, bev_h * bev_w)
        stacked_values = stacked_values.permute(2, 0, 1)  # (H*W, bs*T, C)
        
        # Apply input projection for dimension reduction
        stacked_values = self.input_proj(stacked_values)  # (H*W, bs*T, C//reduction)
        
        # Add temporal embeddings
        if self.with_sin_embedding:
            # Expand time embeddings to spatial dimensions
            time_embeds = time_pos.unsqueeze(0).repeat(bev_h * bev_w, 1, 1)
            time_embeds = time_embeds.reshape(bev_h * bev_w, bs * num_frames, -1)
            stacked_values = stacked_values + time_embeds
        else:
            # Add learnable frame embeddings
            frame_embeds = self.frame_embeds.unsqueeze(0).repeat(bev_h * bev_w, 1, 1)
            frame_embeds = frame_embeds.unsqueeze(1).repeat(1, bs, 1, 1)
            frame_embeds = frame_embeds.reshape(bev_h * bev_w, bs * num_frames, -1)
            stacked_values = stacked_values + frame_embeds
        
        # Prepare queries for current frame prediction
        if self.with_query:
            # Use learnable queries
            bev_queries = self.queries.weight.to(dtype).unsqueeze(1).repeat(1, bs, 1)
            # Shape: (H*W, bs, C//reduction)
        else:
            # Use most recent frame (T-1) as query
            recent_frame = stacked_values[:, :bs, :]  # (H*W, bs, C//reduction)
            bev_queries = recent_frame
        
        # Add current time embedding to queries (time = 0)
        if self.with_sin_embedding:
            current_time = torch.zeros((bs, 1, 1), device=device).to(dtype)
            current_time_pos = self.frame_embeds(current_time).to(dtype).squeeze(-1)
            current_time_pos = current_time_pos.permute(2, 0, 1)  # (embed_dims//2, bs, 1)
            current_time_embed = current_time_pos.squeeze(-1).unsqueeze(0).repeat(bev_h * bev_w, 1, 1)
            bev_queries = bev_queries + current_time_embed
        
        # Apply transformer decoder to predict current frame
        pred_current = self.decoder(
            bev_queries,           # queries for current frame: (H*W, bs, C//reduction)
            stacked_values,        # keys from historical frames: (H*W, bs*T, C//reduction) 
            stacked_values,        # values from historical frames: (H*W, bs*T, C//reduction)
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            prev_bev=stacked_values,
            **kwargs
        )
        
        # Reshape to spatial format
        pred_current = pred_current.reshape(bs, bev_h, bev_w, self.embed_dims // self.reduction)
        pred_current = pred_current.permute(0, 3, 1, 2)  # (bs, C//reduction, H, W)
        
        # Apply output projection
        # pred_current = self.output_proj(pred_current)  # (bs, out_channels, H, W)
        residual_pred = self.output_proj(pred_current)  # (bs, out_channels, H, W)
        
        # 残差机制：pred_current = latest_history + residual_pred
        if self.with_residual:
            # 获取最新历史帧作为基础
            latest_history = historical_feats_org[:, 0]  # (bs, C, H, W) - 最近帧
            # 如果维度不匹配，进行适配
            if hasattr(self, 'residual_adapter'):
                latest_history = self.residual_adapter(latest_history)
            # 残差连接
            pred_current = latest_history + self.residual_weight * residual_pred
        else:
            pred_current = residual_pred
        pred_current = self.output_post_process(pred_current)

        pred_current_flat = pred_current.permute(0, 2, 3, 1).contiguous().view(bs, bev_h * bev_w, self.out_channels)
        
        # Prepare return dictionary
        results = {'pred_current': pred_current, 'pred_current_flat': pred_current_flat}
        
        # Compute loss if ground truth is provided
        if return_loss and self.training:
            assert current_feat is not None, "current_feat must be provided for loss computation"
            loss_temporal = self.criterion(pred_current, current_feat) * self.loss_weight
            results['loss_temporal'] = loss_temporal
        return results
    
    def predict(self, historical_feats, **kwargs):
        """
        Inference mode: predict current frame from historical frames.
        
        Args:
            historical_feats (list[Tensor] or Tensor): Historical BEV features.
        
        Returns:
            Tensor: Predicted current frame features.
        """
        with torch.no_grad():
            results = self.forward(historical_feats, current_feat=None, 
                                 return_loss=False, **kwargs)
            return results['pred_current_flat']
    
    def compute_temporal_loss(self, pred_current, gt_current):
        """
        Compute temporal prediction loss.
        
        Args:
            pred_current (Tensor): Predicted current frame.
            gt_current (Tensor): Ground truth current frame.
        
        Returns:
            Tensor: Temporal prediction loss.
        """
        return self.criterion(pred_current, gt_current) * self.loss_weight
    

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class LongTermPredictorResUncertainty(BaseModule):
    """Implements long-term temporal predictor with residual connection for BEV features.
    
    This predictor uses historical frames to predict current frame features,
    suitable for temporal prediction training with current frame as supervision.
    
    Args:
        in_channels (int): Input feature channels. Defaults: 256.
        out_channels (int): Output feature channels. Defaults: 256.
        embed_dims (int): The feature dimension. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs. Defaults: 512.
        num_history (int): Number of historical frames to use. Defaults: 4.
        reduction (int): Dimension reduction factor for efficiency. Defaults: 4.
        with_query (bool): Whether to use learnable queries. Defaults: True.
        with_sin_embedding (bool): Whether to use sinusoidal time embedding. Defaults: False.
        decoder (dict): Configuration for the transformer decoder.
        bev_h (int): Height of BEV grid. Defaults: 200.
        bev_w (int): Width of BEV grid. Defaults: 200.
        positional_encoding_time (dict): Time positional encoding config.
        positional_encoding_spatial (dict): Spatial positional encoding config.
        loss_weight (float): Weight for temporal prediction loss. Defaults: 1.0.
    """
    
    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 embed_dims=256,
                 feedforward_channels=512,
                 num_history=4,
                 reduction=4,
                 with_query=True,
                 with_sin_embedding=False,
                 decoder=None,
                 bev_h=200,
                 bev_w=200,
                 positional_encoding_time=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 positional_encoding_spatial=dict(
                     type='SinePositionalEncoding',
                     num_feats=32,
                     normalize=True),
                 loss_weight=1.0,
                 init_cfg=None,
                 with_residual=True,           # 新增：是否使用残差连接
                 residual_weight=1.0,          # 新增：残差权重
                 res_post_proc=True,            # 新增：是否在输出层使用归一化
                 **kwargs):
        super().__init__(init_cfg)
        
        # Update positional encoding dimensions
        positional_encoding_time['num_feats'] = embed_dims // 2
        positional_encoding_spatial['num_feats'] = embed_dims // 2 // reduction
        
        # Store basic parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_history = num_history
        self.reduction = reduction
        self.with_query = with_query
        self.with_sin_embedding = with_sin_embedding
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.loss_weight = loss_weight
        
        self.with_residual = with_residual
        self.residual_weight = residual_weight
        
        # Build transformer decoder
        self.decoder = build_transformer_layer_sequence(decoder)
        
        # Build positional encodings
        self.positional_encoding_spatial = build_positional_encoding(
            positional_encoding_spatial)
        
        # Time embedding components
        if self.with_sin_embedding:
            self.frame_embeds = build_positional_encoding(positional_encoding_time)
        else:
            # Learnable frame embeddings for each historical time step
            self.frame_embeds = nn.Parameter(torch.Tensor(
                self.num_history, self.embed_dims // self.reduction))
        
        # Query embeddings for current frame prediction
        if self.with_query:
            self.queries = nn.Embedding(
                bev_h * bev_w, self.embed_dims // self.reduction)
        
        # Feature projection layers
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dims, embed_dims // self.reduction),
            nn.LayerNorm(embed_dims // self.reduction)
        )
        
        # Output projection to match target dimensions
        self.output_proj = nn.Sequential(
            nn.Conv2d(embed_dims // self.reduction, out_channels, 
                     kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channels)
        )
        
        # Input dimension adaptation if needed
        if self.in_channels != self.embed_dims:
            self.input_adapter = nn.Sequential(
                nn.Conv2d(in_channels, embed_dims, kernel_size=1, bias=False),
                LayerNorm2d(embed_dims)
            )
            
        if self.with_residual and self.in_channels != self.out_channels:
            self.residual_adapter = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(out_channels)
            )
        
        if res_post_proc:
            self.output_post_process = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(out_channels),
            )
        
        self.var_head = nn.Conv2d(out_channels, 1, kernel_size=1)

        # Loss function for temporal prediction
        self.criterion = nn.MSELoss()
    
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Initialize transformer layers
        for m in self.modules():
            if 'TemporalCrossAttention' in m.__class__.__name__:
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        
        # Initialize frame embeddings if not using sinusoidal
        if not self.with_sin_embedding:
            nn.init.normal_(self.frame_embeds)
        
        nn.init.zeros_(self.var_head.weight)
        nn.init.constant_(self.var_head.bias, -2.0)
    
    
    @auto_fp16(apply_to=('historical_feats', 'current_feat'))
    def forward(self, historical_feats, current_feat=None, return_loss=True, **kwargs):
        """
        Forward pass for temporal prediction.
        
        Args:
            historical_feats (list[Tensor] or Tensor): Historical BEV features.
                If list: [T-1, T-2, T-3, ..., T-num_history]
                If Tensor: (bs, num_history, C, H, W)
            current_feat (Tensor, optional): Current frame GT for training.
                Shape: (bs, C, H, W). Required when return_loss=True.
            return_loss (bool): Whether to compute loss. Defaults: True.
        
        Returns:
            dict: Contains 'pred_current' and optionally 'loss_temporal'.
        """
        # Handle input format
        if isinstance(historical_feats, list):
            assert len(historical_feats) == self.num_history, \
                f"Expected {self.num_history} historical frames, got {len(historical_feats)}"
            historical_feats = torch.stack(historical_feats, dim=1)  # (bs, T, C, H, W)
        
        bs, num_frames, C, bev_h, bev_w = historical_feats.shape
        assert num_frames == self.num_history, \
            f"Expected {self.num_history} frames, got {num_frames}"
        
        dtype = historical_feats.dtype
        device = historical_feats.device
        
        historical_feats_org = historical_feats.clone()  # 保存原始特征用于残差连接
        
        # Generate spatial positional encoding
        bev_mask = torch.zeros((bs, bev_h, bev_w), device=device).to(dtype)
        bev_pos = self.positional_encoding_spatial(bev_mask).to(dtype)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)  # (H*W, bs, C//reduction)
        
        # Adapt input dimensions if necessary
        if self.in_channels != self.embed_dims:
            historical_feats = historical_feats.reshape(bs * num_frames, C, bev_h, bev_w)
            historical_feats = self.input_adapter(historical_feats)
            C = self.embed_dims
            historical_feats = historical_feats.reshape(bs, num_frames, C, bev_h, bev_w)
        
        # Process temporal features with embeddings
        temporal_values = []
        
        if self.with_sin_embedding:
            # Generate sinusoidal time embeddings for historical frames
            # Use negative time steps: [-1, -2, -3, ..., -num_history]
            time_steps = torch.arange(-1, -(self.num_history + 1), -1, device=device).float()
            time_mask = time_steps.unsqueeze(0).unsqueeze(2).repeat(bs, 1, 1)  # (bs, T, 1)
            time_pos = self.frame_embeds(time_mask).to(dtype).squeeze(-1)
            time_pos = time_pos.permute(2, 0, 1)  # (embed_dims//2, bs, T)
            
            for i in range(num_frames):
                temporal_values.append(historical_feats[:, i])  # (bs, C, H, W)
        else:
            # Use learnable frame embeddings
            for i in range(num_frames):
                frame_embed = self.frame_embeds[i].reshape(1, -1, 1, 1).to(dtype)
                temporal_values.append(historical_feats[:, i])  # Don't add embedding here, add later
        
        # Stack and reshape for transformer
        # (bs, T, C, H, W) -> (H*W, bs*T, C//reduction)
        stacked_values = torch.stack(temporal_values, dim=1)  # (bs, T, C, H, W)
        stacked_values = stacked_values.reshape(bs * num_frames, C, bev_h * bev_w)
        stacked_values = stacked_values.permute(2, 0, 1)  # (H*W, bs*T, C)
        
        # Apply input projection for dimension reduction
        stacked_values = self.input_proj(stacked_values)  # (H*W, bs*T, C//reduction)
        
        # Add temporal embeddings
        if self.with_sin_embedding:
            # Expand time embeddings to spatial dimensions
            time_embeds = time_pos.unsqueeze(0).repeat(bev_h * bev_w, 1, 1)
            time_embeds = time_embeds.reshape(bev_h * bev_w, bs * num_frames, -1)
            stacked_values = stacked_values + time_embeds
        else:
            # Add learnable frame embeddings
            frame_embeds = self.frame_embeds.unsqueeze(0).repeat(bev_h * bev_w, 1, 1)
            frame_embeds = frame_embeds.unsqueeze(1).repeat(1, bs, 1, 1)
            frame_embeds = frame_embeds.reshape(bev_h * bev_w, bs * num_frames, -1)
            stacked_values = stacked_values + frame_embeds
        
        # Prepare queries for current frame prediction
        if self.with_query:
            # Use learnable queries
            bev_queries = self.queries.weight.to(dtype).unsqueeze(1).repeat(1, bs, 1)
            # Shape: (H*W, bs, C//reduction)
        else:
            # Use most recent frame (T-1) as query
            recent_frame = stacked_values[:, :bs, :]  # (H*W, bs, C//reduction)
            bev_queries = recent_frame
        
        # Add current time embedding to queries (time = 0)
        if self.with_sin_embedding:
            current_time = torch.zeros((bs, 1, 1), device=device).to(dtype)
            current_time_pos = self.frame_embeds(current_time).to(dtype).squeeze(-1)
            current_time_pos = current_time_pos.permute(2, 0, 1)  # (embed_dims//2, bs, 1)
            current_time_embed = current_time_pos.squeeze(-1).unsqueeze(0).repeat(bev_h * bev_w, 1, 1)
            bev_queries = bev_queries + current_time_embed
        
        # Apply transformer decoder to predict current frame
        pred_current = self.decoder(
            bev_queries,           # queries for current frame: (H*W, bs, C//reduction)
            stacked_values,        # keys from historical frames: (H*W, bs*T, C//reduction) 
            stacked_values,        # values from historical frames: (H*W, bs*T, C//reduction)
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            prev_bev=stacked_values,
            **kwargs
        )
        
        # Reshape to spatial format
        pred_current = pred_current.reshape(bs, bev_h, bev_w, self.embed_dims // self.reduction)
        pred_current = pred_current.permute(0, 3, 1, 2)  # (bs, C//reduction, H, W)
        
        # Apply output projection
        # pred_current = self.output_proj(pred_current)  # (bs, out_channels, H, W)
        residual_pred = self.output_proj(pred_current)  # (bs, out_channels, H, W)
        
        # 残差机制：pred_current = latest_history + residual_pred
        if self.with_residual:
            # 获取最新历史帧作为基础
            latest_history = historical_feats_org[:, 0]  # (bs, C, H, W) - 最近帧
            # 如果维度不匹配，进行适配
            if hasattr(self, 'residual_adapter'):
                latest_history = self.residual_adapter(latest_history)
            # 残差连接
            pred_current = latest_history + self.residual_weight * residual_pred
        else:
            pred_current = residual_pred
        pred_current = self.output_post_process(pred_current)
        
        # # debug replace pred_current with latest_history
        # pred_current = historical_feats_org[:, 0]  # (bs, C, H, W)

        pred_current_flat = pred_current.permute(0, 2, 3, 1).contiguous().view(bs, bev_h * bev_w, self.out_channels)
        
        # Prepare return dictionary
        results = {'pred_current': pred_current, 'pred_current_flat': pred_current_flat}
        
        log_sigma2  = self.var_head(pred_current)
        results['sigma2'] = torch.exp(log_sigma2.detach())

        # Compute loss if ground truth is provided
        if return_loss and self.training:
            assert current_feat is not None, "current_feat must be provided for loss computation"
            loss_reconstruction = self.criterion(pred_current, current_feat) * self.loss_weight
            loss_uncertainty = self.get_uncertainty_loss(pred_current, current_feat, log_sigma2)
            # print('Debug:',loss_reconstruction, loss_uncertainty)
            results['loss_temporal'] = loss_reconstruction + loss_uncertainty
        return results
    
    def predict(self, historical_feats, **kwargs):
        """
        Inference mode: predict current frame from historical frames.
        
        Args:
            historical_feats (list[Tensor] or Tensor): Historical BEV features.
        
        Returns:
            Tensor: Predicted current frame features.
        """
        with torch.no_grad():
            results = self.forward(historical_feats, current_feat=None, 
                                 return_loss=False, **kwargs)
            return results['pred_current_flat']
    
    def compute_temporal_loss(self, pred_current, gt_current):
        """
        Compute temporal prediction loss.
        
        Args:
            pred_current (Tensor): Predicted current frame.
            gt_current (Tensor): Ground truth current frame.
        
        Returns:
            Tensor: Temporal prediction loss.
        """
        return self.criterion(pred_current, gt_current) * self.loss_weight
    
    def get_uncertainty_loss(self, pred_current, current_feat, log_sigma2):
        """
        pred_current: [B,C,H,W]
        current_feat: [B,C,H,W]
        log_sigma2:   [B,1,H,W]  # 建议网络直接预测这个量
        """
        with torch.cuda.amp.autocast(enabled=False):
            pred32 = pred_current.float()
            tgt32  = current_feat.detach().float()
            ls2    = torch.clamp(log_sigma2.float(), min=-3.0, max=4.0)

            err = (pred32 - tgt32).pow(2).mean(dim=1, keepdim=True)  # [B,1,H,W]
            nll = 0.5 * (torch.exp(-ls2) * err + ls2 + math.log(2*math.pi))
            return nll.mean()