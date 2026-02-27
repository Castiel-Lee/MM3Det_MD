import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import build_positional_encoding, build_transformer_layer_sequence
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import auto_fp16
from .long_term_predictor import LayerNorm2d

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class LongTermPredictor_CrossModality(BaseModule):
    """Implements long-term temporal predictor for BEV features.
    
    This predictor uses historical frames to predict current frame features,
    suitable for temporal prediction training with current frame as supervision.
    
    Args:
        in_channels (int): Input feature channels. Defaults: 256.
        out_channels (int): Output feature channels. Defaults: 256.
        embed_dims (int): The feature dimension. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs. Defaults: 512.
        num_history (int): Number of historical frames to use. Defaults: 4.
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
                 query_dim=None,
                 feedforward_channels=512,
                 num_history=4,
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
        positional_encoding_spatial['num_feats'] = embed_dims // 2
        
        # Store basic parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_history = num_history
        self.with_sin_embedding = with_sin_embedding
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.loss_weight = loss_weight
        self.query_dim = query_dim
        
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
                self.num_history, self.embed_dims))
        
        # Output projection to match target dimensions
        self.output_proj = nn.Sequential(
            nn.Conv2d(embed_dims, out_channels, 
                     kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channels)
        )
        
        # Input dimension adaptation if needed
        if self.in_channels != self.embed_dims:
            self.input_adapter = nn.Sequential(
                nn.Conv2d(in_channels, embed_dims, kernel_size=1, bias=False),
                LayerNorm2d(embed_dims)
            )
            
        if query_dim is not None and query_dim != self.in_channels:
            self.input_adapter_query = nn.Sequential(
                nn.Conv2d(query_dim, embed_dims, kernel_size=1, bias=False),
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
    
    @auto_fp16(apply_to=('historical_feats', 'current_feat', 'query_cross_modal'))
    def forward(self, historical_feats, query_cross_modal, current_feat=None, return_loss=True, **kwargs):
        """
        Forward pass for temporal prediction.
        
        Args:
            historical_feats (list[Tensor] or Tensor): Historical BEV features.
                If list: [T-1, T-2, T-3, ..., T-num_history]
                If Tensor: (bs, num_history, C, H, W)
            query_cross_modal (Tensor): Cross-modal query features for current frame prediction.
                Shape: (bs, C, H*W) or (bs, C, H, W) where C matches historical_feats channels
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
        
        # Handle query_cross_modal format - expect (bs, C, H*W) or (bs, C, H, W)
        original_C = historical_feats.shape[2]  # Get original channel dimension
        
        query_dim = self.query_dim if self.query_dim is not None else original_C
        if query_cross_modal.dim() == 4:  # (bs, C, H, W)
            assert query_cross_modal.shape == (bs, query_dim, bev_h, bev_w), \
                f"Expected query_cross_modal shape ({bs}, {query_dim}, {bev_h}, {bev_w}), got {query_cross_modal.shape}"
            query_cross_modal_reshaped = query_cross_modal.view(bs, query_dim, bev_h * bev_w)  # (bs, C, H*W)
        elif query_cross_modal.dim() == 3:  # (bs, C, H*W)
            assert query_cross_modal.shape == (bs, query_dim, bev_h * bev_w), \
                f"Expected query_cross_modal shape ({bs}, {query_dim}, {bev_h * bev_w}), got {query_cross_modal.shape}"
            query_cross_modal_reshaped = query_cross_modal
        else:
            raise ValueError(f"query_cross_modal should have 3 or 4 dimensions, got {query_cross_modal.dim()}")
        
        # Generate spatial positional encoding
        bev_mask = torch.zeros((bs, bev_h, bev_w), device=device).to(dtype)
        bev_pos = self.positional_encoding_spatial(bev_mask).to(dtype)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)  # (H*W, bs, embed_dims)
        
        # Adapt input dimensions if necessary for both historical_feats and query_cross_modal
        if self.in_channels != self.embed_dims:
            # Adapt historical features
            historical_feats = historical_feats.reshape(bs * num_frames, C, bev_h, bev_w)
            historical_feats = self.input_adapter(historical_feats)
            C = self.embed_dims
            historical_feats = historical_feats.reshape(bs, num_frames, C, bev_h, bev_w)
            
            # Adapt query_cross_modal using the same adapter
            query_cross_modal_spatial = query_cross_modal_reshaped.view(bs, query_dim, bev_h, bev_w)  # (bs, C, H, W)
            if hasattr(self, 'input_adapter_query'):
                query_cross_modal_spatial = self.input_adapter_query(query_cross_modal_spatial) # (bs, embed_dims, H, W)
            else:
                query_cross_modal_spatial = self.input_adapter(query_cross_modal_spatial)  # (bs, embed_dims, H, W)
            query_cross_modal_adapted = query_cross_modal_spatial.view(bs, self.embed_dims, bev_h * bev_w)  # (bs, embed_dims, H*W)
        else:
            query_cross_modal_adapted = query_cross_modal_reshaped  # (bs, C, H*W)
        
        # Convert query_cross_modal to transformer format: (H*W, bs, embed_dims)
        query_cross_modal_final = query_cross_modal_adapted.permute(2, 0, 1)  # (H*W, bs, embed_dims)
        
        # Process temporal features with embeddings
        temporal_values = []
        
        if self.with_sin_embedding:
            # Generate sinusoidal time embeddings for historical frames
            # Use negative time steps: [-1, -2, -3, ..., -num_history]
            time_steps = torch.arange(-1, -(self.num_history + 1), -1, device=device).float()
            time_mask = time_steps.unsqueeze(0).unsqueeze(2).repeat(bs, 1, 1)  # (bs, T, 1)
            time_pos = self.frame_embeds(time_mask).to(dtype).squeeze(-1)
            time_pos = time_pos.permute(2, 0, 1)  # (embed_dims, bs, T)
            
            for i in range(num_frames):
                temporal_values.append(historical_feats[:, i])  # (bs, C, H, W)
        else:
            # Use learnable frame embeddings
            for i in range(num_frames):
                temporal_values.append(historical_feats[:, i])  # (bs, C, H, W)
        
        # Stack and reshape for transformer
        # (bs, T, C, H, W) -> (H*W, bs*T, embed_dims)
        stacked_values = torch.stack(temporal_values, dim=1)  # (bs, T, C, H, W)
        stacked_values = stacked_values.reshape(bs * num_frames, C, bev_h * bev_w)
        stacked_values = stacked_values.permute(2, 0, 1)  # (H*W, bs*T, C)
        
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
        
        # Use provided cross-modal queries (already in correct format and dimension)
        bev_queries = query_cross_modal_final.to(dtype)  # (H*W, bs, embed_dims)
        
        # Add current time embedding to queries (time = 0)
        if self.with_sin_embedding:
            current_time = torch.zeros((bs, 1, 1), device=device).to(dtype)
            current_time_pos = self.frame_embeds(current_time).to(dtype).squeeze(-1)
            current_time_pos = current_time_pos.permute(2, 0, 1)  # (embed_dims, bs, 1)
            current_time_embed = current_time_pos.squeeze(-1).unsqueeze(0).repeat(bev_h * bev_w, 1, 1)
            bev_queries = bev_queries + current_time_embed
        
        # Apply transformer decoder to predict current frame
        pred_current = self.decoder(
            bev_queries,           # queries from cross-modal input: (H*W, bs, embed_dims)
            stacked_values,        # keys from historical frames: (H*W, bs*T, embed_dims) 
            stacked_values,        # values from historical frames: (H*W, bs*T, embed_dims)
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            prev_bev=stacked_values,
            **kwargs
        )
        
        # Reshape to spatial format
        pred_current = pred_current.reshape(bs, bev_h, bev_w, self.embed_dims)
        pred_current = pred_current.permute(0, 3, 1, 2)  # (bs, embed_dims, H, W)
        
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
    
    def predict(self, historical_feats, query_cross_modal, **kwargs):
        """
        Inference mode: predict current frame from historical frames.
        
        Args:
            historical_feats (list[Tensor] or Tensor): Historical BEV features.
            query_cross_modal (Tensor): Cross-modal query features.
        
        Returns:
            Tensor: Predicted current frame features.
        """
        with torch.no_grad():
            results = self.forward(historical_feats, query_cross_modal, 
                                 current_feat=None, return_loss=False, **kwargs)
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