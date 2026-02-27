import copy
import os.path as osp

import fontTools.ttLib
import mmcv.utils.path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_, constant_

from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate

from .spatial_cross_attention_img import MSDeformableAttention3DImg
from .spatial_cross_attention_pts import MSDeformableAttention3DPts
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from .decoder import CustomMSDeformableAttention
from mmcv.runner import force_fp32, auto_fp16

from .utils.feature_memory_reader import FeatureMemoryReader as UniBevFeatureMemoryReader

class ModalityProjectionModule(BaseModule):
    def __init__(self,
                 embed_dims,
                 with_norm = True,
                 with_residual = True,
                 ):
        super().__init__()
        layers = nn.ModuleList(
           [ nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True)]
        )
        if with_norm:
            layers.append(nn.LayerNorm(embed_dims))

        self.net = nn.Sequential(*layers)
        self.with_residual = with_residual
    def forward(self, x):
        out = self.net(x)
        if self.with_residual:
            return x + out
        else:
            return out

@TRANSFORMER.register_module()
class UniBEVTransformer_HistoryFeatPredictor_CrMdFusion_Uncertainty(BaseModule):
    """Implements the UniBEV transformer, based on Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 img_encoder=None,
                 pts_encoder=None,
                 history_decoder=None,
                 decoder=None,
                 embed_dims=256,
                 use_cams_embeds=True,
                 fusion_method='linear',
                 drop_modality = None,
                 feature_norm = None,
                 spatial_norm = None,
                 use_modal_embeds = None,
                 bev_h = 200,
                 bev_w = 200,
                 dual_queries = False,
                 vis_output = None,
                 cna_constant_init = None,
                 adj_num_frames = 6,
                 pts_feat_pred = True,
                 img_feat_pred = True,
                 use_cross_modal=True,
                 cross_modal_config=None,
                 pred_cur_fusion_method='learnable', # 'learnable' or 'hard'
                 cur_weight = 1.0,
                 CR_loss_type = 'mse',
                 **kwargs):
        super(UniBEVTransformer_HistoryFeatPredictor_CrMdFusion_Uncertainty, self).__init__(**kwargs)

        if img_encoder is not None:
            self.img_bev_encoder = build_transformer_layer_sequence(img_encoder)

        if pts_encoder is not None:
            self.pts_bev_encoder = build_transformer_layer_sequence(pts_encoder)
        
        self.adj_num_frames = adj_num_frames
        self.pts_feat_pred, self.img_feat_pred = pts_feat_pred, img_feat_pred
        assert history_decoder is not None, 'history_decoder must be provided'
        if self.pts_feat_pred:
            self.history_pts_feat_predictor = build_transformer_layer_sequence(history_decoder)
            self.pts_feat_memory_reader = UniBevFeatureMemoryReader()
        if self.img_feat_pred:
            self.history_img_feat_predictor = build_transformer_layer_sequence(history_decoder)
            self.img_feat_memory_reader = UniBevFeatureMemoryReader()
        
        self.use_cross_modal = use_cross_modal
        if self.use_cross_modal:
            from .cross_modal_module_uncertainty import CrossModalComplementaryUncertaintyModule
            self.cross_modal_module = CrossModalComplementaryUncertaintyModule(**cross_modal_config)
        
        # weight for the pred and current feature fusion of img and pts
        self.pred_cur_fusion_method = pred_cur_fusion_method
        self.img_pred_weights = nn.Parameter(torch.Tensor(1, embed_dims))  # 图像预测特征权重
        self.img_cur_weights = nn.Parameter(torch.Tensor(1, embed_dims))   # 图像当前特征权重
        self.pts_pred_weights = nn.Parameter(torch.Tensor(1, embed_dims))  # 点云预测特征权重
        self.pts_cur_weights = nn.Parameter(torch.Tensor(1, embed_dims))   # 点云当前特征权重
        self.fusion_norm_layer = nn.Softmax(dim=0)

        self.decoder = build_transformer_layer_sequence(decoder)

        self.dual_queries = dual_queries
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False
        self.cna_constant_norm = cna_constant_init
        self.bev_h = bev_h
        self.bev_w = bev_w

        self.use_cams_embeds = use_cams_embeds
        # self.finetune_init_value = finetune_init_value

        self.fusion_method = fusion_method
        self.cur_weight = cur_weight
        self.CR_loss_type = CR_loss_type

        if self.fusion_method == 'linear' or self.fusion_method == 'avg':
            self.scale_factor = 1
        elif self.fusion_method == 'cat':
            self.scale_factor = 2  # used to scale up the dimension when concatenate
        else:
            raise ValueError('Unrecognizable fusion method:{}'.format(self.fusion_method))

        self.drop_modality = drop_modality
        self.feature_norm = feature_norm
        self.spatial_norm = spatial_norm
        self.use_modal_embeds = use_modal_embeds
        self.two_stage_num_proposals = two_stage_num_proposals
        self.vis_output = vis_output    
        self.init_layers()
    
    def pred_cur_fusion(self, pred_feat, cur_feat, cur_valid=1, modal='img'):
        """
        use learnable weights to fuse predicted feature and current feature
        
        Args:
            pred_feat: pred feature [B, H*W, C]
            cur_feat: current feature [B, C, H, W] 
            cur_valid: valid flag of current feature (0 or 1)
            modal: modality ('img' or 'pts')
            
        Returns:
            fused feature [B, H*W, C]
        """

        if cur_feat is not None and cur_feat.dim() == 4:  # [B, C, H, W] -> [B, H*W, C]
            cur_feat = cur_feat.flatten(2).transpose(1, 2)
        
        if modal == 'img':
            pred_weights = self.img_pred_weights
            cur_weights = self.img_cur_weights
        elif modal == 'pts':
            pred_weights = self.pts_pred_weights
            cur_weights = self.pts_cur_weights
        else:
            raise ValueError(f"Unknown modal: {modal}")
        
        if cur_valid == 1:
            if self.pred_cur_fusion_method == 'learnable':
                weight_list = []
                weight_list.append(pred_weights)  # [1, C]
                weight_list.append(cur_weights)   # [1, C]
                fusion_weights = torch.cat(weight_list, dim=0)  # [2, C]

                normalized_weights = self.fusion_norm_layer(fusion_weights)  # [2, C]
                pred_norm_weights = normalized_weights[0]  # [C]
                cur_norm_weights = normalized_weights[1]   # [C]
                
                fused_feat = pred_feat * pred_norm_weights + cur_feat * cur_norm_weights
            elif self.pred_cur_fusion_method=='hard':
                fused_feat = self.cur_weight * cur_feat + (1 - self.cur_weight) * pred_feat
            else:
                raise ValueError(f"Unknown fusion method: {self.pred_cur_fusion_method}")
            
        else:
            if self.pred_cur_fusion_method=='learnable':
                pred_norm_weights = self.fusion_norm_layer(pred_weights)  # [1, C] -> [1, C]
                fused_feat = pred_feat * pred_norm_weights.squeeze(0)  # [C]
            elif self.pred_cur_fusion_method=='hard':
                fused_feat = pred_feat
            else:
                raise ValueError(f"Unknown fusion method: {self.pred_cur_fusion_method}")
        
        return fused_feat
    
    @property
    def with_img_bev_encoder(self):
        """bool: Whether the img_bev_encoder exists."""
        return hasattr(self, 'img_bev_encoder') and self.img_bev_encoder is not None

    @property
    def with_pts_bev_encoder(self):
        """bool: Whether the pts_bev_encoder exists."""
        return hasattr(self, 'pts_bev_encoder') and self.pts_bev_encoder is not None

    def init_layers(self):
        """Initialize layers of the UniBEVTransformer, based on DETR3D."""
        if self.feature_norm == 'ChannelNormWeights':
            self.feature_norm_layer = nn.Softmax(dim=0)
            self.pts_channel_weights = nn.Parameter(torch.Tensor(self.embed_dims))
            self.img_channel_weights = nn.Parameter(torch.Tensor(self.embed_dims))
        elif self.feature_norm == 'MLP_ChannelNormWeights':
            self.channel_weights_proj = nn.Sequential(
                nn.Linear(self.bev_h * self.bev_w * 2, 2),
                nn.ReLU(inplace=True))
        elif self.feature_norm == 'Leaky_ReLU_MLP_ChannelNormWeights':
            self.channel_weights_proj = nn.Sequential(
                nn.Linear(self.bev_h * self.bev_w * 2, 2),
                nn.LeakyReLU(inplace=True))
        elif self.feature_norm == 'ELU_MLP_ChannelNormWeights':
            self.channel_weights_proj = nn.Sequential(
                nn.Linear(self.bev_h * self.bev_w * 2, 2),
                nn.ELU(inplace=True))
        elif self.feature_norm == 'Sigmoid_MLP_ChannelNormWeights':
            self.channel_weights_proj = nn.Sequential(
                nn.Linear(self.bev_h * self.bev_w * 2, 2),
                nn.Sigmoid())
        elif self.feature_norm == 'ModalityProjection':
            assert self.fusion_method == 'cat'
            self.c_modal_proj = ModalityProjectionModule(self.embed_dims)
            self.l_modal_proj = ModalityProjectionModule(self.embed_dims)

        if self.spatial_norm == 'SpatialNormWeights':
            self.spatial_norm_layer = nn.Softmax(dim=0)
            self.pts_spatial_weights = nn.Parameter(torch.Tensor(self.bev_h*self.bev_w))
            self.img_spatial_weights = nn.Parameter(torch.Tensor(self.bev_h*self.bev_w))

        if self.with_img_bev_encoder:
            self.img_level_embeds = nn.Parameter(torch.Tensor(
                self.num_feature_levels, self.embed_dims))
            self.cams_embeds = nn.Parameter(
                torch.Tensor(self.num_cams, self.embed_dims))

        if self.with_pts_bev_encoder:
            self.pts_level_embeds = nn.Parameter(torch.Tensor(
                self.num_feature_levels, self.embed_dims))

        if self.use_modal_embeds == 'MLP':
            self.modal_embbeding_mlp = nn.Sequential(
                nn.Linear(2, self.embed_dims // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims // 2, self.embed_dims),
                nn.ReLU(inplace=True))
        elif self.use_modal_embeds == 'Fixed':
            self.modal_embbeding_C = nn.Parameter(torch.Tensor(self.embed_dims))
            self.modal_embbeding_L = nn.Parameter(torch.Tensor(self.embed_dims))

        self.reference_points = nn.Linear(self.embed_dims * self.scale_factor, 3)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3DPts) or isinstance(m, MSDeformableAttention3DImg) \
                    or isinstance(m, MultiScaleDeformableAttention) or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        if self.with_pts_bev_encoder:
            normal_(self.pts_level_embeds)
        if self.with_img_bev_encoder:
            normal_(self.img_level_embeds)
            normal_(self.cams_embeds)
        if self.feature_norm == 'ChannelNormWeights':
            if self.cna_constant_norm == True:
                constant_(self.pts_channel_weights, 0.5)
                constant_(self.img_channel_weights, 0.5)
            else:
                normal_(self.pts_channel_weights)
                normal_(self.img_channel_weights)
        if self.feature_norm in ('MLP_ChannelNormWeights',
                                 'Leaky_ReLU_MLP_ChannelNormWeights',
                                 'ELU_MLP_ChannelNormWeights',
                                 'Sigmoid_MLP_ChannelNormWeights'):
            xavier_init(self.channel_weights_proj, distribution='uniform', bias=0)
        if self.feature_norm == 'ModalityProjection':
            xavier_init(self.c_modal_proj)
            xavier_init(self.l_modal_proj)
        if self.spatial_norm == 'SpatialNormWeights':
            normal_(self.pts_spatial_weights)
            normal_(self.img_spatial_weights)

        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        if self.use_modal_embeds == 'MLP':
            xavier_init(self.modal_embbeding_mlp, distribution='uniform', bias=0.)
        elif self.use_modal_embeds == 'Fixed':
            normal_(self.modal_embbeding_C)
            normal_(self.modal_embbeding_L)
        
        # initialize the learnable weights
        normal_(self.img_pred_weights)
        normal_(self.img_cur_weights)
        normal_(self.pts_pred_weights)
        normal_(self.pts_cur_weights)

    def get_probability(self, prob):
        return True if np.random.random() < prob else False

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def _pre_process_img_feats(self, mlvl_img_feats, bev_queries):
        """
        preprocess img features
        """

        img_feat_flatten = []
        img_spatial_shapes = []
        for lvl, feat in enumerate(mlvl_img_feats):
            bs, num_cam, c, h, w = feat.shape
            img_spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.img_level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            img_spatial_shapes.append(img_spatial_shape)
            img_feat_flatten.append(feat)

        img_feat_flatten = torch.cat(img_feat_flatten, 2)
        img_spatial_shapes = torch.as_tensor(img_spatial_shapes, dtype=torch.long, device=bev_queries.device)
        img_level_start_index = torch.cat((img_spatial_shapes.new_zeros((1,)), img_spatial_shapes.prod(1).cumsum(0)[:-1]))

        img_feat_flatten = img_feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        return img_feat_flatten, img_spatial_shapes, img_level_start_index

    def _pre_process_pts_feats(self, mlvl_pts_feats, bev_queries):
        ## process multi-level points features
        pts_feat_flatten = []
        pts_spatial_shapes = []

        for lvl, feat in enumerate(mlvl_pts_feats):

            bs, c, h, w = feat.shape
            pts_spatial_shape = (h, w)
            feat = feat.flatten(2).permute(0, 2, 1)
            # print(' feat size:', feat.size()) # [2, 40000, 512]
            feat = feat + self.pts_level_embeds[None, lvl:lvl + 1, :].to(feat.dtype)
            pts_spatial_shapes.append(pts_spatial_shape)
            pts_feat_flatten.append(feat)

        pts_feat_flatten = torch.cat(pts_feat_flatten, 2)
        pts_spatial_shapes = torch.as_tensor(pts_spatial_shapes, dtype=torch.long, device=bev_queries.device)
        pts_level_start_index = torch.cat((pts_spatial_shapes.new_zeros((1,)), pts_spatial_shapes.prod(1).cumsum(0)[:-1]))

        pts_feat_flatten = pts_feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)

        return pts_feat_flatten, pts_spatial_shapes, pts_level_start_index

    def multi_modal_fusion(self, img_bev_embed, pts_bev_embed):
        assert self.fusion_method is not None
        if self.fusion_method == 'linear':
            fused_bev_embed = self.c_flag * img_bev_embed + self.l_flag * pts_bev_embed
        elif self.fusion_method == 'avg':
            fused_bev_embed = img_bev_embed * self.c_flag / (self.c_flag + self.l_flag) + pts_bev_embed * self.l_flag / (self.c_flag + self.l_flag)
        elif self.fusion_method == 'cat':
            if self.feature_norm == 'ModalityProjection':
                assert img_bev_embed.shape[-1] == self.embed_dims * 2 and pts_bev_embed.shape[-1] == self.embed_dims * 2
                c_pseudo_flag = 1 - self.c_flag
                l_pseudo_flag = 1 - self.l_flag

                c_true_weights = torch.Tensor([self.c_flag]).expand(self.embed_dims)
                l_psudo_weights = torch.Tensor([l_pseudo_flag]).expand(self.embed_dims)
                l_true_weights = torch.Tensor([self.l_flag]).expand(self.embed_dims)
                c_pseudo_weights = torch.Tensor([c_pseudo_flag]).expand(self.embed_dims)

                img_flags = torch.cat((c_true_weights, l_psudo_weights)).cuda()
                pts_flags = torch.cat((c_pseudo_weights, l_true_weights)).cuda()

                fused_bev_embed = img_bev_embed * img_flags + pts_bev_embed * pts_flags
            else:
                fused_bev_embed = torch.cat((img_bev_embed * self.c_flag, pts_bev_embed * self.l_flag), -1)
        else:
            raise NotImplementedError

        if self.use_modal_embeds == 'MLP':
            modal_status = torch.Tensor([self.c_flag, self.l_flag]).cuda()
            modal_embedding = self.modal_embbeding_mlp(modal_status)
            fused_bev_embed += modal_embedding
        elif self.use_modal_embeds == 'Fixed':
            modal_embedding = self.c_flag * self.modal_embbeding_C + self.l_flag * self.modal_embbeding_L
            fused_bev_embed += modal_embedding

        return fused_bev_embed

    def channel_feature_norm(self, img_bev_embed, pts_bev_embed):
        # (bs, bev_h * bev_w, embed_dims)
        if img_bev_embed is None:
            img_bev_embed = torch.zeros_like(pts_bev_embed)
        elif pts_bev_embed is None:
            pts_bev_embed = torch.zeros_like(img_bev_embed)
        vis_data = None
        if self.feature_norm == 'ChannelNormWeights':
            channel_weight_list = []
            channel_weight_list.append(self.img_channel_weights.unsqueeze(0))
            channel_weight_list.append(self.pts_channel_weights.unsqueeze(0))
            feature_weights = torch.cat(channel_weight_list, 0)
            if self.c_flag == 1 and self.l_flag == 1:
                channel_weights_norm = self.feature_norm_layer(feature_weights)
                img_norm_weights = channel_weights_norm[0]
                pts_norm_weights = channel_weights_norm[1]
            else:
                img_norm_weights = self.feature_norm_layer(feature_weights[0:1])[0]
                pts_norm_weights = self.feature_norm_layer(feature_weights[1:2])[0]

            img_bev_embed = img_bev_embed * img_norm_weights
            pts_bev_embed = pts_bev_embed * pts_norm_weights
            if self.vis_output:
                vis_data = dict(
                    feature_weights = feature_weights,
                    channel_weights_norm = channel_weights_norm,
                    img_norm_weights = img_norm_weights,
                    pts_norm_weights = pts_norm_weights
                )
        elif self.feature_norm in ('MLP_ChannelNormWeights',
                                   'Leaky_ReLU_MLP_ChannelNormWeights',
                                   'ELU_MLP_ChannelNormWeights',
                                   'Sigmoid_MLP_ChannelNormWeights'):
            input_bev_feats = torch.cat([img_bev_embed, pts_bev_embed], dim=1).permute(0,2,1)
            multi_modal_channel_weights = self.channel_weights_proj(input_bev_feats)

            if self.c_flag == 1 and self.l_flag ==1:
                multi_modal_channel_norm_weights = F.softmax(multi_modal_channel_weights, dim=-1)
                img_norm_weights = multi_modal_channel_norm_weights[:,:,0]
                pts_norm_weights = multi_modal_channel_norm_weights[:,:,1]

            else:
                img_norm_weights = F.softmax(multi_modal_channel_weights[:,:, :1], dim=-1).squeeze(-1)
                pts_norm_weights = F.softmax(multi_modal_channel_weights[:,:, 1:], dim=-1).squeeze(-1)

            img_bev_embed = img_bev_embed * img_norm_weights[:, None, :]
            pts_bev_embed = pts_bev_embed * pts_norm_weights[:, None, :]

            if self.vis_output:
                vis_data = dict(
                    multi_modal_channel_weights = multi_modal_channel_weights,
                    multi_modal_channel_norm_weights = multi_modal_channel_norm_weights,
                    img_norm_weights = img_norm_weights,
                    pts_norm_weights = pts_norm_weights
                )
        elif self.feature_norm == 'ModalityProjection':
            pseudo_pts_bev_embed = self.l_modal_proj(img_bev_embed)
            pseudo_img_bev_embed = self.c_modal_proj(pts_bev_embed)

            img_bev_embed = torch.cat([img_bev_embed, pseudo_pts_bev_embed], dim=-1)
            pts_bev_embed = torch.cat([pseudo_img_bev_embed, pts_bev_embed], dim=-1)

            if self.vis_output:
                vis_data = dict(
                    pseudo_pts_bev_embed = pseudo_pts_bev_embed,
                    pseudo_img_bev_embed = pseudo_img_bev_embed,
                )

        return img_bev_embed, pts_bev_embed, vis_data

    def spatial_feature_norm(self, img_bev_embed, pts_bev_embed):
        # (bs, bev_h * bev_w, embed_dims)
        vis_data = None
        if self.spatial_norm == 'SpatialNormWeights':
            spatial_weight_list = []
            spatial_weight_list.append(self.img_spatial_weights.unsqueeze(0))
            spatial_weight_list.append(self.pts_spatial_weights.unsqueeze(0))
            spatial_weights = torch.cat(spatial_weight_list, 0)

            if self.c_flag == 1 and self.l_flag == 1:
                spatial_weights_norm = self.spatial_norm_layer(spatial_weights)
                img_spatial_norm_weights = spatial_weights_norm[0]
                pts_spatial_norm_weights = spatial_weights_norm[1]
            else:
                img_spatial_norm_weights = self.spatial_norm_layer(spatial_weights[:1])[0]
                pts_spatial_norm_weights = self.spatial_norm_layer(spatial_weights[1:])[0]

            img_bev_embed = img_bev_embed * img_spatial_norm_weights[None,:,None]
            pts_bev_embed = pts_bev_embed * pts_spatial_norm_weights[None,:,None]

            if self.vis_output:
                vis_data = dict(
                    spatial_weights = spatial_weights,
                    spatial_weights_norm = spatial_weights_norm,
                    img_spatial_norm_weights = img_spatial_norm_weights,
                    pts_spatial_norm_weights = pts_spatial_norm_weights
                )
        return img_bev_embed, pts_bev_embed, vis_data
    
    def process_feat_prediction(self, filename, timestamp, feat_memory_reader, feature_predictor, current_feature=None, B=1):
        if self.training:
            assert current_feature is not None
            (history_feature_list, history_filename_list, history_timestamp_list), memory_info = feat_memory_reader.process_frame(
                input_timestamp=timestamp,
                file_name = filename,
                input_feature = current_feature,
                output_length = self.adj_num_frames,
                update_memory=True
            )
            
            if history_feature_list.__len__() == 0:
                # this is the first frame of this scene, just skip and wait for the next frame
                print('this is the first frame of this scene, just skip and wait for the next frame')
                sigma2 = torch.zeros(B, 1, self.bev_h, self.bev_w, device=current_feature.device, dtype=current_feature.dtype)
                return {'loss_temporal': torch.zeros(1, device='cuda', requires_grad=True),  'pred_current_flat': None,'sigma2': sigma2}
        
            historical_feats=[]
            for i in range(len(history_feature_list)):
                hist_feat = history_feature_list[i]  # (B, 40000, 256)
                hist_feat = hist_feat.view(B, self.bev_h, self.bev_w, -1).permute(0, 3, 1, 2)
                historical_feats.append(hist_feat)
            current_feat = current_feature.view(B, self.bev_h, self.bev_w, -1).permute(0, 3, 1, 2)
            
            results = feature_predictor(
                historical_feats=historical_feats,  # List[Tensor(B,256,200,200)]
                current_feat=current_feat,          # Tensor(B,256,200,200)
                return_loss=True
            )
            return results
        
        else: 
            history_feature_list, history_filename_list, history_timestamp_list = feat_memory_reader.get_history_features(timestamp, self.adj_num_frames)
            
            if history_feature_list.__len__() ==0:
                # no valid history and no current_feature, just give up and wait for the next valid frame
                print('no valid history, just give up and wait for the next valid frame with history')
                assert current_feature is not None, 'current_feature should not be None when no valid history'
                feat_memory_reader.update_memory(filename, timestamp, current_feature)
                sigma2 = torch.zeros(B, 1, self.bev_h, self.bev_w, device=current_feature.device, dtype=current_feature.dtype)
                return {'pred_current_flat': None, 'sigma2': sigma2} 
            
            historical_feats=[]
            for i in range(len(history_feature_list)):
                hist_feat = history_feature_list[i]  # (B, 40000, 256)
                hist_feat = hist_feat.view(B, self.bev_h, self.bev_w, -1).permute(0, 3, 1, 2)
                historical_feats.append(hist_feat)
            
            results= feature_predictor(
                historical_feats=historical_feats,  # List[Tensor(B,256,200,200)]
                return_loss=False,
            ) 
            if current_feature is None:
                feat_memory_reader.update_memory(filename, timestamp, results['pred_current_flat'])   
            else:
                feat_memory_reader.update_memory(filename, timestamp, current_feature)
                
            return results
                  

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                img_mlvl_feats,
                pts_mlvl_feats,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        self.l_flag = 1
        self.c_flag = 1
        # if self.drop_modality is not None and self.training is True:
        if self.drop_modality is not None:
            if isinstance(self.drop_modality, dict):
                dropout_prob = self.drop_modality['dropout_prob']
                lidar_prob = self.drop_modality['lidar_prob']
            elif isinstance(self.drop_modality, float):
                dropout_prob = self.drop_modality
                lidar_prob = self.drop_modality
            else:
                raise ValueError('Unrecognized type: {}'.format(type(self.drop_modality)))
            v_flag = self.get_probability(dropout_prob)
            if v_flag:
                self.l_flag = self.get_probability(lidar_prob)*1
                self.c_flag = 1 - self.l_flag
            # print('dropout_prob:', dropout_prob)
            # print('lidar_prob:', lidar_prob)
            # print('v_flag, l_flag, c_flag:', v_flag, self.l_flag, self.c_flag)

        # if img_mlvl_feats is None:
        #     self.c_flag = 0
        #     bs = pts_mlvl_feats[0].size(0)
        # elif pts_mlvl_feats is None:
        #     self.l_flag = 0
        #     bs = img_mlvl_feats[0].size(0)
        # else:
        #     bs = img_mlvl_feats[0].size(0)
        
        bs = len(kwargs['img_metas'])
        if img_mlvl_feats is None:
            self.c_flag = 0
        if pts_mlvl_feats is None:
            self.l_flag = 0
            
        # print('l_flag, c_flag:', self.l_flag, self.c_flag)
        if bev_pos is not None:
            bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
        if self.dual_queries:
            assert isinstance(bev_queries, list)
            bev_queries_img = bev_queries[0].unsqueeze(1).repeat(1, bs, 1)
            bev_queries_pts = bev_queries[1].unsqueeze(1).repeat(1, bs, 1)
        else:
            bev_queries_img = bev_queries_pts = bev_queries.unsqueeze(1).repeat(1, bs, 1)

        if img_mlvl_feats is not None:
            img_feat_flatten, img_spatial_shapes, img_level_start_index = self._pre_process_img_feats(img_mlvl_feats, bev_queries_img)
            img_bev_embed = self.img_bev_encoder(
                bev_queries_img,
                img_feat_flatten,
                img_feat_flatten,
                bev_h=bev_h,
                bev_w=bev_w,
                bev_pos=bev_pos,
                spatial_shapes = img_spatial_shapes,
                level_start_index = img_level_start_index,
                **kwargs) # encoder.batch_first = True: (bs, bev_h*bev_w, embed_dims)
        else:
            img_bev_embed = None

        if pts_mlvl_feats is not None:
            pts_feat_flatten, pts_spatial_shapes, pts_level_start_index = self._pre_process_pts_feats(pts_mlvl_feats, bev_queries_pts)
            pts_bev_embed = self.pts_bev_encoder(
                bev_queries_pts,
                pts_feat_flatten,
                pts_feat_flatten,
                bev_h=bev_h,
                bev_w=bev_w,
                bev_pos=bev_pos,
                spatial_shapes=pts_spatial_shapes,
                level_start_index=pts_level_start_index,
                **kwargs) # encoder.batch_first = True: (bs, bev_h*bev_w, embed_dims)
        else:
            pts_bev_embed = None

        if self.vis_output is not None:
            vis_data = dict(
                ori_img_bev_embed=img_bev_embed.clone(),
                ori_pts_bev_embed=pts_bev_embed.clone(),
            )
            
            
        batch_frame_names = [osp.split(meta['pts_filename'])[-1].split('.')[0]  for meta in kwargs['img_metas']]
        batch_time_stamps = extract_timestamp_from_filename(batch_frame_names)
        # print(batch_time_stamps)
        assert batch_frame_names.__len__() == 1, 'Only support the single-sample batch!' 
        # process memory reading and feature predicting   
        if self.training:
            predict_recon_loss = {}
            if self.pts_feat_pred:
                assert pts_bev_embed is not None
                results_pts = self.process_feat_prediction(
                    filename=batch_frame_names[0], 
                    timestamp=batch_time_stamps[0], 
                    feat_memory_reader=self.pts_feat_memory_reader, 
                    feature_predictor=self.history_pts_feat_predictor, 
                    current_feature=pts_bev_embed
                )
                if results_pts['loss_temporal'] is not None:
                    predict_recon_loss.update({'loss_feat_pred_pts': results_pts['loss_temporal']})
            
            if self.img_feat_pred:
                assert img_bev_embed is not None
                results_img = self.process_feat_prediction(
                    filename=batch_frame_names[0], 
                    timestamp=batch_time_stamps[0], 
                    feat_memory_reader=self.img_feat_memory_reader, 
                    feature_predictor=self.history_img_feat_predictor, 
                    current_feature=img_bev_embed
                )
                if results_img['loss_temporal'] is not None:
                    predict_recon_loss.update({'loss_feat_pred_img': results_img['loss_temporal']})
       
        else:
            if self.pts_feat_pred:
                results_pts = self.process_feat_prediction(
                        filename=batch_frame_names[0], 
                        timestamp=batch_time_stamps[0], 
                        feat_memory_reader=self.pts_feat_memory_reader, 
                        feature_predictor=self.history_pts_feat_predictor, 
                        current_feature=pts_bev_embed,
                    )
            
            if self.img_feat_pred:
                results_img = self.process_feat_prediction(
                        filename=batch_frame_names[0], 
                        timestamp=batch_time_stamps[0], 
                        feat_memory_reader=self.img_feat_memory_reader, 
                        feature_predictor=self.history_img_feat_predictor, 
                        current_feature=img_bev_embed,
                    )
                
        # ensure img_bev_embed_pred and pts_bev_embed_pred are not None, always        
        if results_img['pred_current_flat'] is not None:
            img_bev_embed_pred = results_img['pred_current_flat']
        else:
            print('img_feat_pred is False, use img_bev_embed instead!')
            assert img_bev_embed is not None, 'img_bev_embed should not be None when img_feat_pred is False!'
            img_bev_embed_pred = img_bev_embed.clone()
            
        if results_pts['pred_current_flat'] is not None:
            pts_bev_embed_pred = results_pts['pred_current_flat']
        else:
            print('pts_feat_pred is False, use pts_bev_embed instead!')
            assert pts_bev_embed is not None, 'pts_bev_embed should not be None when pts_feat_pred is False!'
            pts_bev_embed_pred = pts_bev_embed.clone()

        if self.training:
            img_bev_embed_org = img_bev_embed.clone().detach()
            pts_bev_embed_org = pts_bev_embed.clone().detach()  
            
        img_bev_embed_cur = img_bev_embed
        pts_bev_embed_cur = pts_bev_embed
        
        # debug
        # if pts_bev_embed is not None and img_bev_embed is not None:
        #     # save channel-average mse of img_bev_embed and img_bev_embed_pred
        #     img_bev_embed_mse = torch.mean((img_bev_embed - img_bev_embed_pred)**2, dim=-1).squeeze(0).cpu().numpy()
        #     pts_bev_embed_mse = torch.mean((pts_bev_embed - pts_bev_embed_pred)**2, dim=-1).squeeze(0).cpu().numpy()
        #     import pickle
        #     dir_ = '/dataset/shuangzhi/mmdet3d/feature_mse/hfp/'
        #     with open(dir_+batch_frame_names[0]+'.pkl', 'wb') as f:
        #         pickle.dump([img_bev_embed_mse, pts_bev_embed_mse], f)
            
        u_img, u_pts = results_img['sigma2'], results_pts['sigma2']
        u_img, u_pts  = u_img / (u_img.max().clamp_min(1e-6)), u_pts / (u_pts.max().clamp_min(1e-6))
        c_img, c_pts = 1.0 - u_img, 1.0 - u_pts
        
        if self.use_cross_modal:
            if self.vis_output is not None:
                vis_data.update({
                    'pred_img_bev_embed_before_cross': img_bev_embed_pred.clone(),
                    'pred_pts_bev_embed_before_cross': pts_bev_embed_pred.clone(),
                })
                
            crf_results = self.cross_modal_module(
                img_feat=img_bev_embed_pred,
                pts_feat=pts_bev_embed_pred,
                u_img=u_img, u_pts=u_pts,
                c_img=c_img, c_pts=c_pts,
            )

            if len(crf_results) == 4:
                img_bev_embed_pred, pts_bev_embed_pred, sigma2_img_cmf, sigma2_pts_cmf = crf_results
                u_img_cmf = sigma2_img_cmf / (sigma2_img_cmf.max().clamp_min(1e-6))
                u_pts_cmf = sigma2_pts_cmf / (sigma2_pts_cmf.max().clamp_min(1e-6))
                c_img_cmf = 1.0 - u_img_cmf
                c_pts_cmf = 1.0 - u_pts_cmf
            else:
                img_bev_embed_pred, pts_bev_embed_pred = crf_results
            
            # if pts_bev_embed is not None and img_bev_embed is not None:
            #     # save channel-average mse of img_bev_embed and img_bev_embed_pred
            #     img_bev_embed_mse = torch.mean((img_bev_embed - img_bev_embed_pred)**2, dim=-1).squeeze(0).cpu().numpy()
            #     pts_bev_embed_mse = torch.mean((pts_bev_embed - pts_bev_embed_pred)**2, dim=-1).squeeze(0).cpu().numpy()
            #     import pickle
            #     dir_ = '/dataset/shuangzhi/mmdet3d/feature_mse/cmf/'
            #     with open(dir_+batch_frame_names[0]+'.pkl', 'wb') as f:
            #         pickle.dump([img_bev_embed_mse, pts_bev_embed_mse], f)

            if self.training:
                if len(crf_results) == 4:
                    loss_unc_img = heteroscedastic_nll(img_bev_embed_pred, img_bev_embed_org, sigma2_img_cmf)
                    loss_unc_pts = heteroscedastic_nll(pts_bev_embed_pred, pts_bev_embed_org, sigma2_pts_cmf)
                    predict_recon_loss.update({'loss_unc_img': loss_unc_img, 'loss_unc_pts': loss_unc_pts})

                if self.CR_loss_type == 'mean_std':
                    mean_std_loss = {'loss_CR_mean_std_img':  self.cross_modal_loss(img_bev_embed_pred, img_bev_embed_org),
                                    'loss_CR_mean_std_pts': self.cross_modal_loss(pts_bev_embed_pred, pts_bev_embed_org)}
                    predict_recon_loss.update(mean_std_loss)
                elif self.CR_loss_type == 'mse':
                    mse_CR_loss = {'loss_CR_mse_img': self.cross_modal_loss(img_bev_embed_pred, img_bev_embed_org, loss_type='mse'),
                                'loss_CR_mse_pts': self.cross_modal_loss(pts_bev_embed_pred, pts_bev_embed_org, loss_type='mse')}
                    predict_recon_loss.update(mse_CR_loss)
                else:
                    raise ValueError(f"Unsupported CR_loss_type: {self.CR_loss_type}")
            
            if self.vis_output is not None:
                vis_data.update({
                    'enhanced_img_bev_embed_pred': img_bev_embed_pred.clone(),
                    'enhanced_pts_bev_embed_pred': pts_bev_embed_pred.clone(),
                })
            
        img_bev_embed = self.pred_cur_fusion(img_bev_embed_pred, img_bev_embed_cur, cur_valid=self.c_flag, modal='img')
        pts_bev_embed = self.pred_cur_fusion(pts_bev_embed_pred, pts_bev_embed_cur, cur_valid=self.l_flag, modal='pts')
        
        self.c_flag, self.l_flag = 1, 1
        
        img_bev_embed, pts_bev_embed, vis_data_channel = self.channel_feature_norm(img_bev_embed, pts_bev_embed)
        img_bev_embed, pts_bev_embed, vis_data_spatial = self.spatial_feature_norm(img_bev_embed, pts_bev_embed)

        fused_bev_embed = self.multi_modal_fusion(img_bev_embed, pts_bev_embed)

        query_pos, query = torch.split(object_query_embed, self.embed_dims * self.scale_factor, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        fused_bev_embed = fused_bev_embed.permute(1, 0, 2)

        ## Visualization of features
        if self.training is False and self.vis_output is not None:
            assert isinstance(self.vis_output, dict)
            outdir = self.vis_output['outdir']
            pts_path = kwargs['img_metas'][0]['pts_filename']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            result_path = osp.join(outdir, file_name)
            mmcv.utils.path.mkdir_or_exist(result_path)
            if vis_data_channel is not None:
                vis_data.update(vis_data_channel)
            if vis_data_spatial is not None:
                vis_data.update(vis_data_spatial)

            for key in self.vis_output['keys'] + self.vis_output['special_keys']:
                vis_data[key] = locals()[key]
            for attr in self.vis_output['attrs']:
                vis_data[attr] = getattr(self, attr)

            vis_data.update(dict(lidar_file_name=file_name))
            torch.save(vis_data, osp.join(result_path, 'vis_data.pt'))
        ##
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=fused_bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        inter_references_out = inter_references
        
        if self.training and predict_recon_loss:
            return fused_bev_embed, inter_states, init_reference_out, inter_references_out, predict_recon_loss
        return fused_bev_embed, inter_states, init_reference_out, inter_references_out

    def cross_modal_loss(self, crs_out, cur_out, loss_type='mean_std', eps=1e-5):
        """
        计算均值-标准差损失或MSE损失
        Args:
            crs_out: 交叉输出 (B,C,H,W)
            cur_out: 当前输出 (B,C,H,W)  
            loss_type: 'mean_std' 或 'mse'
            eps: 防止除零的小常数
        """
        cur_out = cur_out.detach()
        
        if loss_type == 'mse':
            return F.mse_loss(crs_out, cur_out)
        elif loss_type == 'mean_std':
            mx, vx = crs_out.mean((2,3)), crs_out.var((2,3)).add(eps).sqrt()
            my, vy = cur_out.mean((2,3)), cur_out.var((2,3)).add(eps).sqrt()
            return (mx - my).abs().mean() + (vx - vy).abs().mean()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

def heteroscedastic_nll(pred, target, sigma2, eps=1e-6):
    err = (pred - target.detach())**2
    return (err / (2*sigma2 + eps) + 0.5*torch.log(sigma2 + eps)).mean()

def extract_timestamp_from_filename(filename_list):
    '''
    extract timesampe from nuscenes lidar filename
    '''
    timestamp_seconds_list = []
    for filename in filename_list:
        assert '__LIDAR_TOP__' in filename
        if '.pcd.bin' in filename:
            filename = filename.replace('.pcd.bin', '')
        timestamp_str = filename[-16:]
        seconds = int(timestamp_str[:10])
        microseconds = int(timestamp_str[10:])
        timestamp_seconds = seconds + microseconds / 1000000.0
        timestamp_seconds_list.append(timestamp_seconds)
    return timestamp_seconds_list