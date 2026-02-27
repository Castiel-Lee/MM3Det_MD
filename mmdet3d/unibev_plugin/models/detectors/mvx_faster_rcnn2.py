import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet.models import DETECTORS
from .mvx_two_stage2 import MVXTwoStageDetector2


@DETECTORS.register_module()
class MVXFasterRCNN2(MVXTwoStageDetector2):
    """Multi-modality VoxelNet using Faster R-CNN."""

    def __init__(self, **kwargs):
        super(MVXFasterRCNN2, self).__init__(**kwargs)
