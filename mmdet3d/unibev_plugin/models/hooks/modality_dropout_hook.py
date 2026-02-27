# ---------------------------------------------
# ModalityDropoutHook for MMDet3D
# ---------------------------------------------

import torch
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module() 
class ModalityDropoutHook(Hook):
    """
    Hook to randomly drop one of img/points modalities during training/testing
    
    Args:
        drop_exist_rate (float): Probability that dropout occurs (img or points will be dropped)
        drop_points_rate (float): Given dropout occurs, probability to drop points (otherwise drop img)
        drop_on_train (bool): Apply dropout during training
        drop_on_val (bool): Apply dropout during validation
    """
    
    def __init__(self, 
                 drop_exist_rate=0.5,
                 drop_points_rate=0.5,
                 drop_on_train=True,
                 drop_on_val=False):
        self.drop_exist_rate = drop_exist_rate
        self.drop_points_rate = drop_points_rate
        self.drop_on_train = drop_on_train
        self.drop_on_val = drop_on_val
        
    def before_train_iter(self, runner):
        """Apply modality dropout during training"""
        if self.drop_on_train:
            self._apply_dropout(runner.data_batch)
            
    def before_val_iter(self, runner):
        """Apply modality dropout during validation"""
        if self.drop_on_val:
            self._apply_dropout(runner.data_batch)
            
    def _apply_dropout(self, data_batch):
        """
        Randomly drop either img or points based on the specified rates
        
        Logic:
        1. First check if dropout should occur (drop_exist_rate)
        2. If dropout occurs, decide which modality to drop:
           - drop_points_rate probability: drop points (set points=None)
           - (1-drop_points_rate) probability: drop img (set img=None)
        """
        # Check if dropout should occur
        if torch.rand(1).item() < self.drop_exist_rate:
            # Dropout occurs, decide which modality to drop
            if torch.rand(1).item() < self.drop_points_rate:
                # Drop points
                if 'points' in data_batch:
                    data_batch['points'] = None
                    print(f"Dropped 'points' modality")
            else:
                # Drop img
                if 'img' in data_batch:
                    data_batch['img'] = None
                    print(f"Dropped 'img' modality")


# ---------------------------------------------
# Usage Example in Config File
# ---------------------------------------------

"""
# In your config.py file:

custom_hooks = [
    dict(
        type='ModalityDropoutHook',
        drop_exist_rate=0.3,      # 30% chance dropout occurs
        drop_points_rate=0.6,     # If dropout occurs, 60% chance drop points, 40% chance drop img
        drop_on_train=True,
        drop_on_val=False
    )
]
"""