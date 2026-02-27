# ---------------------------------------------
# Modality Dropout Utils for Testing
# ---------------------------------------------

import torch
import mmcv
import json


class TestModalityDropout:
    """
    Modality dropout for testing scenarios
    """
    
    def __init__(self, drop_exist_rate=0.5, drop_points_rate=0.5, drop_first_frame=True, time_interval=1.0, enabled=False,
                 lidar_enabled=True, cam_enabled=True):
        self.drop_exist_rate = drop_exist_rate
        self.drop_points_rate = drop_points_rate
        self.enabled = enabled
        self.drop_first_frame = drop_first_frame
        self.current_timestamp = 0.0
        self.time_interval = time_interval  # seconds, for continuous data flow
        self.lidar_enabled = lidar_enabled
        self.cam_enabled = cam_enabled
        
    def __call__(self, data_batch):
        """Apply modality dropout to test data"""
        if not self.enabled:
            return data_batch

        if not self.drop_first_frame:
            assert data_batch['img_metas'].__len__() == 1, \
                "Currently, only single frame dropout is supported for continuous data flow."
            current_timestamp = data_batch['img_metas'][0].data[0][0]['pts_filename'].split('_')[-1].split('.')[0]
            current_timestamp = float(current_timestamp)/1e6
            if abs(self.current_timestamp - current_timestamp) > self.time_interval:
                print(f"Test: Current timestamp {current_timestamp} is more than {self.time_interval} seconds apart from the last processed timestamp {self.current_timestamp}.")
                self.current_timestamp = current_timestamp
                
                if 'points' in data_batch and not self.lidar_enabled:
                    data_batch['points'] = None
                    # print(f"Test: Dropped 'points' modality")
                if 'img' in data_batch and not self.cam_enabled:
                    data_batch['img'] = None
                    # print(f"Test: Dropped 'img' modality")
                return data_batch
            
            
        # Check if dropout should occur
        if torch.rand(1).item() < self.drop_exist_rate:
            # Dropout occurs, decide which modality to drop
            if torch.rand(1).item() < self.drop_points_rate:
                # Drop points
                if 'points' in data_batch:
                    data_batch['points'] = None
                    # print(f"Test: Dropped 'points' modality")
            else:
                # Drop img
                if 'img' in data_batch:
                    data_batch['img'] = None
                    # print(f"Test: Dropped 'img' modality")
        if not self.drop_first_frame:
            self.current_timestamp = current_timestamp
        
        if 'points' in data_batch and not self.lidar_enabled:
            data_batch['points'] = None
            # print(f"Test: Dropped 'points' modality")
        if 'img' in data_batch and not self.cam_enabled:
            data_batch['img'] = None
            # print(f"Test: Dropped 'img' modality")
            
        return data_batch

class TestModalityIndepDropout:
    """
    Modality dropout for testing scenarios
    """
    
    def __init__(self, drop_img_rate=0.5, drop_pts_rate=0.5, drop_first_frame=True, time_interval=1.0, enabled=False,
                 lidar_enabled=True, cam_enabled=True):
        self.drop_img_rate = drop_img_rate
        self.drop_pts_rate = drop_pts_rate
        self.enabled = enabled
        self.drop_first_frame = drop_first_frame
        self.current_timestamp = 0.0
        self.time_interval = time_interval  # seconds, for continuous data flow
        self.lidar_enabled = lidar_enabled
        self.cam_enabled = cam_enabled
        
    def __call__(self, data_batch):
        """Apply modality dropout to test data"""
        if not self.enabled:
            return data_batch

        if not self.drop_first_frame:
            assert data_batch['img_metas'].__len__() == 1, \
                "Currently, only single frame dropout is supported for continuous data flow."
            current_timestamp = data_batch['img_metas'][0].data[0][0]['pts_filename'].split('_')[-1].split('.')[0]
            current_timestamp = float(current_timestamp)/1e6
            if abs(self.current_timestamp - current_timestamp) > self.time_interval:
                print(f"Test: Current timestamp {current_timestamp} is more than {self.time_interval} seconds apart from the last processed timestamp {self.current_timestamp}.")
                self.current_timestamp = current_timestamp
                
                if 'points' in data_batch and not self.lidar_enabled:
                    data_batch['points'] = None
                    # print(f"Test: Dropped 'points' modality")
                if 'img' in data_batch and not self.cam_enabled:
                    data_batch['img'] = None
                    # print(f"Test: Dropped 'img' modality")
                return data_batch
            
            
        # Check if dropout should occur
        if torch.rand(1).item() < self.drop_img_rate:
            if 'img' in data_batch:
                data_batch['img'] = None
                # print(f"Test: Dropped 'img' modality")
        
        if torch.rand(1).item() < self.drop_pts_rate:
            if 'points' in data_batch:
                data_batch['points'] = None
                # print(f"Test: Dropped 'points' modality")

        if not self.drop_first_frame:
            self.current_timestamp = current_timestamp
        
        if 'points' in data_batch and not self.lidar_enabled:
            data_batch['points'] = None
            # print(f"Test: Dropped 'points' modality")
        if 'img' in data_batch and not self.cam_enabled:
            data_batch['img'] = None
            # print(f"Test: Dropped 'img' modality")
            
        return data_batch


def single_gpu_test_with_dropout(model,
                                data_loader,
                                show=False,
                                out_dir=None,
                                show_score_thr=0.3,
                                test_dropout=None):
    """Test model with single gpu and optional modality dropout.
    
    This function is nearly identical to the official single_gpu_test,
    with the addition of modality dropout capability.
    
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save visualization results.
            Default: False.
        out_dir (str): The path to save visualization results.
            Default: None.
        show_score_thr (float): Score threshold for visualization.
            Default: 0.3.
        test_dropout (TestModalityDropout): Modality dropout instance.
            Default: None.
            
    Returns:
        list[dict]: The prediction results.
    """
    from mmcv.image import tensor2imgs
    from os import path as osp
    from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                                SingleStageMono3DDetector)
    
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    for i, data in enumerate(data_loader):
        # Apply modality dropout if enabled - THIS IS THE ONLY ADDITION
        if test_dropout:
            data = test_dropout(data)
            
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector, Base3DSegmentor,
                         SingleStageMono3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(
                    data,
                    result,
                    out_dir=out_dir,
                    show=show,
                    score_thr=show_score_thr)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'][0],
                                                  torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for j, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[j],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
                        
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
            
    return results