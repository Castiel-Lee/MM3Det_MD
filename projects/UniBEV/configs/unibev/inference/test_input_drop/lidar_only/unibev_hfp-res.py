_base_ = ['../../../unibev_featpred-res_nus_LC_cnw_256_modality_dropout.py']

samples_per_gpu = 1
workers_per_gpu = 0
modality_dropout_prob = None

test_modality_dropout = dict(
    enabled=True,             
    drop_exist_rate=0.5,      
    drop_points_rate=0.5,       
    indep_drop=True,  
    drop_first_frame=False,    
    lidar_enabled=True,
    cam_enabled=False,
)

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,)

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            pts_feat_pred = True,
            img_feat_pred = False,
            drop_modality=modality_dropout_prob,),),)