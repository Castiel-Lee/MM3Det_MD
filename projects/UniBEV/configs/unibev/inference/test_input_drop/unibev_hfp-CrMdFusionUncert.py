_base_ = ['../../unibev_featpred-res_CrMdFusionUncert-IdtDp-Stage1_nus_LC_cnw_256_modality_dropout.py']

samples_per_gpu = 1
workers_per_gpu = 6
modality_dropout_prob = None

test_modality_dropout = dict(
    enabled=True,              
    drop_exist_rate=0.5,       # drop rate when dependent drop
    drop_points_rate=0.5,       # drop rate when points drop (also lidar drop when independent drop)
    indep_drop=True,  # drop lidar and point independently
    drop_first_frame=False,    
)

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,)

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            drop_modality=modality_dropout_prob,),),)