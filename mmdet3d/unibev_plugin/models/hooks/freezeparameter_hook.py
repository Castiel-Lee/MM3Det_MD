# 在你的plugin目录中创建 freeze_hook.py
from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class FreezeParametersHook(Hook):
    def __init__(self, keep_patterns=['history_decoder'], freeze_BN=False):
        self.keep_patterns = keep_patterns
        self.freeze_BN = freeze_BN
    
    def before_run(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        trainable_params = []
        frozen_params = []
        trainable_BNs = []
        frozen_BNs = []
        
        for name, param in model.named_parameters():
            should_train = any(pattern in name for pattern in self.keep_patterns)
            param.requires_grad = should_train
            
            if should_train:
                trainable_params.append(name)
            else:
                frozen_params.append(name)
                
        if self.freeze_BN:
            for name, module in model.named_modules():
                should_train = any(pattern in name for pattern in self.keep_patterns)
                # print(name, str(type(module)), should_train)
                if 'BatchNorm' in str(type(module)):
                    if should_train:
                        module.track_running_stats = True
                        trainable_BNs.append(name)
                    else:
                        module.track_running_stats = False
                        frozen_BNs.append(name)
            
            if trainable_BNs or frozen_BNs:
                runner.logger.info(f"BatchNorm layers - Trainable: {len(trainable_BNs)}, Frozen: {len(frozen_BNs)}")
                        
        runner.logger.info(f"Frozen {len(frozen_params)} parameter groups")
        runner.logger.info(f"Training {len(trainable_params)} parameter groups")
        
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found!")
        
        runner.logger.info("Trainable parameters:")
        for name in trainable_params:
            runner.logger.info(f"  {name}")
        
        
        if self.freeze_BN:
            runner.logger.info("Trainable BatchNorm layers:")
            for name in trainable_BNs:
                runner.logger.info(f"  ✓ (BN) {name}")