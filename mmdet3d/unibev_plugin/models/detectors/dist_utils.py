import torch.distributed as dist
import random
import torch

def decide_global_skip(p, device):
    """主卡决定是否 skip，并广播到所有 GPU"""
    local_decision = torch.tensor([int(random.random() < p)], device=device)
    if dist.is_initialized():
        dist.broadcast(local_decision, src=0)
    return bool(local_decision.item())