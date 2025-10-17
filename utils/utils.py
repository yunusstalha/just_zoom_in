'''
some of it taken from 
https://github.com/karpathy/nanochat/blob/master/nanochat/common.py
https://github.com/huggingface/nanoVLM/tree/main
'''

import os
import math
import time
import torch
import torch.distributed as dist

import logging
from .logger import setup_default_logging

setup_default_logging()
logger = logging.getLogger(__name__)

BANNER = r"""
       █████                     █████       ███████████                                               ███            
      ░░███                     ░░███       ░█░░░░░░███                                               ░░░             
       ░███  █████ ████  █████  ███████     ░     ███░    ██████   ██████  █████████████              ████  ████████  
       ░███ ░░███ ░███  ███░░  ░░░███░           ███     ███░░███ ███░░███░░███░░███░░███  ██████████░░███ ░░███░░███ 
       ░███  ░███ ░███ ░░█████   ░███           ███     ░███ ░███░███ ░███ ░███ ░███ ░███ ░░░░░░░░░░  ░███  ░███ ░███ 
 ███   ░███  ░███ ░███  ░░░░███  ░███ ███     ████     █░███ ░███░███ ░███ ░███ ░███ ░███             ░███  ░███ ░███ 
░░████████   ░░████████ ██████   ░░█████     ███████████░░██████ ░░██████  █████░███ █████            █████ ████ █████
 ░░░░░░░░     ░░░░░░░░ ░░░░░░     ░░░░░     ░░░░░░░░░░░  ░░░░░░   ░░░░░░  ░░░░░ ░░░ ░░░░░            ░░░░░ ░░░░ ░░░░░ 
                        the Ohio State University -  Photogrammetric Computer Vision Lab - 2025                                
"""


def generate_run_name(cfg):
    """Generates a descriptive run name from the config."""
    # Model details
    encoder_name = cfg.model.encoder_name.split('/')[-1]
    decoder_details = f"decL{cfg.model.decoder_num_layers}H{cfg.model.decoder_num_heads}"

    # Training details
    batch_size = f"bs{cfg.training.batch_size}"
    learning_rate = f"lr{cfg.training.learning_rate}"
    epochs = f"e{cfg.training.num_epochs}"

    # Data details
    data_details = f"seq{cfg.data.sequence_length}grid{cfg.data.grid_size}"

    # Timestamp
    date = time.strftime("%m%d-%H%M")

    # Combine everything
    run_name = (
        f"{encoder_name}_{decoder_details}_{data_details}_"
        f"{batch_size}_{learning_rate}_{epochs}_{date}"
    )
    return run_name

def print_banner():
    """Prints the project banner to the console."""
    print0(BANNER)
    
def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def is_ddp():
    return int(os.environ.get('RANK', -1)) != -1

def get_dist_info():
    '''Helper function for ddp.
        WORLD_SIZE: The total number of processes (and usually GPUs) in your training job.
        RANK: A unique ID for each process, from 0 to WORLD_SIZE - 1.
        LOCAL_RANK: A unique ID for each process on a single machine. If you're training on one 8-GPU machine, LOCAL_RANK will go from 0 to 7.
    '''
    if is_ddp():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1
    
def compute_init():
    """Basic initialization for ddp."""

    # CUDA is currently required
    assert torch.cuda.is_available(), "CUDA is needed for a distributed run atm"

    # Seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # skipping full reproducibility for now, possibly investigate slowdown later
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Precision
    torch.set_float32_matmul_precision("high") # uses tf32 instead of fp32 for matmuls

    # Distributed setup: Distributed Data Parallel (DDP), optional
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp:
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device) # make "cuda" default to this device
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device("cuda")

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    """Companion function to compute_init, to clean things up before script exit"""
    if is_ddp():
        dist.destroy_process_group()

class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass

def get_lr(it, max_lr, warmup_steps, max_steps):
    """Calculates the learning rate for a given step."""
    # 1) linear warmup for warmup_steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > max_steps, return min learning rate
    if it > max_steps:
        return max_lr * 0.1  # min_lr is 10% of max_lr
    # 3) in between, use cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (max_lr * 0.1) + coeff * (max_lr * 0.9)