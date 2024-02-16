import json
import logging
import os
import sys

import torch
import torch.distributed as dist
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

from dmlcloud.util import git_diff, git_hash, print_worker
from .checkpoint import ExtendedJSONEncoder


def setup_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO if dist.get_rank() == 0 else logging.WARNING)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(logging.Formatter())
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(logging.Formatter())

    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(stderr_handler)


def delimiter(n=40, newline=True):
    delim = '-' * n
    if newline:
        delim += '\n'
    return delim


def log_delimiter(n=40):
    logging.info(delimiter(n, newline=False))


def log_diagnostics(device):
    msg = f'Training distributed on {dist.get_world_size()} workers/gpus\n'
    msg += f'Using torch.distributed backend: {dist.get_backend()}\n'
    msg += delimiter()
    msg += f'SLURM_JOB_ID = {os.environ.get("SLURM_JOB_ID")}\n'
    msg += f'SLURM_STEP_ID = {os.environ.get("SLURM_STEP_ID")}\n'
    msg += f'SLURM_STEP_NODELIST = {os.environ.get("SLURM_STEP_NODELIST")}\n'
    msg += f'SLURM_TASKS_PER_NODE = {os.environ.get("SLURM_TASKS_PER_NODE")}\n'
    msg += f'SLURM_STEP_GPUS = {os.environ.get("SLURM_STEP_GPUS")}\n'
    msg += f'SLURM_GPUS_ON_NODE = {os.environ.get("SLURM_GPUS_ON_NODE")}\n'
    msg += f'SLURM_CPUS_PER_TASK = {os.environ.get("SLURM_CPUS_PER_TASK")}\n'
    msg += f'SLURM_CPU_BIND_LIST = {os.environ.get("SLURM_CPU_BIND_LIST")}\n'
    msg += delimiter()
    msg += f'Gloo available: {dist.is_gloo_available()}\n'
    msg += f'NCCL available: {dist.is_nccl_available()}\n'
    msg += f'MPI available: {dist.is_mpi_available()}\n'
    msg += delimiter()
    msg += f'CUDA_VISIBLE_DEVICES = {os.environ.get("CUDA_VISIBLE_DEVICES")}\n'
    msg += f'Device count: {torch.cuda.device_count()}'
    logging.info(msg)
    print_worker(f'Using {device}')
    log_delimiter()


def log_config(config):
    msg = 'CONFIG:\n'
    msg += json.dumps(config.dct, indent=4, cls=ExtendedJSONEncoder) + '\n'
    msg += delimiter(newline=False)
    logging.info(msg)


def log_git():
    msg = f'Git Hash: {git_hash()}\n'
    msg += f'Git Diff:\n{git_diff()}\n'
    msg += delimiter()
    logging.info(msg)


def log_model(model):
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    msg = f'# trainable parameters:     {n_trainable_params/1e6:.1f}M\n'
    msg += f'# non-trainable parameters: {n_non_trainable_params/1e6:.1f}M'
    logging.info(msg)