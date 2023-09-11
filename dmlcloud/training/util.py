import json
import logging
import os
import sys

import horovod.torch as hvd
import torch
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

from ..git import git_diff, git_hash
from .checkpoint import ExtendedJSONEncoder


def print_worker(msg, barrier=True):
    if barrier:
        hvd.barrier()
    print(f'Worker {hvd.rank()} ({hvd.cross_rank()}.{hvd.local_rank()}): {msg}', flush=True)
    if barrier:
        hvd.barrier()


def setup_horovod(print_status=True):
    hvd.init()
    n_tasks = int(os.environ.get('SLURM_NTASKS', 0))
    if n_tasks > 1 and hvd.size() == 1:
        print(
            'CRITICAL: Horovod only sees a single task! Run "horovodrun --check-build" an verify that MPI is supported. Terminating...'
        )
        sys.exit(1)

    if print_status:
        print_worker('STARTED')

    hvd.barrier()  # make sure that all processes are running at this point
    # this is very important, otherwise subsequent broadcast operations might time out


def setup_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO if hvd.rank() == 0 else logging.WARNING)

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
    msg = f'Training distributed on {hvd.size()} workers/gpus\n'
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
    msg += f'MPI built: {hvd.mpi_built()}\n'
    msg += f'NCCL built: {hvd.nccl_built() > 0}\n'
    msg += f'Gloo built: {hvd.gloo_built()}\n'
    msg += f'CUDA built: {hvd.cuda_built()}\n'
    msg += f'DDL built: {hvd.ddl_built()}\n'
    msg += f'ROCm built: {hvd.rocm_built()}\n'
    msg += f'oneCCL built: {hvd.ccl_built()}\n'
    msg += delimiter()
    msg += f'MPI enabled: {hvd.mpi_enabled()}\n'
    msg += f'Gloo enabled: {hvd.gloo_enabled()}\n'
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


def global_grad_norm(parameters, norm_type=2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.0)

    first_device = grads[0].device
    grouped_grads = _group_tensors_by_device_and_dtype([[g.detach() for g in grads]])

    if norm_type == torch.inf:
        norms = [g.detach().abs().max().to(first_device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        norms = []
        for (device, _), [grads] in grouped_grads.items():
            norms.extend([torch.norm(g, norm_type) for g in grads])

        total_norm = torch.norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)

    return total_norm
