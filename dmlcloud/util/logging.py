import logging
import os
import sys
from pathlib import Path
from datetime import datetime
import subprocess

import torch
import numpy as np
import torch.distributed as dist

import dmlcloud
from .git import git_hash


def add_log_handlers(logger: logging.Logger):
    logger.setLevel(logging.INFO if dist.get_rank() == 0 else logging.WARNING)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(logging.Formatter())
    logger.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(logging.Formatter())
    logger.addHandler(stderr_handler)


def general_diagnostics() -> str:
    msg = f'Training on {dist.get_world_size()} GPUs\n'
    msg += f'Date: {datetime.now()}\n\n'

    msg += '* GENERAL:\n'
    msg += f'      - argv: {sys.argv}\n'
    msg += f'      - cwd: {Path.cwd()}\n'
    
    msg += f'      - host (root): {os.environ.get("HOSTNAME")}\n'  
    msg += f'      - user: {os.environ.get("USER")}\n'
    msg += f'      - git-hash: {git_hash()}\n'
    msg += f'      - conda-env: {os.environ.get("CONDA_DEFAULT_ENV", "N/A")}\n'
    msg += f'      - sys-prefix: {sys.prefix}\n'
    msg += f'      - backend: {dist.get_backend()}\n'
    msg += f'      - cuda: {torch.cuda.is_available()}\n'

    if torch.cuda.is_available():
        msg += '* GPUs (root):\n'
        nvsmi = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode()
        for line in nvsmi.splitlines():
            msg += f'      - {line}\n'

    msg += '* VERSIONS:\n'
    msg += f'      - python: {sys.version}\n'
    msg += f'      - dmlcloud: {dmlcloud.__version__}\n'
    msg += f'      - cuda: {torch.version.cuda}\n'
    try:
        msg += f'      - ' + Path('/proc/driver/nvidia/version').read_text().splitlines()[0] + '\n'
    except (FileNotFoundError, IndexError):
        pass

    msg += f'      - torch: {torch.__version__}\n'
    
    try:
        import torchvision
        msg += f'      - torchvision: {torchvision.__version__}\n'
    except ImportError:
        pass

    try:
        import torchtext
        msg += f'      - torchtext: {torchtext.__version__}\n'
    except ImportError:
        pass

    try:
        import torchaudio
        msg += f'      - torchaudio: {torchaudio.__version__}\n'
    except ImportError:
        pass

    try:
        import einops
        msg += f'      - einops: {einops.__version__}\n'
    except ImportError:
        pass

    try:
        import numpy as np
        msg += f'      - numpy: {np.__version__}\n'
    except ImportError:
        pass

    try:
        import pandas as pd
        msg += f'      - pandas: {pd.__version__}\n'
    except ImportError:
        pass

    try:
        import xarray as xr
        msg += f'      - xarray: {xr.__version__}\n'
    except ImportError:
        pass

    try:
        import sklearn
        msg += f'      - sklearn: {sklearn.__version__}\n'
    except ImportError:
        pass

    if 'SLURM_JOB_ID' in os.environ:
        msg += '* SLURM:\n'
        msg += f'      - SLURM_JOB_ID = {os.environ.get("SLURM_JOB_ID")}\n'
        msg += f'      - SLURM_STEP_ID = {os.environ.get("SLURM_STEP_ID")}\n'
        msg += f'      - SLURM_STEP_NODELIST = {os.environ.get("SLURM_STEP_NODELIST")}\n'
        msg += f'      - SLURM_TASKS_PER_NODE = {os.environ.get("SLURM_TASKS_PER_NODE")}\n'
        msg += f'      - SLURM_STEP_GPUS = {os.environ.get("SLURM_STEP_GPUS")}\n'
        msg += f'      - SLURM_GPUS_ON_NODE = {os.environ.get("SLURM_GPUS_ON_NODE")}\n'
        msg += f'      - SLURM_CPUS_PER_TASK = {os.environ.get("SLURM_CPUS_PER_TASK")}'

    return msg