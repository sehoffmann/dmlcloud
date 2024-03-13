import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist

import dmlcloud
from . import slurm
from .git import git_hash
from .thirdparty import try_get_version


class IORedirector:
    """
    Context manager to redirect stdout and stderr to a file.
    Data is written to the file and the original streams.
    """

    class Stdout:
        def __init__(self, parent):
            self.parent = parent

        def write(self, data):
            self.parent.file.write(data)
            self.parent.stdout.write(data)

        def flush(self):
            self.parent.file.flush()
            self.parent.stdout.flush()

    class Stderr:
        def __init__(self, parent):
            self.parent = parent

        def write(self, data):
            self.parent.file.write(data)
            self.parent.stderr.write(data)

        def flush(self):
            self.parent.file.flush()
            self.parent.stderr.flush()

    def __init__(self, log_file: Path):
        self.path = log_file
        self.file = None
        self.stdout = None
        self.stderr = None

    def install(self):
        if self.file is not None:
            return

        self.file = self.path.open('a')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.stdout.flush()
        self.stderr.flush()

        sys.stdout = self.Stdout(self)
        sys.stderr = self.Stderr(self)

    def uninstall(self):
        self.stdout.flush()
        self.stderr.flush()

        sys.stdout = self.stdout
        sys.stderr = self.stderr

        self.file.close()

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.uninstall()


def add_log_handlers(logger: logging.Logger):
    if logger.hasHandlers():
        return

    logger.setLevel(logging.INFO if dist.get_rank() == 0 else logging.WARNING)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(lambda record: record.levelno < logging.WARNING)
    stdout_handler.setFormatter(logging.Formatter())
    logger.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(logging.Formatter())
    logger.addHandler(stderr_handler)


def experiment_header(
    name: str | None,
    checkpoint_dir: str | None,
    date: datetime,
) -> str:
    msg = f'...............  Experiment: {name if name else "N/A"}  ...............\n'
    msg += f'- Date: {date}\n'
    msg += f'- Checkpoint Dir: {checkpoint_dir if checkpoint_dir else "N/A"}\n'
    msg += f'- Training on {dist.get_world_size()} GPUs\n'
    return msg


def general_diagnostics() -> str:
    msg = '* GENERAL:\n'
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
        msg += '      - ' + Path('/proc/driver/nvidia/version').read_text().splitlines()[0] + '\n'
    except (FileNotFoundError, IndexError):
        pass

    msg += f'      - torch: {torch.__version__}\n'
    if try_get_version('torchvision'):
        msg += f'      - torchvision: {try_get_version("torchvision")}\n'
    if try_get_version('torchtext'):
        msg += f'      - torchtext: {try_get_version("torchtext")}\n'
    if try_get_version('torchaudio'):
        msg += f'      - torchaudio: {try_get_version("torchaudio")}\n'
    if try_get_version('einops'):
        msg += f'      - einops: {try_get_version("einops")}\n'
    if try_get_version('numpy'):
        msg += f'      - numpy: {try_get_version("numpy")}\n'
    if try_get_version('pandas'):
        msg += f'      - pandas: {try_get_version("pandas")}\n'
    if try_get_version('xarray'):
        msg += f'      - xarray: {try_get_version("xarray")}\n'
    if try_get_version('sklearn'):
        msg += f'      - sklearn: {try_get_version("sklearn")}\n'

    if 'SLURM_JOB_ID' in os.environ:
        msg += '* SLURM:\n'
        msg += f'      - SLURM_JOB_ID = {slurm.slurm_job_id()}\n'
        msg += f'      - SLURM_STEP_ID = {slurm.slurm_step_id()}\n'
        msg += f'      - SLURM_STEP_NODELIST = {os.environ.get("SLURM_STEP_NODELIST")}\n'
        msg += f'      - SLURM_TASKS_PER_NODE = {os.environ.get("SLURM_TASKS_PER_NODE")}\n'
        msg += f'      - SLURM_STEP_GPUS = {os.environ.get("SLURM_STEP_GPUS")}\n'
        msg += f'      - SLURM_GPUS_ON_NODE = {os.environ.get("SLURM_GPUS_ON_NODE")}\n'
        msg += f'      - SLURM_CPUS_PER_TASK = {os.environ.get("SLURM_CPUS_PER_TASK")}'

    return msg
