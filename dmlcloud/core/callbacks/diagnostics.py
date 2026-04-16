import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.cuda
from omegaconf import OmegaConf

import dmlcloud.core.logging as dml_logging
import dmlcloud.slurm as dmlcloud_slurm
from dmlcloud.core.distributed import world_size
from dmlcloud.git import git_hash
from dmlcloud.util.thirdparty import is_imported, ML_MODULES, try_get_version
from dmlcloud.version import __version__ as dmlcloud_version
from .common import Callback


class DiagnosticsCallback(Callback):
    """
    A callback that logs diagnostics information at the beginning of training.

    Args:
        verbose_config: If True, print full config. If False, print only key fields (default: False).
    """

    def __init__(self, verbose_config: bool = False):
        self.verbose_config = verbose_config

    def _experiment_header(
        self,
        name: str | None,
        run_dir: str | None,
        date: datetime,
    ) -> str:
        msg = f'...............  Experiment: {name if name else "N/A"}  ...............\n'
        msg += f'- Date: {date}\n'
        msg += f'- Checkpoint Dir: {run_dir if run_dir else "N/A"}\n'
        msg += f'- Training on {world_size()} GPUs\n'
        return msg

    def _general_diagnostics(self) -> str:
        msg = '* GENERAL:\n'
        msg += f'    - argv: {sys.argv}\n'
        msg += f'    - cwd: {Path.cwd()}\n'

        msg += f'    - host (root): {os.environ.get("HOSTNAME")}\n'
        msg += f'    - user: {os.environ.get("USER")}\n'
        msg += f'    - git-hash: {git_hash()}\n'
        msg += f'    - conda-env: {os.environ.get("CONDA_DEFAULT_ENV", "N/A")}\n'
        msg += f'    - sys-prefix: {sys.prefix}\n'
        msg += f'    - backend: {torch.distributed.get_backend()}\n'
        msg += f'    - cuda: {torch.cuda.is_available()}\n'

        msg += '* VERSIONS:\n'
        msg += f'    - python: {sys.version}\n'
        msg += f'    - cuda (torch): {torch.version.cuda}\n'
        try:
            msg += '      - ' + Path('/proc/driver/nvidia/version').read_text().splitlines()[0] + '\n'
        except (FileNotFoundError, IndexError):
            pass

        msg += f'    - dmlcloud: {dmlcloud_version}\n'

        for module_name in ML_MODULES:
            if is_imported(module_name):
                msg += f'    - {module_name}: {try_get_version(module_name)}\n'

        if 'SLURM_JOB_ID' in os.environ:
            msg += '* SLURM:\n'
            msg += f'    - SLURM_JOB_ID = {dmlcloud_slurm.slurm_job_id()}\n'
            msg += f'    - SLURM_STEP_ID = {dmlcloud_slurm.slurm_step_id()}\n'
            msg += f'    - SLURM_STEP_NODELIST = {os.environ.get("SLURM_STEP_NODELIST")}\n'
            msg += f'    - SLURM_TASKS_PER_NODE = {os.environ.get("SLURM_TASKS_PER_NODE")}\n'
            msg += f'    - SLURM_STEP_GPUS = {os.environ.get("SLURM_STEP_GPUS")}\n'
            msg += f'    - SLURM_GPUS_ON_NODE = {os.environ.get("SLURM_GPUS_ON_NODE")}\n'
            msg += f'    - SLURM_CPUS_PER_TASK = {os.environ.get("SLURM_CPUS_PER_TASK")}'

        return msg

    def pre_run(self, pipe):
        header = '\n' + self._experiment_header(pipe.name, pipe.run_dir, pipe.start_time)
        dml_logging.info(header)

        diagnostics = self._general_diagnostics()

        if self.verbose_config:
            # Print full config
            diagnostics += '\n* CONFIG:\n'
            diagnostics += '\n'.join(f'    {line}' for line in OmegaConf.to_yaml(pipe.config, resolve=True).splitlines())
        else:
            # Print only key config fields for brevity
            diagnostics += '\n* CONFIG SUMMARY:\n'
            key_fields = ['name', 'datamodules', 'batch_size', 'lr', 'base_lr', 'epochs', 'loss', 'compile', 'wandb_project']
            for field in key_fields:
                if field in pipe.config:
                    value = pipe.config[field]
                    # Truncate long values
                    value_str = str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:97] + '...'
                    diagnostics += f'    - {field}: {value_str}\n'

            # Show model modules count
            if 'model' in pipe.config and 'modules' in pipe.config.model:
                diagnostics += f'    - model.modules: {len(pipe.config.model.modules)} modules\n'

        dml_logging.info(diagnostics)

    def post_stage(self, stage):
        if len(stage.pipe.stages) > 1:
            dml_logging.info(f'Finished stage in {stage.end_time - stage.start_time}')

    def post_run(self, pipe):
        dml_logging.info(f'Finished training in {pipe.stop_time - pipe.start_time} ({pipe.stop_time})')
        if pipe.has_checkpointing:
            dml_logging.info(f'Outputs have been saved to {pipe.run_dir}')
