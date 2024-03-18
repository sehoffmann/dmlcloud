import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from dmlcloud.util.wandb import wandb_is_initialized, wandb_set_startup_timeout
from .checkpoint import CheckpointDir, find_slurm_checkpoint, generate_checkpoint_path
from .metrics import MetricTracker, Reduction
from .stage import Stage
from .util.distributed import local_rank
from .util.logging import add_log_handlers, experiment_header, general_diagnostics, IORedirector


class TrainingPipeline:
    def __init__(self, config: Optional[Union[OmegaConf, Dict]] = None, name: Optional[str] = None):
        if config is None:
            self.config = OmegaConf.create()
        elif not isinstance(config, OmegaConf):
            self.config = OmegaConf.create(config)
        else:
            self.config = config

        self.name = name

        self.logger = logging.getLogger('dmlcloud')
        self.checkpoint_dir = None
        self.io_redirector = None
        self.resumed = None
        self.tracker = MetricTracker()
        self.device = None
        self.start_time = None
        self.stop_time = None
        self.current_stage = None

        self.wandb = False
        self._wandb_initalizer = None

        self.stages = []
        self.datasets = {}
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}

    @property
    def checkpointing_enabled(self):
        return self.checkpoint_dir is not None

    def register_model(
        self,
        name: str,
        model: torch.nn.Module,
        use_ddp: bool = True,
        save_latest: bool = True,
        save_interval: Optional[int] = None,
        save_best: bool = False,
        best_metric: str = 'val/loss',
        verbose: bool = True,
    ):
        if name in self.models:
            raise ValueError(f'Model with name {name} already exists')
        if use_ddp:
            model = DistributedDataParallel(model, broadcast_buffers=False)
        model = model.to(self.device)
        self.models[name] = model

        if verbose:
            msg = f'Model "{name}":\n'
            msg += f'  - Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f} kk\n'
            msg += f'  - DDP: {use_ddp}\n'
            msg += f'  - {model}'
            self.logger.info(msg)

    def register_optimizer(self, name: str, optimizer, scheduler=None):
        if name in self.optimizers:
            raise ValueError(f'Optimizer with name {name} already exists')
        self.optimizers[name] = optimizer
        if scheduler is not None:
            self.schedulers[name] = scheduler

    def register_dataset(self, name: str, dataset: Union[DataLoader, Dataset, Sequence], verbose: bool = True):
        if name in self.datasets:
            raise ValueError(f'Dataset with name {name} already exists')

        self.datasets[name] = dataset
        if verbose:
            msg = f'Dataset "{name}":\n'
            try:
                length = len(dataset)
                msg += f'  - Batches (Total): ~{length * dist.get_world_size()}\n'
                msg += f'  - Batches (/Worker): {length}\n'
            except TypeError:  # __len__ not implemented
                msg += '  - Batches (Total): N/A\n'
                msg += '  - Batches (/Worker): N/A\n'
            self.logger.info(msg)

    def append_stage(self, stage: Stage, max_epochs: Optional[int] = None, name: Optional[str] = None):
        if not isinstance(stage, Stage):
            raise ValueError('stage must be a Stage object')

        stage.pipeline = self
        stage.max_epochs = max_epochs
        stage.name = name
        self.stages.append(stage)

    def enable_checkpointing(
        self,
        root: str,
        resume: bool = True,
    ):
        if self.checkpointing_enabled:
            raise ValueError('Checkpointing already enabled')

        path = None
        if resume and CheckpointDir(root).is_valid:
            path = root
            self.resumed = True
        elif resume and find_slurm_checkpoint(root):
            path = find_slurm_checkpoint(root)
            self.resumed = True
        if path is None:
            path = generate_checkpoint_path(root=root, name=self.name, creation_time=self.start_time)
            self.resumed = False
        self.checkpoint_dir = CheckpointDir(path)

    def enable_wandb(
        self,
        project: str | None = None,
        entity: str | None = None,
        group: str | None = None,
        tags: List[str] | None = None,
        startup_timeout: int = 360,
        **kwargs,
    ):
        import wandb  # import now to avoid potential long import times later on

        self.wandb = True

        def initializer():
            wandb_set_startup_timeout(startup_timeout)
            wandb.init(
                config=OmegaConf.to_container(self.config, resolve=True),
                name=self.name,
                entity=entity,
                project=project,
                group=group,
                tags=tags,
                **kwargs,
            )

        self._wandb_initalizer = initializer

    def track_reduce(
        self,
        name: str,
        value: torch.Tensor,
        step: Optional[int] = None,
        reduction: Reduction = Reduction.MEAN,
        dim: Optional[List[int]] = None,
        reduce_globally: bool = True,
    ):
        if name not in self.tracker:
            self.tracker.register_metric(name, reduction, dim, reduce_globally)

        self.tracker.track(name, value)

    def track(
        self,
        name: str,
        value: Any,
        step: Optional[int] = None,
    ):
        if name not in self.tracker:
            self.tracker.register_metric(name)

        self.tracker.track(name, value)

    def run(self):
        """
        Starts the training and runs all registered stages.
        """
        with _RunGuard(self):
            self._pre_run()
            for stage in self.stages:
                stage.run()
            self._post_run()

    def pre_run(self):
        pass

    def post_run(self):
        pass

    def resume_run(self):
        pass

    def _pre_run(self):
        if len(self.stages) == 0:
            raise ValueError('No stages defined. Use append_stage() to add stages to the pipeline.')

        if not dist.is_initialized():
            raise ValueError(
                'Default process group not initialized! Call torch.distributed.init_process_group() first.'
            )

        if torch.cuda.is_available():
            if local_rank() is None:
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cuda', local_rank())
        else:
            self.device = torch.device('cpu')

        if self.checkpointing_enabled:
            self._init_checkpointing()

        if self.wandb:
            self._wandb_initalizer()

        self.start_time = datetime.now()

        add_log_handlers(self.logger)
        header = '\n' + experiment_header(self.name, self.checkpoint_dir, self.start_time)
        self.logger.info(header)

        if self.resumed:
            self._resume_run()

        diagnostics = general_diagnostics()
        diagnostics += '\n* CONFIG:\n' + OmegaConf.to_yaml(self.config)
        self.logger.info(diagnostics)

        self.pre_run()

    def _init_checkpointing(self):
        if not self.checkpoint_dir.is_valid:
            self.checkpoint_dir.create()
            self.checkpoint_dir.save_config(self.config)
        self.io_redirector = IORedirector(self.checkpoint_dir.log_file)
        self.io_redirector.install()

    def _resume_run(self):
        self.logger.info(f'Resuming training from checkpoint: {self.checkpoint_dir}')
        self.resume_run()

    def _post_run(self):
        self.stop_time = datetime.now()
        self.logger.info(f'Finished training in {self.stop_time - self.start_time} ({self.stop_time})')
        if self.checkpointing_enabled:
            self.logger.info(f'Outputs have been saved to {self.checkpoint_dir}')
        self.post_run()

    def _pre_epoch(self):
        pass

    def _post_epoch(self):
        if self.wandb:
            import wandb

            metrics = {name: self.tracker[name][-1] for name in self.tracker}
            wandb.log(metrics)

    def _cleanup(self, exc_type, exc_value, traceback):
        """
        Called by _RunGuard to ensure that the pipeline is properly cleaned up
        """
        if exc_type is KeyboardInterrupt:
            self.logger.info('------- Training interrupted by user -------')
        elif exc_type is not None:
            self.logger.error(
                '------- Training failed with an exception -------', exc_info=(exc_type, exc_value, traceback)
            )

        if self.wandb:
            import wandb

            if wandb_is_initialized():
                wandb.finish(exit_code=0 if exc_type is None else 1)

        if self.io_redirector is not None:
            self.io_redirector.uninstall()

        return False


class _RunGuard:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        return self.pipeline._cleanup(exc_type, exc_value, traceback)
