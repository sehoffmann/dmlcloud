import logging
from typing import Optional, Union, Dict, List, Sequence, Any
from datetime import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf

from .checkpoint import CheckpointDir, generate_checkpoint_path, find_slurm_checkpoint
from .stage import Stage
from .metrics import MetricTracker, Reduction
from .util.distributed import local_rank
from .util.logging import add_log_handlers, general_diagnostics, experiment_header, IORedirector


class TrainingPipeline:



    def __init__(
            self, 
            cfg : Optional[Union[OmegaConf, Dict]] = None, 
            name : Optional[str] = None
        ):

        if cfg is None:
            self.cfg = OmegaConf.create()
        elif not isinstance(cfg, OmegaConf):
            self.cfg = OmegaConf.create(cfg)
        else:
            self.cfg = cfg
        
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
        verbose: bool = True
        ):
        if name in self.models:
            raise ValueError(
                f'Model with name {name} already exists'
            )
        if use_ddp:
            model = DistributedDataParallel(
                model, broadcast_buffers=False
            )
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
            raise ValueError(
                f'Optimizer with name {name} already exists'
            )
        self.optimizers[name] = optimizer
        if scheduler is not None:
            self.schedulers[name] = scheduler


    def register_dataset(self, name: str, dataset: Union[DataLoader, Dataset, Sequence], verbose: bool = True):
        if name in self.datasets:
            raise ValueError(
                f'Dataset with name {name} already exists'
            )

        self.datasets[name] = dataset
        if verbose:
            msg = f'Dataset "{name}":\n'
            msg += f'  - Batches (/Worker): {len(dataset)}\n'
            msg += f'  - Batches (Total): ~{len(dataset) * dist.get_world_size()}\n'
            self.logger.info(msg)


    def append_stage(self, stage: Stage, max_epochs : Optional[int] =None, name : Optional[str]=None):
        if not isinstance(stage, Stage):
            raise ValueError(
                'stage must be a Stage object'
            )

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
        elif resume and find_slurm_checkpoint(root):
            path = find_slurm_checkpoint(root)

        if path is None:
            path = generate_checkpoint_path(
                root=root,
                name=self.name,
                creation_time=self.start_time
            )
        self.checkpoint_dir = CheckpointDir(path)


    def track_reduce(self, 
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
            if self.checkpoint_dir.is_valid:
                self.resumed = True
            else:
                self.checkpoint_dir.create()
                self.resumed = False
            self.io_redirector = IORedirector(self.checkpoint_dir.log_file)
            self.io_redirector.install()
        else:
            self.resumed = False

        self.start_time = datetime.now()

        add_log_handlers(self.logger)
        header = experiment_header(self.name, self.checkpoint_dir, self.start_time)
        self.logger.info(header)

        if self.resumed:
            self._resume_run()

        diagnostics = general_diagnostics()
        diagnostics += '\n* CONFIG:\n' + OmegaConf.to_yaml(self.cfg)
        self.logger.info(diagnostics)

        self.pre_run()


    def _post_run(self):
        self.stop_time = datetime.now()
        self.logger.info(f'Finished training in {self.stop_time - self.start_time} ({self.stop_time})')
        if self.checkpointing_enabled:
            self.logger.info(f'Outputs have been saved to {self.checkpoint_dir}')
        self.post_run()


    def _resume_run(self):
        self.logger.info(f'Resuming training from checkpoint: {self.checkpoint_dir}')
        self.resume_run()

    def run(self):
        with _Guard(self):
            self._pre_run()
            for stage in self.stages:
                stage.run()
            self._post_run()


    
class _Guard:
    """
    Used by TrainingPipeline to ensure that the pipeline is properly cleaned up
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if self.pipeline.io_redirector is not None:
            self.pipeline.io_redirector.uninstall()