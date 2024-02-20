import logging
from typing import Optional, Union, Dict, List, Sequence
from datetime import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf

from .stage import Stage
from .metrics import MetricTracker, Reduction
from .util.distributed import local_rank
from .util.logging import add_log_handlers, general_diagnostics


class TrainingPipeline:

    def __init__(self, cfg : Optional[Union[OmegaConf, Dict]] = None):        
        if cfg is None:
            self.cfg = OmegaConf.create()
        elif not isinstance(cfg, OmegaConf):
            self.cfg = OmegaConf.create(cfg)
        else:
            self.cfg = cfg

        self.logger = logging.getLogger('dmlcloud')
        self.tracker = MetricTracker()
        self.device = None
        self.start_time = None
        self.stop_time = None

        self.stages = []
        self.datasets = {}
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}


    def register_model(self, name, model, use_ddp=True, verbose=True):
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
        value,
        step: Optional[int] = None,
    ):
        if name not in self.tracker:
            self.tracker.register_metric(name)

        self.tracker.track(name, value)


    def pre_run(self):
        pass

    def post_run(self):
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

        self.start_time = datetime.now()

        add_log_handlers(self.logger)
        self.logger.info(general_diagnostics())
        self.logger.info('* CONFIG:\n' + OmegaConf.to_yaml(self.cfg))
        
        self.pre_run()

    def _post_run(self):
        self.stop_time = datetime.now()
        self.logger.info(f'Finished training in {self.stop_time - self.start_time}')
        self.post_run()

    def run(self):
        self._pre_run()
        for stage in self.stages:
            stage.run()
        self._post_run()


    
