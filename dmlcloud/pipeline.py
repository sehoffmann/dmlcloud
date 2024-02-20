import logging
from typing import Optional, Union, Dict, List
from datetime import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from omegaconf import OmegaConf

from .stage import Stage
from .metrics import MetricTracker, Reduction
from .util.distributed import local_rank


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
        self.stages = []
        self.start_time = None
        self.stop_time = None

    def register_model(self, name, model, use_ddp=True):
        if use_ddp:
            model = DistributedDataParallel(
                model, device_ids=[self.device], broadcast_buffers=False
            )
        self.models[name] = model

    def register_optimizer(self, name : str, optimizer):
        self.optimizers[name] = optimizer

    def init_wandb(self):
        '''
        wandb_set_startup_timeout(600)
        exp_name = self.cfg.wb_name if self.cfg.wb_name else self.cfg.name
        wandb.init(
            project=self.cfg.wb_project,
            name=exp_name,
            tags=self.cfg.wb_tags,
            dir=self.model_dir,
            id=self.job_id,
            resume='must' if self.is_resumed else 'never',
            config=self.cfg.as_dictionary(),
        )
        '''
        pass

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
        self.pre_run()

    def _post_run(self):
        self.stop_time = datetime.now()
        self.logger.info(f'Finished pipeline in {self.stop_time - self.start_time}')
        self.post_run()

    def run(self):
        self._pre_run()
        for stage in self.stages:
            stage.run()
        self._post_run()


    
