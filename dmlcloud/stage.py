import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import torch
from progress_table import ProgressTable

from .metrics import MetricTracker, Reduction
from .util.distributed import is_root, root_only


class Stage:
    """
    Hook Points:
        - pre_stage()
        - post_stage()
        - pre_epoch()
        - post_epoch()
    """

    def __init__(self):
        self.pipeline = None  # set by the pipeline
        self.max_epochs = None  # set by the pipeline
        self.name = None  # set by the pipeline

        self.start_time = None
        self.stop_time = None
        self.epoch_start_time = None
        self.epoch_stop_time = None
        self.current_epoch = 1
        self._stop_requested = False

        self.metric_prefix = None
        self.table = None

    @property
    def tracker(self) -> MetricTracker:
        return self.pipeline.tracker

    @property
    def logger(self):
        return self.pipeline.logger

    @property
    def device(self):
        return self.pipeline.device

    @property
    def config(self):
        return self.pipeline.config

    def track_reduce(
        self,
        name: str,
        value: torch.Tensor,
        step: Optional[int] = None,
        reduction: Reduction = Reduction.MEAN,
        dim: Optional[List[int]] = None,
        reduce_globally: bool = True,
        prefixed: bool = True,
    ):
        if prefixed and self.metric_prefix:
            name = f'{self.metric_prefix}/{name}'
        self.pipeline.track_reduce(name, value, step, reduction, dim, reduce_globally)

    def track(self, name: str, value, step: Optional[int] = None, prefixed: bool = True):
        if prefixed and self.metric_prefix:
            name = f'{self.metric_prefix}/{name}'
        self.pipeline.track(name, value, step)

    def stop_stage(self):
        self._stop_requested = True

    def pre_stage(self):
        """
        Executed before the stage starts.
        Use this method to setup aby stage-specific data sets or models.
        """
        pass

    def post_stage(self):
        """
        Executed after the stage finishes.
        Use this method to clean up any stage-specific resources or to save any intermediate results/artifacts.
        """
        pass

    def pre_epoch(self):
        """
        Executed before each epoch.
        """
        pass

    def post_epoch(self):
        """
        Executed after each epoch and after the metrics have been reduced.
        """
        pass

    def run_epoch(self):
        """
        Train the model for one epoch. Must be implemented by subclasses.
        """
        raise NotImplementedError()

    def table_columns(self) -> List[Union[str, Dict[str, Any]]]:
        """
        Override this method to customize the metrics displayed in the progress table.

        Should return a list containing either strings or dicts.
        If a string, it will be used as both the display name and the metric name.
        If a dict, it should contain a 'name' key and a 'metric' key.
        The 'name' key will be used as the display name, and the 'metric' key will be used as the metric name.
        Additional keys are forwarded to the ProgressTable.add_column method.
        If 'metric' is None, then the user is responsible for updating the column manually.
        """
        columns = [
            {'name': 'Epoch', 'metric': 'misc/epoch'},
            {'name': 'Time/Epoch', 'metric': None},
        ]
        if self.max_epochs is not None:
            columns.append({'name': 'ETA', 'metric': None})
        return columns

    def run(self):
        """
        Runs this stage. Either until max_epochs are reached, or until stop_stage() is called.
        """
        self._pre_stage()
        while self.max_epochs is None or self.current_epoch <= self.max_epochs:
            self._pre_epoch()
            self.run_epoch()
            self._post_epoch()
            if self._stop_requested:
                break
        self._post_stage()

    def _pre_stage(self):
        self.start_time = datetime.now()
        self.table = ProgressTable(file=sys.stdout)
        self._setup_table()

        if len(self.pipeline.stages) > 1:
            self.logger.info(f'\n========== STAGE: {self.name} ==========')

        self.pre_stage()

        for handler in self.logger.handlers:
            handler.flush()

        if is_root():
            self.table._print_header()

    def _post_stage(self):
        self.stop_time = datetime.now()
        if is_root():
            self.table.close()
        if len(self.pipeline.stages) > 1:
            self.logger.info(f'Finished stage in {self.stop_time - self.start_time}')
        self.post_stage()

    def _pre_epoch(self):
        self.epoch_start_time = datetime.now()
        self.pre_epoch()
        self.pipeline._pre_epoch()

    def _post_epoch(self):
        self.epoch_stop_time = datetime.now()
        self._reduce_metrics()
        self.post_epoch()
        self.pipeline._post_epoch()
        self._update_table()
        self.current_epoch += 1

    def _reduce_metrics(self):
        self.track(name='misc/epoch', value=self.current_epoch, prefixed=False)
        self.track(
            name='misc/epoch_time', value=(self.epoch_stop_time - self.epoch_stop_time).total_seconds(), prefixed=False
        )
        self.tracker.next_epoch()
        pass

    @root_only
    def _setup_table(self):
        for column_dct in self._metrics():
            display_name = column_dct.pop('name')
            column_dct.pop('metric')
            self.table.add_column(display_name, **column_dct)

    @root_only
    def _update_table(self):
        self.table.update('Epoch', self.current_epoch)
        self.table.update('Time/Epoch', (datetime.now() - self.start_time) / self.current_epoch)
        self.table.update(
            'ETA', (datetime.now() - self.start_time) / self.current_epoch * (self.max_epochs - self.current_epoch)
        )
        for column_dct in self._metrics():
            display_name = column_dct['name']
            metric_name = column_dct['metric']
            if metric_name is not None:
                self.table.update(display_name, self.tracker[metric_name][-1])
        self.table.next_row()

    def _metrics(self):
        metrics = []
        for column in self.table_columns():
            if isinstance(column, str):
                metrics.append({'name': column, 'metric': column})
            elif isinstance(column, dict):
                if 'name' not in column:
                    raise ValueError('Column dict must contain a "name" key')
                if 'metric' not in column:
                    raise ValueError('Column dict must contain a "metric" key')
                metrics.append(column)
            else:
                raise ValueError(f'Invalid column: {column}. Must be a string or a dict.')
        return metrics


class TrainValStage(Stage):
    def __init__(self):
        super().__init__()
        self.is_train = True

    def run_epoch(self):
        self.train_epoch()
        self.val_epoch()

    def step(self, batch) -> torch.Tensor:
        raise NotImplementedError()

    def train_step(self, batch):
        return self.step(batch)

    def val_step(self, batch):
        return self.step(batch)

    def train_epoch(self):
        self.is_train = True
        self.metric_prefix = 'train'

        train_ds = self.pipeline.datasets.get('train')
        if train_ds is None:
            raise ValueError(
                'No "train" dataset found in pipeline. Use register_dataset("train", ...) to register a dataset.'
            )

        if hasattr(train_ds, 'sampler') and hasattr(train_ds.sampler, 'set_epoch'):
            train_ds.sampler.set_epoch(self.current_epoch)

        for batch in train_ds:
            for optimizer in self.pipeline.optimizers.values():
                optimizer.zero_grad()

            loss = self.train_step(batch)
            loss.backward()

            for optimizer in self.pipeline.optimizers.values():
                optimizer.step()

            self.track_reduce('loss', loss)

        for scheduler in self.pipeline.schedulers.values():
            scheduler.step()

    @torch.no_grad()
    def val_epoch(self):
        self.is_train = False
        self.metric_prefix = 'val'

        val_ds = self.pipeline.datasets.get('val')
        if val_ds is None:
            raise ValueError(
                'No "val" dataset found in pipeline. Use register_dataset("val", ...) to register a dataset.'
            )

        for batch in val_ds:
            loss = self.val_step(batch)
            self.track_reduce('loss', loss)

    def table_columns(self):
        columns = super().table_columns()
        columns.insert(1, {'name': '[Train] Loss', 'metric': 'train/loss'})
        columns.insert(2, {'name': '[Val] Loss', 'metric': 'val/loss'})
        return columns
