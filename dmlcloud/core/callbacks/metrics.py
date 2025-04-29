import csv
from pathlib import Path
from typing import TYPE_CHECKING, Union

from .common import Callback


if TYPE_CHECKING:
    from dmlcloud.core.stage import Stage


class ReduceMetricsCallback(Callback):
    """
    A callback that reduces the metrics at the end of each epoch and appends them to the history.
    """

    def __init__(self, log_every_n_steps=50):
        self.log_every_n_steps = log_every_n_steps

    def _reduce_epoch_metrics(self, stage):
        metrics = stage.metrics.reduce()
        stage.history.append_metrics(**metrics)

    def _reduce_step_metrics(self, stage):
        metrics = stage.step_metrics.reduce()
        stage.step_history.append_metrics(**metrics)

    def post_epoch(self, stage: 'Stage'):
        stage.log('misc/epoch', stage.current_epoch, prefixed=False, reduction='max')
        self._reduce_epoch_metrics(stage)
        stage.step = 0  # Reset the step counter

    def post_step(self, stage: 'Stage'):
        stage.log('misc/step', stage.global_step, prefixed=False, reduction='max')

        if stage.global_step % self.log_every_n_steps == 0:
            self._reduce_step_metrics(stage)

        stage.step += 1
        stage.global_step += 1

    def post_stage(self, stage):
        has_unreduced_metrics = False
        for metric in stage.step_metrics.metrics.values():
            if metric.update_called:
                has_unreduced_metrics = True
                break

        # need to check global_step > 0 to avoid reducing when finish_step() was never called once
        if has_unreduced_metrics and stage.global_step > 0:
            self._reduce_step_metrics(stage)


class CsvCallback(Callback):
    """
    Saves metrics to a CSV file at the end of each epoch.
    """

    def __init__(self, directory: Union[str, Path]):
        """
        Initialize the callback with the given path.

        Args:
            directory (Union[str, Path]): The path to the directory where the CSV files will be saved.
        """
        self.directory = Path(directory)
        self.last_steps = {}

    def _build_name(self, stage: 'Stage', prefix: str):
        duplicate_stages = [s for s in stage.pipe.stages if s.name == stage.name]
        idx = duplicate_stages.index(stage)
        if len(duplicate_stages) > 1:
            return self.directory / f'{prefix}_{stage.name}_{idx + 1}.csv'
        else:
            return self.directory / f'{prefix}_{stage.name}.csv'

    def epoch_path(self, stage: 'Stage'):
        return self._build_name(stage, 'epoch_metrics')

    def step_path(self, stage: 'Stage'):
        return self._build_name(stage, 'step_metrics')

    def pre_stage(self, stage: 'Stage'):
        # If for some reason we can't write to the file or it exists already, its better to fail early
        with open(self.epoch_path(stage), 'x'):
            pass

    def _write_history(self, file, history, step_metric, step_name):
        writer = csv.writer(file)

        metric_names = list(history.keys())
        metric_names.remove(step_metric)

        writer.writerow([step_name] + metric_names)  # Header
        for row in history.rows():
            csv_row = [row[step_metric]] + [row[name] for name in metric_names]
            writer.writerow(csv_row)

    def _maybe_write_step_metrics(self, stage: 'Stage'):
        if stage.step_history.num_steps > self.last_steps.get(stage, 0):
            self.last_steps[stage] = stage.step_history.num_steps
            with open(self.step_path(stage), 'w') as f:
                self._write_history(f, stage.step_history, 'misc/step', 'step')

    def post_epoch(self, stage: 'Stage'):
        with open(self.epoch_path(stage), 'w') as f:
            self._write_history(f, stage.history, 'misc/epoch', 'epoch')

    def post_step(self, stage: 'Stage'):
        self._maybe_write_step_metrics(stage)

    def post_stage(self, stage):
        self._maybe_write_step_metrics(stage)  # edge case: last steps of training
