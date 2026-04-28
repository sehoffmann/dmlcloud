import sys
from typing import Callable, Optional, TYPE_CHECKING

from progress_table import ProgressTable

from dmlcloud.core.distributed import is_root
from dmlcloud.util.logging import DevNullIO, TimedeltaFormatter
from .common import Callback

if TYPE_CHECKING:
    from dmlcloud.core.stage import Stage


class TableCallback(Callback):
    """
    A callback that updates a table with the latest metrics from a stage.

    By default the table prints one row per epoch.  Call
    ``enable_step_updates()`` to switch to intra-epoch mode: the current row
    is kept alive and refreshed on every ``post_step``, so long-running
    single-epoch stages (e.g. prediction) show live progress.
    """

    def __init__(self):
        self._table = None
        self.tracked_metrics = {}
        self.formatters = {}
        self._step_updates = False

    def enable_step_updates(self):
        """Opt into intra-epoch live progress updates.

        Must be called before the first ``add_column`` / ``get_table`` call
        so the ``ProgressTable`` is created with ``interactive=1``.
        """
        self._step_updates = True

    def get_table(self, stage: 'Stage'):
        if self._table is None:
            interactive = 1 if self._step_updates else 0
            self._table = ProgressTable(file=sys.stdout if is_root() else DevNullIO(), interactive=interactive)
            # Override the default cell formatter so None / NaN cells render
            # as the empty string instead of raising
            # ``TypeError: unsupported format string passed to NoneType.__format__``.
            # This regularly happens when the user logs a metric on some
            # epochs but not all (e.g. ``val_freq > 1``): cells skipped that
            # epoch stay as None and crash the default ``fmt``. Crashing
            # inside ``table.close()`` at ``post_stage`` aborts the whole
            # stage after the training has already completed.
            _default_fmt = self._table.custom_cell_format

            def _none_safe_fmt(x):
                if x is None:
                    return ""
                try:
                    return _default_fmt(x)
                except (TypeError, ValueError):
                    return str(x)

            self._table.custom_cell_format = _none_safe_fmt
            self.track_metric(stage, 'Epoch', width=5)
            self.track_metric(stage, 'Took', 'misc/epoch_time', formatter=TimedeltaFormatter(), width=7)
            if stage._run_epoch_overridden:
                self.track_metric(stage, 'ETA', 'misc/eta', formatter=TimedeltaFormatter(), width=7)
        return self._table

    def set_table(self, value):
        self._table = value

    def track_metric(
        self,
        stage: 'Stage',
        name: str,
        metric: Optional[str] = None,
        formatter: Optional[Callable] = None,
        width: Optional[int] = None,
        color: Optional[str] = None,
        alignment: Optional[str] = None,
    ):
        """
        Track a metric in the table.

        If no metric name is provided, only a column is created and the caller must update the value manually.
        If a formatter is provided, the metric value will be passed through the formatter before being displayed.

        For a detailed description of width, color, and alignment, see `ProgressTable.add_column`.

        Args:
            name (str): The name of the column.
            metric (str, optional): The name of the metric to track. Defaults to None.
            formatter (Callable, optional): A function that takes the metric value and returns a string. Defaults to None.
            width (int, optional): The width of the column. Defaults to None.
            color (str, optional): The color of the column. Defaults to None.
            alignment (str, optional): The alignment of the column. Defaults to None.
        """
        if formatter and not metric:
            raise ValueError('Cannot provide a formatter without a metric name')

        table = self.get_table(stage)
        table.add_column(name, width=width, color=color, alignment=alignment)

        if metric:
            self.tracked_metrics[name] = metric
            self.formatters[name] = formatter

    def pre_stage(self, stage: 'Stage'):
        self.get_table(stage)  # Ensure the table has been created at this point

    def post_stage(self, stage: 'Stage'):
        table = self.get_table(stage)
        table.close()

    def pre_epoch(self, stage: 'Stage'):
        table = self.get_table(stage)
        if 'Epoch' in self.get_table(stage).column_names:
            table['Epoch'] = stage.current_epoch

    def post_epoch(self, stage: 'Stage'):
        table = self.get_table(stage)
        metrics = stage.history.last()

        for column_name, metric_name in self.tracked_metrics.items():
            if column_name not in table.column_names:  # When does this happen?
                continue

            if metric_name in metrics:
                value = metrics[metric_name]
                formatter = self.formatters[column_name]
                if formatter is not None:
                    value = formatter(value)
                table.update(column_name, value)
            else:
                pass  # don't update -> empty cell

        table.next_row()

    def post_step(self, stage: 'Stage'):
        if not self._step_updates:
            return

        # Compute only the metrics that are actually displayed in the table.
        # Avoids paying the cost of reduce() on untracked metrics (e.g. eval
        # metrics backed by torchmetrics) that are not shown here.
        # This runs after ReduceMetricsCallback (priority TABLE > METRIC_REDUCTION),
        # so stage.metrics already contains the values logged this step.
        table = self.get_table(stage)

        for column_name, metric_name in self.tracked_metrics.items():
            if column_name not in table.column_names:
                continue
            if metric_name not in stage.metrics.metrics:
                continue
            metric = stage.metrics.metrics[metric_name]
            if not metric.update_called:
                continue

            value = metric.compute()
            if hasattr(value, 'item'):
                value = value.item()
            formatter = self.formatters[column_name]
            if formatter is not None:
                value = formatter(value)
            table.update(column_name, value)
