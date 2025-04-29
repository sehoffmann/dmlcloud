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
    """

    def __init__(self):
        self._table = None
        self.tracked_metrics = {}
        self.formatters = {}

    def get_table(self, stage: 'Stage'):
        if self._table is None:
            self._table = ProgressTable(file=sys.stdout if is_root() else DevNullIO(), interactive=0)
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
