from pathlib import Path
from typing import TYPE_CHECKING, Union

from dmlcloud.core.callbacks import Callback

if TYPE_CHECKING:
    from dmlcloud.core.stage import Stage


class TensorboardCallback(Callback):
    """
    A callback that logs metrics to Tensorboard.
    """

    def __init__(self, log_dir: Union[str, Path]):
        self.log_dir = Path(log_dir)
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        except ImportError:
            raise ImportError('tensorflow is required for the TensorboardCallback')

    def pre_run(self, pipe):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=self.log_dir)

    def post_epoch(self, stage: 'Stage'):
        metrics = stage.history.last()
        for key, value in metrics.items():
            if value is not None:
                self.writer.add_scalar(key, value, stage.current_epoch)

    def cleanup(self, pipe, exc_type, exc_value, traceback):
        if self.writer is not None:
            self.writer.close()
