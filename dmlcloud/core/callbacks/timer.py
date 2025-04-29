from datetime import datetime
from typing import TYPE_CHECKING

from .common import Callback


if TYPE_CHECKING:
    from dmlcloud.core.stage import Stage


class TimerCallback(Callback):
    """
    A callback that logs the time taken for each epoch.
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.epoch_start_time = None
        self.epoch_end_time = None

    def pre_stage(self, stage: 'Stage'):
        self.start_time = datetime.now()

    def post_stage(self, stage: 'Stage'):
        self.end_time = datetime.now()

    def pre_epoch(self, stage: 'Stage'):
        self.epoch_start_time = datetime.now()

    def post_epoch(self, stage: 'Stage'):
        self.epoch_end_time = datetime.now()

        epoch_time = (stage.epoch_end_time - self.epoch_start_time).total_seconds()
        total_time = (stage.epoch_end_time - self.start_time).total_seconds()
        stage.log('misc/epoch_time', epoch_time, prefixed=False, log_step=False)
        stage.log('misc/total_time', total_time, prefixed=False, log_step=False)

        if stage._run_epoch_overridden:
            average_epoch_time = (stage.epoch_end_time - self.start_time) / (stage.current_epoch + 1)
            eta = average_epoch_time * (stage.max_epochs - stage.current_epoch - 1)
            stage.log('misc/eta', eta.total_seconds(), prefixed=False, log_step=False)
