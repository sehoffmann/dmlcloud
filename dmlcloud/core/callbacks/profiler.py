from typing import TYPE_CHECKING

from torch.profiler import profile, ProfilerActivity

from dmlcloud.core.distributed import is_root

from .common import Callback


if TYPE_CHECKING:
    from dmlcloud.core.stage import Stage


class ProfilerCallback(Callback):
    """
    A callback that profiles the training process and saves the results to a file.
    """

    def __init__(self, epochs=None, record_shapes=False, schedule=None):
        self.epochs = epochs
        self.record_shapes = record_shapes
        self.schedule = schedule

        self.profiler = None
        self._capturing = False

    def pre_epoch(self, stage: 'Stage'):
        if self.epochs and stage.current_epoch not in self.epochs:
            return

        self.profiler = profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            record_shapes=self.record_shapes,
            schedule=self.schedule,
        )
        self.profiler.__enter__()
        self._capturing = True

    def post_epoch(self, stage):
        if self.epochs and (stage.current_epoch - 1) not in self.epochs:
            return

        self.profiler.__exit__(None, None, None)
        self._capturing = False

        if is_root():
            print(self.profiler.key_averages().table(sort_by="self_cuda_time_total"))

        if stage.run_dir:
            outfile = str(stage.run_dir / f'{stage.name}_epoch{stage.current_epoch - 1}_trace.json')
            self.profiler.export_chrome_trace(outfile)

    def cleanup(self, pipe, exc_type, exc_value, traceback):
        if self._capturing:
            self.profiler.__exit__(exc_type, exc_value, traceback)
            self._capturing = False
