from enum import IntEnum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from dmlcloud.core.pipeline import Pipeline
    from dmlcloud.core.stage import Stage


class CallbackList:
    """
    A priority queue of callbacks.
    """

    def __init__(self):
        self.callbacks = []

    def append(self, callback: 'Callback', priority: int = 0):
        """
        Append a callback to the list with the given priority.

        Args:
            callback (Callback): The callback to append.
            priority (int, optional): The priority of the callback. Defaults to 0.
        """
        self.callbacks.append((priority, callback))

    def __iter__(self):
        for _, callback in sorted(self.callbacks, key=lambda x: x[0]):
            yield callback

    def __len__(self):
        return len(self.callbacks)

    def __add__(self, other: 'CallbackList'):
        result = CallbackList()
        result.callbacks = self.callbacks + other.callbacks
        return result


class CbPriority(IntEnum):
    """
    Default priorities for callbacks used by the pipeline and stage classes.
    """

    WANDB_INIT = -200
    CHECKPOINT = -190
    STAGE_TIMER = -180
    DIAGNOSTICS = -170
    CUDA = -160
    GIT = -150
    METRIC_REDUCTION = -100

    OBJECT_METHODS = 0

    PROFILER = 100
    WANDB_LOGGER = 110
    CSV = 110
    TENSORBOARD = 110
    TABLE = 120


class Callback:
    """
    A callback that can be registered to a stage or the whole pipeline to receive updates on the training progress.
    """

    def pre_run(self, pipe: 'Pipeline'):
        """
        Executed before the pipeline starts.
        """
        pass

    def post_run(self, pipe: 'Pipeline'):
        """
        Executed after the pipeline finishes.
        """
        pass

    def cleanup(self, pipe: 'Pipeline', exc_type, exc_value, traceback):
        """
        Executed after the pipeline finishes, even if an error occurred.
        E.g. to close file handles.

        Args:
            pipe (Pipeline): The pipeline that is being cleaned up.
            exc_type (type): The type of the exception that caused the cleanup or None if no exception occurred.
            exc_value (Exception): The exception that caused the cleanup or None if no exception occurred.
            traceback (Traceback): The traceback of the exception that caused the cleanup or None if no exception occurred.
        """
        pass

    def pre_stage(self, stage: 'Stage'):
        """
        Executed before the stage starts.
        """
        pass

    def post_stage(self, stage: 'Stage'):
        """
        Executed after the stage finishes.
        """
        pass

    def pre_epoch(self, stage: 'Stage'):
        """
        Executed before each epoch.
        """
        pass

    def post_epoch(self, stage: 'Stage'):
        """
        Executed after each epoch.
        """
        pass

    def post_step(self, stage: 'Stage'):
        """
        Executed after each step. Stage must call `finish_step` to trigger this callback.
        """
        pass
