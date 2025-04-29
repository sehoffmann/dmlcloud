from typing import TYPE_CHECKING

from omegaconf import OmegaConf

from dmlcloud.core.callbacks import Callback
from dmlcloud.util.wandb import wandb_is_initialized, wandb_set_startup_timeout


if TYPE_CHECKING:
    from dmlcloud.core.pipeline import Pipeline
    from dmlcloud.core.stage import Stage


class WandbInitCallback(Callback):
    """
    A callback that initializes Weights & Biases and closes it at the end.
    This is separated from the WandbLoggerCallback to ensure it is called right at the beginning of training.
    """

    def __init__(self, project, entity, group, tags, startup_timeout, **kwargs):
        try:
            import wandb
        except ImportError:
            raise ImportError('wandb is required for the WandbInitCallback')

        self.wandb = wandb
        self.project = project
        self.entity = entity
        self.group = group
        self.tags = tags
        self.startup_timeout = startup_timeout
        self.kwargs = kwargs

    def pre_run(self, pipe: 'Pipeline'):
        wandb_set_startup_timeout(self.startup_timeout)
        self.wandb.init(
            config=OmegaConf.to_container(pipe.config, resolve=True),
            name=pipe.name,
            project=self.project,
            entity=self.entity,
            group=self.group,
            tags=self.tags,
            **self.kwargs,
        )

    def cleanup(self, pipe, exc_type, exc_value, traceback):
        if wandb_is_initialized():
            self.wandb.finish(exit_code=0 if exc_type is None else 1)


class WandbLoggerCallback(Callback):
    """
    A callback that logs metrics to Weights & Biases.
    """

    def __init__(self):
        try:
            import wandb
        except ImportError:
            raise ImportError('wandb is required for the WandbLoggerCallback')

        self.wandb = wandb

    def post_epoch(self, stage: 'Stage'):
        metrics = stage.history.last()
        self.wandb.log(metrics, commit=True, step=stage.current_epoch)
