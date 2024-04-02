import os
import sys


class WandbModuleWrapper:
    def __getattr__(self, name):
        import wandb

        return getattr(wandb, name)

    def __setattr__(self, name, value):
        import wandb

        setattr(wandb, name, value)


wandb = WandbModuleWrapper()


def wandb_set_startup_timeout(seconds: int):
    assert isinstance(seconds, int)
    os.environ['WANDB__SERVICE_WAIT'] = f'{seconds}'


def wandb_is_imported():
    return 'wandb' in sys.modules


def wandb_is_initialized():
    return wandb.run is not None
