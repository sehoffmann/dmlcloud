import os

import wandb


def wandb_set_startup_timeout(seconds: int):
    assert isinstance(seconds, int)
    os.environ['WANDB__SERVICE_WAIT'] = f'{seconds}'


def wandb_is_initialized():
    return wandb.run is not None
