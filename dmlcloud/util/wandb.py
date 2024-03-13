import os


def wandb_set_startup_timeout(seconds: int):
    assert isinstance(seconds, int)
    os.environ['WANDB__SERVICE_WAIT'] = f'{seconds}'


def wandb_is_initialized():
    import wandb

    return wandb.run is not None
