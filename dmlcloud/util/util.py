import argparse
import enum
import os

import horovod.torch as hvd
import torch
import wandb


def hvd_allreduce(val, *args, **kwargs):
    tensor = torch.as_tensor(val)
    reduced = hvd.allreduce(tensor, *args, **kwargs)
    return reduced.cpu().numpy()


def set_wandb_startup_timeout(seconds: int):
    assert isinstance(seconds, int)
    os.environ['WANDB__SERVICE_WAIT'] = f'{seconds}'


def is_wandb_initialized():
    return wandb.run is not None


def is_hvd_initialized():
    try:
        hvd.size()
        return True
    except ValueError:
        return False


class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    From https://stackoverflow.com/a/60750535/4546885
    """

    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.value for e in enum_type))

        super().__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum(values)
        setattr(namespace, self.dest, value)
