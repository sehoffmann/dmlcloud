import os
from pathlib import Path
from typing import TYPE_CHECKING, Union

import dmlcloud.core.checkpoint as dml_checkpoint
from dmlcloud.util.logging import IORedirector
from .common import Callback

if TYPE_CHECKING:
    from dmlcloud.core.pipeline import Pipeline


class CheckpointCallback(Callback):
    """
    Creates the checkpoint directory and optionally setups io redirection.
    """

    def __init__(self, run_dir: Union[str, Path], redirect_io: bool = True):
        """
        Initialize the callback with the given path.

        Args:
            run_dir: The path to the checkpoint directory.
            redirect_io: Whether to redirect the IO to a file. Defaults to True.
        """
        self.run_dir = Path(run_dir)
        self.redirect_io = redirect_io
        self.io_redirector = None

    def pre_run(self, pipe: 'Pipeline'):
        if not dml_checkpoint.is_valid_checkpoint_dir(self.run_dir):
            dml_checkpoint.create_checkpoint_dir(self.run_dir)
            dml_checkpoint.save_config(pipe.config, self.run_dir)

        self.io_redirector = IORedirector(pipe.run_dir / 'log.txt')
        self.io_redirector.install()

        with open(pipe.run_dir / "environment.txt", 'w') as f:
            for k, v in os.environ.items():
                f.write(f"{k}={v}\n")

    def cleanup(self, pipe, exc_type, exc_value, traceback):
        if self.io_redirector is not None:
            self.io_redirector.uninstall()
