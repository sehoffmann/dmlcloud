import sys
from pathlib import Path

from dmlcloud.util import git_hash
from .common import ArgparseVar, ConfigVar, SubConfig


class MetaConfig(SubConfig):
    trainer_cls = ConfigVar()
    model_dir = ConfigVar()
    job_id = ConfigVar()
    command_line = ConfigVar()
    git_hash = ConfigVar()
    checkpoint_dir = ArgparseVar('-d', '--dir', type=Path, help='The directory where runs are stored')
    name = ArgparseVar('-n', '--name', help='The name of the experiment')

    def set_defaults(self):
        self.trainer_cls = None
        self.model_dir = None
        self.job_id = None
        self.command_line = ' '.join(sys.argv)
        self.git_hash = git_hash()
        self.checkpoint_dir = Path('./checkpoints').resolve()
        self.name = None
