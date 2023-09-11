import sys
from pathlib import Path

from ...git import git_hash
from .common import ArgparseVar, ConfigVar, SubConfig


class MetaConfig(SubConfig):
    trainer_cls = ConfigVar()
    project_dir = ArgparseVar('-d', '--dir', type=Path, help='The project directory')
    model_dir = ConfigVar()
    id_prefix = ArgparseVar('--prefix', type=str, help='The id prefix for the experiment')
    command_line = ConfigVar()
    git_hash = ConfigVar()

    def set_defaults(self):
        self.trainer_cls = None
        self.project_dir = Path('./').resolve()
        self.model_dir = None
        self.id_prefix = ''
        self.command_line = ' '.join(sys.argv)
        self.git_hash = git_hash()
