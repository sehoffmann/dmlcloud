from .common import ArgparseVar, SubConfig


class WandbConfig(SubConfig):
    wb_project = ArgparseVar('-p', '--project', type=str, help='The wandb project name')
    wb_experiment = ArgparseVar('-e', '--exp', type=str, help='The wandb experiment name')
    wb_tags = ArgparseVar('--tags', nargs='+', type=str, help='The wandb tags')

    def set_defaults(self):
        self.wb_project = None
        self.wb_experiment = None
        self.wb_tags = []
