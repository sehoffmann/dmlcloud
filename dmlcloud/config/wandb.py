from .common import ArgparseVar, SubConfig


class WandbConfig(SubConfig):
    wb_project = ArgparseVar('-p', '--project', type=str, help='The wandb project name')
    wb_name = ArgparseVar('--wb-name', type=str, help='Can be used to override the wandb experiment name')
    wb_tags = ArgparseVar('--tags', nargs='+', type=str, help='The wandb tags')

    def set_defaults(self):
        self.wb_project = None
        self.wb_name = None
        self.wb_tags = []
