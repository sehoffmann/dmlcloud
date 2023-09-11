from pathlib import Path

from ...training.config import DefaultConfig, SubConfig
from .experiment import CVClassificationTrainer
from .models import MODEL_CONFIGS
from .tasks import TASKS
from .transform import TRANSFORM_PRESETS


class ExperimentConfig(SubConfig):
    def set_defaults(self):
        self.task = None
        self.data_dir = Path('./data').resolve()
        self.direct_path = False
        self.model_preset = None
        self.train_transform_preset = None
        self.eval_transform_preset = None

    def add_arguments(self, parser):
        parser.add_argument('--model', choices=list(MODEL_CONFIGS.keys()), default=None, help='The model to use')
        parser.add_argument('--data', default=None, help='Path to the dataset')
        parser.add_argument('--direct-path', action='store_true', help='Whether to use the direct path for the dataset')
        parser.add_argument(
            '--train-transform', choices=TRANSFORM_PRESETS, default=None, help='The transform to use for training'
        )
        parser.add_argument(
            '--eval-transform', choices=TRANSFORM_PRESETS, default=None, help='The transform to use for evaluation'
        )

    def parse_args(self, args):
        self.task = args.task
        if args.model is not None:
            self.model_preset = args.model
        if args.data is not None:
            self.data_dir = Path(args.data).resolve()
        if args.direct_path:
            self.direct_path = True
        if args.train_transform is not None:
            self.train_transform_preset = args.train_transform
        if args.eval_transform is not None:
            self.eval_transform_preset = args.eval_transform

    @property
    def task(self):
        return self.dct['task']

    @task.setter
    def task(self, value):
        self.dct['task'] = value

    @property
    def data_dir(self):
        return Path(self.dct['data_dir'])

    @data_dir.setter
    def data_dir(self, value):
        self.dct['data_dir'] = str(value)

    @property
    def model_preset(self):
        return self.dct['model_preset']

    @model_preset.setter
    def model_preset(self, value):
        self.dct['model_preset'] = value

    @property
    def train_transform_preset(self):
        return self.dct['train_transform_preset']

    @train_transform_preset.setter
    def train_transform_preset(self, value):
        self.dct['train_transform_preset'] = value

    @property
    def eval_transform_preset(self):
        return self.dct['eval_transform_preset']

    @eval_transform_preset.setter
    def eval_transform_preset(self, value):
        self.dct['eval_transform_preset'] = value


def create_config():
    cfg = DefaultConfig()
    cfg.set_sub_config('experiment', ExperimentConfig)
    cfg.trainer_cls = CVClassificationTrainer
    return cfg


def create_trainer(args):
    cfg = create_config()
    task = TASKS[args.task]

    task.default_config(cfg)
    cfg.parse_args(args)
    task.postprocess_config(cfg)
    return CVClassificationTrainer(cfg)


def add_parser(subparsers):
    for name in TASKS:
        parser = subparsers.add_parser(name)
        parser.set_defaults(create_trainer=create_trainer, task=name)
        create_config().create_parser(parser)
