import argparse

from .meta import MetaConfig
from .training import TrainingConfig
from .wandb import WandbConfig


class BaseConfig:
    def __init__(self, dct=None):
        self._sub_configs = []
        self.dct = {}

        self._sub_configs = []
        self._setup_sub_configs()

        if dct:
            self.dct.update(dct)

    def __getattr__(self, name):
        if name == '_sub_configs':
            return super().__getattribute__(name)

        for cfg in self._sub_configs:
            try:
                return getattr(cfg, name)
            except AttributeError:
                pass
        raise AttributeError(f'Config has no attribute {name}')

    def __setattr__(self, name, value):
        if name != '_sub_configs':
            for cfg in self._sub_configs:
                if hasattr(cfg, name):
                    setattr(cfg, name, value)
                    return
        super().__setattr__(name, value)

    def _setup_sub_configs(self):
        raise NotImplementedError()

    def set_sub_config(self, key, cls):
        self.dct.setdefault(key, {})
        sub_cfg = cls(self, self.dct, key)
        self._sub_configs.append(sub_cfg)

    def create_parser(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        for sub_cfg in self._sub_configs:
            sub_cfg.add_arguments(parser)
        return parser

    def parse_args(self, args):
        for sub_cfg in self._sub_configs:
            sub_cfg.parse_args(args)

    def as_dictionary(self):
        return dict(self.dct)


class DefaultConfig(BaseConfig):
    def _setup_sub_configs(self):
        self.set_sub_config('meta', MetaConfig)
        self.set_sub_config('training', TrainingConfig)
        self.set_sub_config('wandb', WandbConfig)
