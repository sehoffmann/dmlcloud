import argparse

import horovod.torch as hvd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torchvision.datasets import MNIST

from ..training import ClassificationTrainer
from ..training.config import DefaultConfig, SubConfig


class MNISTModelConfig(SubConfig):
    def set_defaults(self):
        self.model_type = 'MLP'

    def add_arguments(self, parser):
        parser.add_argument('--model', choices=['MLP', 'CNN'], default=None, help='The model to use')

    def parse_args(self, args):
        if args.model is not None:
            self.model_type = args.model

    @property
    def model_type(self):
        return self.dct['type']

    @model_type.setter
    def model_type(self, value):
        self.dct['type'] = value


class MNISTTrainer(ClassificationTrainer):
    def create_model(self):
        if self.cfg.model_type == 'MLP':
            return nn.Sequential(
                *[
                    nn.Flatten(),
                    nn.Linear(28 * 28, 128),
                    nn.ReLU(),
                    nn.Linear(128, 10),
                ]
            )
        elif self.cfg.model_type == 'CNN':
            return nn.Sequential(
                nn.Conv2d(1, 16, 3, 1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, 2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(14 * 14 * 32, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
            )

    def create_dataset(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        train = MNIST(root='mnist', train=True, transform=transform, download=True)
        test = MNIST(root='mnist', train=False, transform=transform, download=True)
        train_sampler = torch.utils.data.DistributedSampler(
            train, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True, seed=self.cfg.seed
        )
        val_sampler = torch.utils.data.DistributedSampler(
            test, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False, seed=self.cfg.seed
        )
        train_dl = torch.utils.data.DataLoader(train, batch_size=self.cfg.batch_size, sampler=train_sampler)
        test_dl = torch.utils.data.DataLoader(test, batch_size=self.cfg.batch_size, sampler=val_sampler)
        return train_dl, test_dl

    def create_optimizer(self, params, lr):
        return torch.optim.AdamW(params, lr=lr, weight_decay=self.cfg.weight_decay)

    def create_scheduler(self):
        return CosineAnnealingLR(self.optimizer, T_max=self.cfg.epochs, eta_min=1e-7)


def create_config():
    cfg = DefaultConfig()
    cfg.set_sub_config('model', MNISTModelConfig)
    cfg.trainer_cls = MNISTTrainer
    return cfg


def create_mnist_trainer(args):
    cfg = create_config()
    cfg.parse_args(args)
    return MNISTTrainer(cfg)


def add_parser(subparsers):
    parser = subparsers.add_parser('mnist', help='Train a model on MNIST')
    parser.set_defaults(create_trainer=create_mnist_trainer)
    create_config().create_parser(parser)


def main():
    parser = argparse.ArgumentParser()
    add_parser(parser)
    args = parser.parse_args()
    trainer = create_mnist_trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
