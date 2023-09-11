import horovod.torch as hvd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from ...training import ClassificationTrainer
from .tasks import TASKS


class CVClassificationTrainer(ClassificationTrainer):
    def create_model(self):
        create_fn = self.cfg.dct['model']['create_fn']
        kwargs = dict(self.cfg.dct['model'])
        kwargs.pop('create_fn')
        return create_fn(TASKS[self.cfg.task], **kwargs)

    def create_transform(self, train=True):
        dct = self.cfg.dct['train_transform'] if train else self.cfg.dct['val_transform']
        kwargs = dict(dct)
        create_fn = kwargs.pop('create_fn')
        return create_fn(**kwargs)

    def create_dataset(self):
        task = TASKS[self.cfg.task]

        train_transform = self.create_transform(train=True)
        val_transform = self.create_transform(train=False)

        path = self.cfg.data_dir if self.cfg.direct_path else self.cfg.data_dir / task.name
        train, test = task.create_datasets(self.cfg, path, train_transform, val_transform)

        train_sampler = torch.utils.data.DistributedSampler(
            train, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True, seed=self.cfg.seed
        )
        val_sampler = torch.utils.data.DistributedSampler(
            test, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False, seed=self.cfg.seed
        )

        train_dl = torch.utils.data.DataLoader(
            train, batch_size=self.cfg.batch_size, sampler=train_sampler, num_workers=4
        )
        test_dl = torch.utils.data.DataLoader(test, batch_size=self.cfg.batch_size, sampler=val_sampler, num_workers=4)

        return train_dl, test_dl

    def create_optimizer(self, params, lr):
        return torch.optim.AdamW(params, lr=lr, weight_decay=self.cfg.weight_decay)

    def create_scheduler(self):
        return CosineAnnealingLR(self.optimizer, T_max=self.cfg.epochs, eta_min=1e-7)
