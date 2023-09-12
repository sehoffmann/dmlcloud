import sys

import pytest
import torch
from dmlcloud.config import DefaultConfig
from dmlcloud.training import BaseTrainer, ClassificationTrainer


class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        return torch.randn(10), torch.randint(0, 10, size=(1,)).item()


class SmokeTrainer(BaseTrainer):
    def create_dataset(self):
        train_dl = torch.utils.data.DataLoader(DummyDataset(), batch_size=4)
        val_dl = torch.utils.data.DataLoader(DummyDataset(), batch_size=4)
        return train_dl, val_dl

    def create_model(self):
        return torch.nn.Linear(10, 10)

    def create_loss(self):
        return torch.nn.CrossEntropyLoss()

    def create_optimizer(self, params, lr):
        return torch.optim.SGD(params, lr=lr)

    def forward_step(self, batch_idx, batch):
        x, y = (tensor.to(self.device) for tensor in batch)
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        return loss


class SmokeClassificationTrainer(ClassificationTrainer):
    def create_dataset(self):
        train_dl = torch.utils.data.DataLoader(DummyDataset(), batch_size=4)
        val_dl = torch.utils.data.DataLoader(DummyDataset(), batch_size=4)
        return train_dl, val_dl

    def create_model(self):
        return torch.nn.Linear(10, 10)

    def create_optimizer(self, params, lr):
        return torch.optim.SGD(params, lr=lr)


class TestSmoke:
    def test_smoke(self):
        cfg = DefaultConfig()
        trainer = SmokeTrainer(cfg)
        trainer.train(use_checkpointing=False, use_wandb=False, print_diagnostics=False)

    def test_classification_smoke(self):
        cfg = DefaultConfig()
        trainer = SmokeClassificationTrainer(cfg)
        trainer.train(use_checkpointing=False, use_wandb=False, print_diagnostics=False)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
