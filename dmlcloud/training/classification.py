import torch
from torch import nn

from ..metrics import accuracy, top5_error
from .trainer import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    def forward_step(self, batch_idx, batch):
        X, label = (tensor.to(self.device, non_blocking=True) for tensor in batch)
        pred = self.model(X)

        with torch.no_grad():
            self.log_metric('acc', accuracy(pred, label))
            self.log_metric('top5_error', top5_error(pred, label))

        return self.loss_fn(pred, label)

    def create_loss(self):
        return nn.CrossEntropyLoss()

    def metric_names(self):
        return ['train/acc', 'val/acc', 'val/top5_error']
