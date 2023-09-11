import csv

import horovod.torch as hvd
import torch

Var = 'var'
Std = 'std'
Statistics = 'statistics'


class Metric:
    def __init__(self, name, reduction=hvd.Average, allreduce=True):
        self.name = name
        self.reduction = reduction
        self.batch_values = []
        self.allreduce = allreduce
        assert not allreduce or reduction is not None, 'Cannot allreduce without reduction'

    def _reduce(self, value, dim=None):
        if value.dim() == 0:
            return value
        elif self.reduction == hvd.Average:
            return value.mean(dim=dim)
        elif self.reduction == hvd.Sum:
            return value.sum(dim=dim)
        elif self.reduction == hvd.Min:
            return value.min(dim=dim)[0]
        elif self.reduction == hvd.Max:
            return value.max(dim=dim)[0]
        elif self.reduction == hvd.Product:
            return value.prod(dim=dim)
        elif self.reduction == Var:
            return value.var(dim=dim)
        elif self.reduction == Std:
            return value.std(dim=dim)
        else:
            raise ValueError(f'Unknown reduction {self.reduction}')

    def add_batch_value(self, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()  # this is very important to avoid memory leaks and for performance

        if self.reduction is None:
            self.batch_values.append(value)
        else:
            tensor = torch.as_tensor(value, device='cpu')
            tensor = self._reduce(tensor, dim=0)
            self.batch_values.append(tensor)

    def reduce_locally(self):
        if self.reduction is None:
            return self.batch_values[0]
        tensor = torch.stack(self.batch_values)
        tensor = self._reduce(tensor, dim=0)
        return tensor

    def reduce(self):
        tensor = self.reduce_locally()
        if self.allreduce:
            if self.reduction is Var:
                return hvd.allreduce_(
                    tensor, op=hvd.Average, name=f'metric/{self.name}'
                )  # not properly bessel corrected, but close enough
            elif self.reduction is Std:
                var = hvd.allreduce(tensor**2, op=hvd.Average, name=f'metric/{self.name}')
                return torch.sqrt(var)
            else:
                return hvd.allreduce_(tensor, op=self.reduction, name=f'metric/{self.name}')
        else:
            return tensor


class MetricSaver:
    def __init__(self, epochs=None):
        self.epochs = epochs or []
        self.current_metrics = {}

    @property
    def last(self):
        return self.epochs[-1]

    def get_metrics(self, name):
        return [epoch[name] for epoch in self.epochs]

    def reduce(self):
        reduced = {}
        for name, metric in self.current_metrics.items():
            reduced[name] = metric.reduce()
        self.epochs.append(reduced)
        self.current_metrics = {}

    def log_metric(self, name, value, reduction=hvd.Average, allreduce=True):
        if reduction == Statistics:
            self.log_metric(f'{name}/mean', value, reduction=hvd.Average, allreduce=allreduce)
            self.log_metric(f'{name}/std', value, reduction=Std, allreduce=allreduce)
            self.log_metric(f'{name}/min', value, reduction=hvd.Min, allreduce=allreduce)
            self.log_metric(f'{name}/max', value, reduction=hvd.Max, allreduce=allreduce)
        else:
            if name not in self.current_metrics:
                self.current_metrics[name] = Metric(name, reduction, allreduce)
            metric = self.current_metrics[name]
            metric.add_batch_value(value)

    def log_python_object(self, name, value):
        self.log_metric(name, value, reduction=None, allreduce=False)

    def scalar_metrics(self, with_epoch=False):
        scalars = []
        for epoch, metrics in enumerate(self.epochs):
            dct = {}
            if with_epoch:
                dct['epoch'] = epoch + 1

            for name, value in metrics.items():
                if isinstance(value, torch.Tensor) and value.dim() == 0:
                    dct[name] = value.item()
                elif not isinstance(value, torch.Tensor):
                    dct[name] = value
            scalars.append(dct)

        return scalars

    def scalars_to_csv(self, path):
        with open(path, 'w') as file:
            scalar_metrics = self.scalar_metrics(with_epoch=True)
            writer = csv.DictWriter(file, fieldnames=scalar_metrics[0].keys())
            writer.writeheader()
            writer.writerows(scalar_metrics)
