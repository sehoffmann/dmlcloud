import sys
sys.path.insert(0, './')

import torch
import torch.distributed as dist
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from dmlcloud.pipeline import TrainingPipeline
from dmlcloud.stage import Stage
from dmlcloud.util.distributed import init_process_group_auto, is_root, root_first


class MNISTStage(Stage):


    def pre_stage(self):
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        with root_first():
            train_dataset = datasets.MNIST(root='data', train=True, download=is_root(), transform=transform)
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            self.train_loader = DataLoader(train_dataset, batch_size=32, sampler=self.train_sampler)

            val_dataset = datasets.MNIST(root='data', train=False, download=is_root(), transform=transform)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            self.val_loader = DataLoader(val_dataset, batch_size=32, sampler=val_sampler)

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(784, 10),  
        ).to(self.pipeline.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss = nn.CrossEntropyLoss()


    def _log_metrics(self, img, target, output, loss):
        self.track_reduce('loss', loss)
        self.track_reduce('accuracy', (output.argmax(1) == target).float().mean())


    def _train_epoch(self):
        self.model.train()
        self.metric_prefix = 'train'
        self.train_sampler.set_epoch(self.current_epoch)

        for img, target in self.train_loader:
            img, target = img.to(self.pipeline.device), target.to(self.pipeline.device)

            self.optimizer.zero_grad()
            output = self.model(img)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            self._log_metrics(img, target, output, loss)


    @torch.no_grad()
    def _val_epoch(self):
        self.model.eval()
        self.metric_prefix = 'val'

        for img, target in self.val_loader:
            img, target = img.to(self.pipeline.device), target.to(self.pipeline.device)

            output = self.model(img)
            loss = self.loss(output, target)

            self._log_metrics(img, target, output, loss)


    def run_epoch(self):
        self._train_epoch()
        self._val_epoch()        


    def table_columns(self):
        columns = super().table_columns()
        columns.insert(1, {'name': '[Train] Loss', 'metric': 'train/loss'})
        columns.insert(2, {'name': '[Val] Loss', 'metric': 'val/loss'})
        columns.insert(3, {'name': '[Train] Acc.', 'metric': 'train/accuracy'})
        columns.insert(4, {'name': '[Val] Acc.', 'metric': 'val/accuracy'})
        return columns



def main():
    init_process_group_auto()

    pipeline = TrainingPipeline({})
    pipeline.append_stage(MNISTStage(), max_epochs=10)
    pipeline.run()


if __name__ == '__main__':
    main()