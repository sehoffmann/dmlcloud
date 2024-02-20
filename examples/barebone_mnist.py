import sys
sys.path.insert(0, './')

import torch
import torch.distributed as dist
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from dmlcloud.pipeline import TrainingPipeline
from dmlcloud.stage import Stage


class MNISTStage(Stage):


    def pre_stage(self):
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        val_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

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


    def run_epoch(self):
        self.model.train()
        self.metric_prefix = 'train'
        for img, target in self.train_loader:
            img, target = img.to(self.pipeline.device), target.to(self.pipeline.device)
            self.optimizer.zero_grad()
            output = self.model(img)
            loss = self.loss(output, target)
            self._log_metrics(img, target, output, loss)
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        self.metric_prefix = 'val'
        for img, target in self.val_loader:
            img, target = img.to(self.pipeline.device), target.to(self.pipeline.device)
            with torch.no_grad():
                output = self.model(img)
                loss = self.loss(output, target)
                self._log_metrics(img, target, output, loss)


    def table_columns(self):
        columns = super().table_columns()
        columns.insert(1, {'name': '[Train] Loss', 'metric': 'train/loss'})
        columns.insert(2, {'name': '[Train] Acc.', 'metric': 'train/accuracy'})
        columns.insert(3, {'name': '[Val] Loss', 'metric': 'val/loss'})
        columns.insert(4, {'name': '[Val] Acc.', 'metric': 'val/accuracy'})
        return columns



def main():
    dist.init_process_group(init_method='tcp://localhost:23456', rank=0, world_size=1)

    pipeline = TrainingPipeline({})
    pipeline.append_stage(MNISTStage(), max_epochs=10)
    pipeline.run()


if __name__ == '__main__':
    main()