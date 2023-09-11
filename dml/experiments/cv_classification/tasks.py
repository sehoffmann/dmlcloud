import torchvision.datasets as datasets

from . import transform
from .models import MODEL_CONFIGS

CropType = transform.CropType


class Task:
    def __init__(
        self,
        name,
        dataset_cls,
        num_classes,
        input_channels,
        img_size,
        resize_size=None,
        mean=None,
        std=None,
        crop_type=CropType.NONE,
        crop_padding=0,
        random_flips=True,
    ):
        self.name = name
        self.dataset_cls = dataset_cls
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.img_size = img_size
        self.resize_size = resize_size
        self.mean = mean or tuple([0.5 for _ in range(self.input_channels)])
        self.std = std or tuple([0.5 for _ in range(self.input_channels)])
        self.crop_type = crop_type
        self.crop_padding = crop_padding
        self.random_flips = random_flips

    def default_config(self, config):
        config.train_transform_preset = 'none'
        config.val_transform_preset = 'none'

    def postprocess_config(self, config):
        config.dct['model'] = MODEL_CONFIGS[config.model_preset]
        config.dct['train_transform'] = transform.create_config_dict(self, config.train_transform_preset)
        config.dct['val_transform'] = transform.create_config_dict(self, config.val_transform_preset)

    def create_datasets(self, config, path, train_transform, val_transform):
        train = self.dataset_cls(root=path, train=True, transform=train_transform, download=True)
        test = self.dataset_cls(root=path, train=False, transform=val_transform)
        return train, test


class CIFAR10(Task):
    def __init__(self):
        super().__init__(
            'cifar10',
            datasets.CIFAR10,
            10,
            3,
            32,
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
            crop_type=CropType.RANDOM,
            crop_padding=4,
        )

    def default_config(self, config):
        super().default_config(config)
        config.model_preset = 'resnet18'
        config.train_transform_preset = 'cifar10'
        config.epochs = 50


class CIFAR100(Task):
    def __init__(self):
        super().__init__(
            'cifar100',
            datasets.CIFAR100,
            100,
            3,
            32,
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761),
            crop_type=CropType.RANDOM,
            crop_padding=4,
        )

    def default_config(self, config):
        super().default_config(config)
        config.model_preset = 'cnn'
        config.train_transform_preset = 'cifar10'
        config.epochs = 30


class ImageNet(Task):
    def __init__(self):
        super().__init__(
            'imagenet',
            datasets.ImageNet,
            1000,
            3,
            224,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            crop_type=CropType.RANDOM_RESIZED,
            resize_size=256,
        )

    def default_config(self, config):
        super().default_config(config)
        config.model_preset = 'resnet18'
        config.train_transform_preset = 'imagenet'
        config.val_transform_preset = 'cv_eval'
        config.epochs = 90

    def create_datasets(self, config, path, train_transform, val_transform):
        train = self.dataset_cls(root=path, split='train', transform=train_transform)
        val = self.dataset_cls(root=path, split='val', transform=train_transform)
        return train, val


class EMNIST(Task):
    def __init__(self):
        super().__init__('emnist', datasets.EMNIST, 47, 1, 28, mean=(0.5,), std=(0.5,), random_flips=False)

    def default_config(self, config):
        super().default_config(config)
        config.model_preset = 'cnn'
        config.epochs = 30


class FashionMNIST(Task):
    def __init__(self):
        super().__init__('fashion_mnist', datasets.FashionMNIST, 10, 1, 28, mean=(0.5,), std=(0.5,))

    def default_config(self, config):
        super().default_config(config)
        config.model_preset = 'cnn'
        config.epochs = 30


class SVHN(Task):
    def __init__(self):
        super().__init__('svhn', datasets.SVHN, 10, 3, 32, mean=(0.5,), std=(0.5,))

    def default_config(self, config):
        super().default_config(config)
        config.model_preset = 'cnn'
        config.train_transform_preset = 'svhn'
        config.epochs = 30


_tasks = [CIFAR10(), CIFAR100(), ImageNet(), EMNIST(), FashionMNIST(), SVHN()]
TASKS = {task.name: task for task in _tasks}
