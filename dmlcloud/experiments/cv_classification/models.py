from torch import nn
from torchvision.models.resnet import ResNet, resnet101, resnet152, resnet18, resnet34, resnet50


def create_mlp(task, hidden_layers, layer_norm=False):
    hidden_layers = list(hidden_layers)
    hidden_layers.insert(0, task.input_channels * task.img_size * task.img_size)

    layers = [nn.Flatten()]
    for i in range(1, len(hidden_layers)):
        layers += [nn.Linear(hidden_layers[i - 1], hidden_layers[i], bias=not layer_norm)]
        if layer_norm:
            layers += [nn.LayerNorm(hidden_layers[i])]
        layers += [nn.ReLU()]
    layers += [nn.Linear(hidden_layers[-1], task.num_classes)]
    return nn.Sequential(*layers)


def create_cnn(task, blocks, instance_norm=False, kernel_size=3):
    last_channel = task.input_channels
    layers = []
    for block in blocks:
        for channels in block:
            layers += [nn.Conv2d(last_channel, channels, kernel_size, padding='same', bias=not instance_norm)]
            if instance_norm:
                layers += [nn.InstanceNorm2d(channels, affine=True)]
            layers += [nn.ReLU()]
            last_channel = channels
        layers += [nn.MaxPool2d(2)]
    layers += [
        nn.Flatten(),
        nn.Linear(last_channel * (task.img_size // 2 ** len(blocks)) ** 2, task.num_classes),
    ]
    return nn.Sequential(*layers)


def create_resnet(task, block, layers, **kwargs):
    return ResNet(block, layers, num_classes=task.num_classes, **kwargs)


def create_resnet18(task):
    return resnet18(num_classes=task.num_classes)


def create_resnet34(task):
    return resnet34(num_classes=task.num_classes)


def create_resnet50(task):
    return resnet50(num_classes=task.num_classes)


def create_resnet101(task):
    return resnet101(num_classes=task.num_classes)


def create_resnet152(task):
    return resnet152(num_classes=task.num_classes)


MODEL_CONFIGS = {
    'mlp_minimal': {
        'create_fn': create_mlp,
        'hidden_layers': [128],
        'layer_norm': False,
    },
    'mlp': {
        'create_fn': create_mlp,
        'hidden_layers': [128, 128, 128],
        'layer_norm': True,
    },
    'cnn_minimal': {
        'create_fn': create_cnn,
        'blocks': [[32, 32]],
        'instance_norm': False,
    },
    'cnn': {
        'create_fn': create_cnn,
        'blocks': [[32, 32], [64, 64], [128, 128]],
        'instance_norm': True,
    },
    'resnet18': {
        'create_fn': create_resnet18,
    },
    'resnet34': {
        'create_fn': create_resnet34,
    },
    'resnet50': {
        'create_fn': create_resnet50,
    },
    'resnet101': {
        'create_fn': create_resnet101,
    },
    'resnet152': {
        'create_fn': create_resnet152,
    },
}
