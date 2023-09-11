from enum import Enum

import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

TRANSFORM_PRESETS = [
    'none',
    'cv_eval',
    'rand_augment',
    'trivial_augment',
    'augmix',
    'cifar10',
    'imagenet',
    'svhn',
]


class CropType(Enum):
    NONE = 'none'
    RANDOM = 'random'
    CENTER = 'center'
    RANDOM_RESIZED = 'random_resized'


def default_cv_train_transform(
    mean,
    std,
    interpolation=InterpolationMode.BILINEAR,
    flip_prob=None,
    crop_size=None,
    crop_type=CropType.NONE,
    crop_padding=0,
    autoaug_policy=None,
    ra_magnitude=9,
    **kwargs,
):
    trans = []
    if crop_type == CropType.RANDOM_RESIZED:
        trans += [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
    elif crop_type == CropType.RANDOM:
        trans += [transforms.RandomCrop(crop_size, padding=crop_padding)]
    elif crop_type == CropType.CENTER:
        trans += [transforms.CenterCrop(crop_size)]
    if flip_prob:
        trans += [transforms.RandomHorizontalFlip(flip_prob)]

    if autoaug_policy == 'rand_augment':
        trans += [transforms.RandAugment(interpolation=interpolation, magnitude=ra_magnitude)]
    elif autoaug_policy == 'trivial_augment':
        trans += [transforms.TrivialAugmentWide(interpolation=interpolation)]
    elif autoaug_policy == 'augmix':
        trans += [transforms.AugMix(interpolation=interpolation)]
    elif autoaug_policy:
        policy = transforms.AutoAugmentPolicy(autoaug_policy)
        trans += [transforms.AutoAugment(policy, interpolation=interpolation)]

    trans += [
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(trans)


def resize_and_center_crop(mean, std, crop_size, resize_size, interpolation=InterpolationMode.BILINEAR, **kwargs):
    trans = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(trans)


def normalize_only(mean, std, **kwargs):
    return transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean, std),
        ]
    )


def create_config_dict(task, preset):
    dct = {
        'mean': task.mean,
        'std': task.std,
        'interpolation': InterpolationMode.BILINEAR,
    }
    if preset == 'none':
        dct['create_fn'] = normalize_only
    elif preset == 'cv_eval':
        dct['create_fn'] = resize_and_center_crop
        dct['crop_size'] = task.img_size
        dct['resize_size'] = task.resize_size
    else:
        dct['create_fn'] = default_cv_train_transform
        dct['flip_prob'] = 0.5 if task.random_flips else 0.0
        dct['crop_size'] = task.img_size
        dct['crop_type'] = task.crop_type
        dct['crop_padding'] = task.crop_padding
        dct['autoaug_policy'] = preset
        if preset == 'rand_augment':
            dct['ra_magnitude'] = 9
    return dct
