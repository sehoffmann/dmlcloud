import torch.distributed as dist


def scale_lr(lr, per_worker_batch_size, base_batch_size):
    lr_scaling = dist.get_world_size()
    lr_scaling *= per_worker_batch_size / base_batch_size
    return lr * lr_scaling, lr_scaling


def scale_param_group(param_group, config):
    lr_enabled = param_group['scale_lr'] if 'scale_lr' in param_group else config.scale_lr

    scaled_params = dict(param_group)
    if 'lr' in param_group and lr_enabled:
        scaled_params['lr'], _ = scale_lr(
            param_group['lr'], config.batch_size, config.base_batch_size
        )

    return scaled_params
