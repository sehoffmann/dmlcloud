import horovod.torch as hvd


def scale_lr(lr, per_worker_batch_size, base_batch_size, use_adasum, use_gpu):
    if use_adasum and hvd.nccl_built() and use_gpu:
        lr_scaling = hvd.local_size()  # gpu adasum needs scaling by local size
    elif use_adasum:
        lr_scaling = 1.0  # cpu adasum doesn't need per_batch_scaling
    else:
        lr_scaling = hvd.size()
    lr_scaling *= per_worker_batch_size / base_batch_size
    return lr * lr_scaling, lr_scaling


def scale_beta1(beta1, per_worker_batch_size, base_batch_size):
    factor = hvd.size() * per_worker_batch_size / base_batch_size
    return beta1**factor


def scale_beta2(beta2, per_worker_batch_size, base_batch_size):
    factor = hvd.size() * per_worker_batch_size / base_batch_size
    return beta2**factor


def scale_param_group(param_group, config, use_gpu):
    lr_enabled = param_group['scale_lr'] if 'scale_lr' in param_group else config.scale_lr
    beta1_enabled = param_group['scale_beta1'] if 'scale_beta1' in param_group else config.scale_beta1
    beta2_enabled = param_group['scale_beta2'] if 'scale_beta2' in param_group else config.scale_beta2

    scaled_params = dict(param_group)
    if 'lr' in param_group and lr_enabled:
        scaled_params['lr'], _ = scale_lr(
            param_group['lr'], config.batch_size, config.base_batch_size, config.adasum, use_gpu
        )

    if 'betas' in param_group:
        if beta1_enabled:
            beta1 = scale_beta1(param_group['betas'][0], config.batch_size, config.base_batch_size)
        else:
            beta1 = param_group['betas'][0]
        if beta2_enabled:
            beta2 = scale_beta2(param_group['betas'][1], config.batch_size, config.base_batch_size)
        else:
            beta2 = param_group['betas'][1]
        scaled_params['betas'] = (beta1, beta2)

    if 'momentum' in param_group and beta1_enabled:
        scaled_params['momentum'] = scale_beta1(param_group['momentum'], config.batch_size, config.base_batch_size)

    return scaled_params
