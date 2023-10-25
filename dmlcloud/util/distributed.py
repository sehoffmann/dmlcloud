import numpy as np
import torch.distributed as dist


def print_worker(msg, barrier=True, flush=True):
    if barrier:
        dist.barrier()
    print(f'Worker {dist.get_rank()} ({dist.get_group_rank()}.{dist.get_process_group_ranks()}): {msg}', flush=flush)
    if barrier:
        dist.barrier()


def shard_indices(n, rank, size, shuffle=True, drop_remainder=False, seed=0):
    indices = np.arange(n)

    if shuffle:
        np.random.Generator(np.random.MT19937(seed)).shuffle(indices)

    if drop_remainder:
        indices = indices[: n - n % size]

    return indices[rank::size]