import os
import sys

import horovod.torch as hvd
import numpy as np
import torch


def hvd_print_worker(msg, barrier=True, flush=True):
    if barrier:
        hvd.barrier()
    print(f'Worker {hvd.rank()} ({hvd.cross_rank()}.{hvd.local_rank()}): {msg}', flush=flush)
    if barrier:
        hvd.barrier()


def setup_horovod(print_status=True):
    hvd.init()
    n_tasks = int(os.environ.get('SLURM_NTASKS', 0))
    if n_tasks > 1 and hvd.size() == 1:
        print(
            'CRITICAL: Horovod only sees a single task! Run "horovodrun --check-build" an verify that MPI is supported. Terminating...'
        )
        sys.exit(1)

    if print_status:
        hvd_print_worker('STARTED')

    hvd.barrier()  # make sure that all processes are running at this point
    # this is very important, otherwise subsequent broadcast operations might time out


def hvd_is_initialized():
    try:
        hvd.size()
        return True
    except ValueError:
        return False


def hvd_allreduce(val, *args, **kwargs):
    tensor = torch.as_tensor(val)
    reduced = hvd.allreduce(tensor, *args, **kwargs)
    return reduced.cpu().numpy()


def shard_indices(n, rank, size, shuffle=True, drop_remainder=False, seed=0):
    indices = np.arange(n)

    if shuffle:
        np.random.Generator(np.random.MT19937(seed)).shuffle(indices)

    if drop_remainder:
        indices = indices[: n - n % size]

    return indices[rank::size]
