import os

import numpy as np
import torch.distributed as dist

from .tcp import find_free_port, get_local_ips


def is_root_rank():
    return dist.get_rank() == 0


def root_only(fn):
    """
    Decorator for methods that should only be called on the root rank.
    """

    def wrapper(self, *args, **kwargs):
        if self.is_root_rank:
            return fn(self, *args, **kwargs)

    return wrapper

def mpi_local_comm():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    local_comm = comm.split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
    return local_comm


def local_rank():
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ["LOCAL_RANK"])
    try:
        return mpi_local_comm().get_rank()
    except:
        pass
    raise ValueError('Can not determine local rank')
    

def local_size():
    if 'LOCAL_WORLD_SIZE' in os.environ:
        return int(os.environ["LOCAL_WORLD_SIZE"])
    try:
        return mpi_local_comm().get_size()
    except:
        pass
    raise ValueError('Can not determine local world size')


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


def init_MPI_process_group(ip_idx=0, port=None, verbose=False, **kwargs):
    """
    This method setups up the distributed backend using MPI, even
    if torch was not built with MPI support. For this to work, you
    need to have mpi4py installed and the root rank must be reachable
    via TCP.

    If port is None, we will automatically try to find a free port.

    ip_idx can be used to specify which IP address to use if the root
    has multiple IP addresses. The default is 0, which means the first.

    kwargs are passed to torch.distributed.init_process_group.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    port = find_free_port() if port is None and rank == 0 else None

    if rank == 0:
        ip = get_local_ips()[ip_idx]
    else:
        ip = None

    ip = comm.bcast(ip, root=0)
    port = comm.bcast(port, root=0)
    url = f'tcp://{ip}:{port}'

    if verbose and rank == 0:
        print(f'Initializing torch.distributed using url {url}', flush=True)
    comm.Barrier()

    dist.init_process_group(
        init_method=url,
        world_size=size,
        rank=rank,
        **kwargs,
    )

    return rank, size
