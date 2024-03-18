import os
from contextlib import contextmanager

import torch.distributed as dist

from .tcp import find_free_port, get_local_ips


def is_root():
    return dist.get_rank() == 0


def root_only(fn):
    """
    Decorator for methods that should only be called on the root rank.
    """

    def wrapper(*args, **kwargs):
        if is_root():
            return fn(*args, **kwargs)

    return wrapper


@contextmanager
def root_first():
    """
    Context manager that ensures that the root rank executes the code first before all other ranks
    """
    if is_root():
        try:
            yield
        finally:
            dist.barrier()
    else:
        dist.barrier()
        try:
            yield
        finally:
            pass


def mpi_local_comm():
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
        return local_comm
    except ImportError:
        return None


def local_rank():
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ["LOCAL_RANK"])
    local_comm = mpi_local_comm()
    if local_comm is not None:
        return local_comm.Get_rank()
    else:
        return None


def local_size():
    if 'LOCAL_WORLD_SIZE' in os.environ:
        return int(os.environ["LOCAL_WORLD_SIZE"])
    local_comm = mpi_local_comm()
    if local_comm is not None:
        return local_comm.Get_size()
    else:
        return None


def print_worker(msg, barrier=True, flush=True):
    if barrier:
        dist.barrier()
    print(f'Worker {dist.get_rank()} ({dist.get_group_rank()}.{dist.get_process_group_ranks()}): {msg}', flush=flush)
    if barrier:
        dist.barrier()


def init_process_group_dummy():
    """
    Initializes the process group with a single process.
    Uses HashStore under the hood. Useful for applications that
    only run on a single gpu.
    """
    store = dist.HashStore()
    dist.init_process_group(store=store, rank=0, world_size=1, backend='gloo')


def init_process_group_MPI(ip_idx=0, port=None, **kwargs):
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

    comm.Barrier()

    dist.init_process_group(
        init_method=url,
        world_size=size,
        rank=rank,
        **kwargs,
    )

    return rank, size


def init_process_group_auto(ip_idx=0, port=None, **kwargs):
    """
    Tries to initialize torch.distributed in the following order:
    1. If the MASTER_PORT environment variable is set, use environment variable initialization
    2. If a MPI context is available, e.g. from slurm or mpirun, use MPI to exchange ip addresses (see init_process_group_MPI)
    3. Otherwise, use a single process group (see init_process_group_dummy)
    """

    # determine init method
    method = 'dummy'
    if os.environ.get('MASTER_PORT'):
        method = 'env'
    else:
        try:
            from mpi4py import MPI

            if MPI.COMM_WORLD.Get_size() > 1:
                method = 'MPI'
        except ImportError:
            pass

    if method == 'env':
        dist.init_process_group(init_method='env://', **kwargs)
    elif method == 'MPI':
        init_process_group_MPI(ip_idx=ip_idx, port=port, **kwargs)
    else:
        init_process_group_dummy()


def deinitialize_torch_distributed():
    """
    Deinitializes the torch distributed framework.
    At the time of writing, `dist.destroy_process_group()` is not well documented.
    Hence, this function.
    """
    dist.destroy_process_group()
