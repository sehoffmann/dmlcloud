from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
import xarray as xr
from torch.utils.data import get_worker_info, IterableDataset


def shard_indices(
    n: int, rank: int, world_size: int, shuffle: bool = False, drop_remainder: bool = True, seed: int = 0
) -> list[int]:
    indices = np.arange(n)

    if shuffle:
        np.random.Generator(np.random.MT19937(seed)).shuffle(indices)

    if drop_remainder:
        indices = indices[: n - n % world_size]

    return indices[rank::world_size].tolist()  # this also converts np.int64 to python's int


def sharded_xr_dataset(
    ds: xr.Dataset | xr.DataArray,
    chunk_size: int,
    dim: str,
    shuffle: bool = False,
    seed: int = 0,
    rank: int | None = None,
    world_size: int | None = None,
    process_group: dist.ProcessGroup | None = None,
    load: bool = False,
    load_kwargs: dict | None = None,
) -> Iterable[xr.Dataset | xr.DataArray]:
    num_total_elements = len(ds[dim])
    num_chunks = num_total_elements // chunk_size

    if rank is None:
        rank = dist.get_rank(process_group)
    if world_size is None:
        world_size = dist.get_world_size(process_group)

    chunk_indices = shard_indices(num_chunks, rank, world_size, shuffle=shuffle, drop_remainder=True, seed=seed)

    for chunk_idx in chunk_indices:
        start = chunk_idx * chunk_size
        end = start + chunk_size
        chunk = ds.isel({dim: slice(start, end)})

        if load:
            kwargs = load_kwargs or {}
            chunk.load(**kwargs)

        yield chunk


class ShardedXrDataset(IterableDataset):
    def __init__(
        self,
        ds: xr.Dataset | xr.DataArray,
        chunk_size: int,
        dim: str,
        shuffle: bool = False,
        seed: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
        process_group: dist.ProcessGroup | None = None,
        load: bool = False,
        load_kwargs: dict | None = None,
    ):
        self.ds = ds
        self.chunk_size = chunk_size
        self.dim = dim
        self.shuffle = shuffle
        self.seed = seed
        self.load = load
        self.load_kwargs = load_kwargs

        if rank is None:
            self.rank = dist.get_rank(process_group)
        else:
            self.rank = rank

        if world_size is None:
            self.world_size = dist.get_world_size(process_group)
        else:
            self.world_size = world_size

    def __len__(self):
        num_total_elements = len(self.ds[self.dim])
        num_chunks = num_total_elements // self.chunk_size
        return num_chunks // self.world_size

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            rank = self.rank
            world_size = self.world_size
        else:
            rank = self.rank * worker_info.num_workers + worker_info.id
            world_size = self.world_size * worker_info.num_workers

        return sharded_xr_dataset(
            self.ds,
            self.chunk_size,
            self.dim,
            self.shuffle,
            self.seed,
            rank,
            world_size,
            self.load,
            self.load_kwargs,
        )


def interleave_batches(
    iterable: Iterable[torch.Tensor], num_batches: int, pin_memory: bool = False
) -> Iterable[torch.Tensor]:
    """
    Interleaves batches from an iterable of batches.
    Important: Returned batches must be used immediately or copied to avoid overwriting.
    """
    if num_batches < 1:
        raise ValueError('num_batches must be greater than 0')

    if num_batches == 1:
        yield from iterable

    batches = []
    memory = None
    batch_size = None
    slice_size = None
    for batch in iterable:
        if memory is None:
            batch_size = batch.shape[0]
            slice_size = batch_size // num_batches
            if batch_size % num_batches != 0:
                raise ValueError(f'Batch dimension ({batch_size}) must be divisible by num_batches={num_batches}')
            memory = torch.empty(
                (num_batches, *batch.shape), dtype=batch.dtype, device=batch.device, pin_memory=pin_memory
            )

        batches.append(batch)

        if len(batches) == num_batches:
            for i in range(num_batches):
                for j in range(num_batches):
                    memory[i, j * slice_size : (j + 1) * slice_size] = batches[j][i * slice_size : (i + 1) * slice_size]
            batches = []
            for i in range(num_batches):
                yield memory[i]
