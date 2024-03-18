from typing import Iterable

import numpy as np
import torch.distributed as dist
import xarray as xr
from torch.utils.data import IterableDataset


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
    drop_remainder: bool = True,
    seed: int = 0,
    rank: int | None = None,
    world_size: int | None = None,
    process_group: dist.ProcessGroup | None = None,
    load: bool = True,
) -> Iterable[xr.Dataset | xr.DataArray]:
    num_total_elements = len(ds[dim])
    num_chunks = num_total_elements // chunk_size

    if rank is None:
        rank = dist.get_rank(process_group)
    if world_size is None:
        world_size = dist.get_world_size(process_group)

    chunk_indices = shard_indices(
        num_chunks, rank, world_size, shuffle=shuffle, drop_remainder=drop_remainder, seed=seed
    )

    for chunk_idx in chunk_indices:
        start = chunk_idx * chunk_size
        end = start + chunk_size
        chunk = ds.isel({dim: slice(start, end)})
        if load:
            chunk.load()
        yield chunk


class ShardedXrDataset(IterableDataset):
    def __init__(
        self,
        ds: xr.Dataset | xr.DataArray,
        chunk_size: int,
        dim: str,
        shuffle: bool = False,
        drop_remainder: bool = True,
        seed: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
        process_group: dist.ProcessGroup | None = None,
        load: bool = True,
    ):
        self.ds = ds
        self.chunk_size = chunk_size
        self.dim = dim
        self.shuffle = shuffle
        self.drop_remainder = drop_remainder
        self.seed = seed
        self.load = load

        if rank is None:
            self.rank = dist.get_rank(process_group)
        else:
            self.rank = rank

        if world_size is None:
            self.world_size = dist.get_world_size(process_group)
        else:
            self.world_size = world_size

    def __iter__(self):
        return sharded_xr_dataset(
            self.ds,
            self.chunk_size,
            self.dim,
            self.shuffle,
            self.drop_remainder,
            self.seed,
            self.rank,
            self.world_size,
            self.load,
        )
