from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
import xarray as xr
from torch.utils.data import get_worker_info, IterableDataset


def shard_indices(
    num_elements: int,
    rank: int,
    world_size: int,
    shuffle: bool = False,
    even_shards: bool = True,
    seed: int = 0,
) -> list[int]:
    """
    even_shards: If True, every worker receives the same number of shards, and the last shards are dropped.
    """
    indices = np.arange(num_elements)

    if shuffle:
        np.random.Generator(np.random.MT19937(seed)).shuffle(indices)

    if even_shards:
        indices = indices[: num_elements - num_elements % world_size]

    return indices[rank::world_size].tolist()  # this also converts np.int64 to python's int


def chunk_and_shard_indices(
    num_elements: int,
    chunk_size: int,
    rank: int,
    world_size: int,
    chunk_overlap: int = 0,
    even_shards: bool = True,
    equal_chunks: bool = True,
    shuffle: bool = False,
    seed: int = 0,
):
    if equal_chunks:
        num_chunks = num_elements // chunk_size
    else:
        num_chunks = (num_elements + chunk_size - 1) // chunk_size

    chunk_indices = shard_indices(num_chunks, rank, world_size, shuffle=shuffle, even_shards=even_shards, seed=seed)
    chunks = []
    for chunk_idx in chunk_indices:
        start = chunk_idx * chunk_size
        end = start + chunk_size + chunk_overlap
        chunks.append((start, end))
    return chunks


def sharded_xr_dataset(
    ds: xr.Dataset | xr.DataArray,
    dim: str,
    chunk_size: int,
    chunk_overlap: int = 0,
    even_shards: bool = True,
    equal_chunks: bool = True,
    shuffle: bool = False,
    seed: int = 0,
    rank: int | None = None,
    world_size: int | None = None,
    process_group: dist.ProcessGroup | None = None,
    load: bool = False,
    load_kwargs: dict | None = None,
) -> Iterable[xr.Dataset | xr.DataArray]:
    if rank is None:
        rank = dist.get_rank(process_group)
    if world_size is None:
        world_size = dist.get_world_size(process_group)

    num_elements = len(ds[dim])
    chunks = chunk_and_shard_indices(
        num_elements,
        chunk_size,
        rank,
        world_size,
        chunk_overlap=chunk_overlap,
        even_shards=even_shards,
        equal_chunks=equal_chunks,
        shuffle=shuffle,
        seed=seed,
    )
    for start, end in chunks:
        chunk = ds.isel({dim: slice(start, end)})
        if load:
            kwargs = load_kwargs or {}
            chunk.load(**kwargs)
        yield chunk


class ShardedXrDataset(IterableDataset):
    def __init__(
        self,
        ds: xr.Dataset | xr.DataArray,
        dim: str,
        chunk_size: int,
        chunk_overlap: int = 0,
        even_shards: bool = True,
        equal_chunks: bool = True,
        shuffle: bool = False,
        seed: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
        process_group: dist.ProcessGroup | None = None,
        load: bool = False,
        load_kwargs: dict | None = None,
    ):
        self.ds = ds
        self.dim = dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.even_shards = even_shards
        self.equal_chunks = equal_chunks
        self.shuffle = shuffle
        self.seed = seed
        self.load = load
        self.load_kwargs = load_kwargs

        self.rank = rank if rank is not None else dist.get_rank(process_group)
        self.world_size = world_size if world_size is not None else dist.get_world_size(process_group)
        self._num_iters = 0

    def set_epoch(self, epoch: int):
        self._num_iters = epoch

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
            self.dim,
            self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            even_shards=self.even_shards,
            equal_chunks=self.equal_chunks,
            shuffle=self.shuffle,
            seed=self.seed + self._num_iters,
            rank=rank,
            world_size=world_size,
            load=self.load,
            load_kwargs=self.load_kwargs,
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


def interleave_dict_batches(
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
    memory = {}
    slice_size = {}
    for batch in iterable:
        if not memory:
            for k, tensor in batch.items():
                batch_size = tensor.shape[0]
                if batch_size % num_batches != 0:
                    raise ValueError(f'Batch dimension ({batch_size}) must be divisible by num_batches={num_batches}')
                slice_size[k] = batch_size // num_batches
                memory[k] = torch.empty(
                    (num_batches, *tensor.shape), dtype=tensor.dtype, device=tensor.device, pin_memory=pin_memory
                )

        batches.append(batch)

        if len(batches) == num_batches:
            for k in memory:
                for i in range(num_batches):
                    for j in range(num_batches):
                        source = batches[j][k][i * slice_size[k] : (i + 1) * slice_size[k]]
                        memory[k][i, j * slice_size[k] : (j + 1) * slice_size[k]] = source
            batches = []
            for i in range(num_batches):
                yield {k: memory[k][i] for k in memory.keys()}
