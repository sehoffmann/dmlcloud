import sys

import xarray as xr
import numpy as np
import pytest
from dmlcloud.util.data import shard_indices, chunked_xr_dataset
from numpy.testing import assert_array_equal


class TestSharding:

    def test_types(self):
        indices = shard_indices(10, 0, 2, shuffle=False, drop_remainder=False)
        assert isinstance(indices, list)
        assert all(isinstance(i, int) for i in indices)

    def test_even(self):
        assert shard_indices(10, 0, 2, shuffle=False, drop_remainder=False) == [0, 2, 4, 6, 8]
        assert shard_indices(10, 1, 2, shuffle=False, drop_remainder=False) == [1, 3, 5, 7, 9]

    def test_uneven(self):
        assert shard_indices(10, 0, 3, shuffle=False, drop_remainder=False) == [0, 3, 6, 9]
        assert shard_indices(10, 1, 3, shuffle=False, drop_remainder=False) == [1, 4, 7]
        assert shard_indices(10, 2, 3, shuffle=False, drop_remainder=False) == [2, 5, 8]

        assert shard_indices(11, 0, 2, shuffle=False, drop_remainder=False) == [0, 2, 4, 6, 8, 10]
        assert shard_indices(11, 1, 2, shuffle=False, drop_remainder=False) == [1, 3, 5, 7, 9]

    def test_dropping(self):
        assert shard_indices(10, 0, 3, shuffle=False, drop_remainder=True) == [0, 3, 6]
        assert shard_indices(10, 1, 3, shuffle=False, drop_remainder=True) == [1, 4, 7]
        assert shard_indices(10, 2, 3, shuffle=False, drop_remainder=True) == [2, 5, 8]

        assert shard_indices(11, 0, 2, shuffle=False, drop_remainder=True) == [0, 2, 4, 6, 8]
        assert shard_indices(11, 1, 2, shuffle=False, drop_remainder=True) == [1, 3, 5, 7, 9]

    def test_shuffling(self):
        indices = shard_indices(10, 0, 2, shuffle=True, drop_remainder=False, seed=0)
        assert len(indices) == 5
        assert len(np.unique(indices)) == 5
        assert indices != list(sorted(indices))
        assert (np.array(indices) >= 0).all() and (np.array(indices) <= 9).all()


class TestChunking:

    def test_basic(self):
        ds = xr.DataArray(np.arange(100), dims=['x'], name='var').to_dataset()
        world_size = 3
        chunk_size = 15

        chunks_1 = list(chunked_xr_dataset(ds, chunk_size, 'x', world_size=world_size, rank=0, shuffle=False))
        chunks_2 = list(chunked_xr_dataset(ds, chunk_size, 'x', world_size=world_size, rank=1, shuffle=False))
        chunks_3 = list(chunked_xr_dataset(ds, chunk_size, 'x', world_size=world_size, rank=2, shuffle=False))

        assert len(chunks_1) == 2
        assert len(chunks_2) == 2
        assert len(chunks_3) == 2

        assert isinstance(chunks_1[0], xr.Dataset)

        assert chunks_1[0].x.size == 15
        assert chunks_1[1].x.size == 15
        assert chunks_2[0].x.size == 15
        assert chunks_2[1].x.size == 15
        assert chunks_3[0].x.size == 15
        assert chunks_3[1].x.size == 15

        assert_array_equal(chunks_1[0]['var'], np.arange(0, 15))
        assert_array_equal(chunks_2[0]['var'], np.arange(15, 30))
        assert_array_equal(chunks_3[0]['var'], np.arange(30, 45))
        assert_array_equal(chunks_1[1]['var'], np.arange(45, 60))
        assert_array_equal(chunks_2[1]['var'], np.arange(60, 75))
        assert_array_equal(chunks_3[1]['var'], np.arange(75, 90))


    def test_shuffled(self):
        ds = xr.DataArray(np.arange(100), dims=['x'], name='var').to_dataset()
        world_size = 3
        chunk_size = 15

        chunks_1 = list(chunked_xr_dataset(ds, chunk_size, 'x', world_size=world_size, rank=0, shuffle=True, seed=0))
        chunks_2 = list(chunked_xr_dataset(ds, chunk_size, 'x', world_size=world_size, rank=1, shuffle=True, seed=0))
        chunks_3 = list(chunked_xr_dataset(ds, chunk_size, 'x', world_size=world_size, rank=2, shuffle=True, seed=0))

        assert len(chunks_1) == 2
        assert len(chunks_2) == 2
        assert len(chunks_3) == 2

        catted = xr.concat(chunks_1 + chunks_2 + chunks_3, dim='x')['var'].values
        assert catted.tolist() != list(range(90))
        assert list(sorted(catted.tolist())) == list(range(90))

        chunk = chunks_1[0]['var'].values
        assert chunk.tolist() == list(range(chunk[0], chunk[-1] + 1))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
