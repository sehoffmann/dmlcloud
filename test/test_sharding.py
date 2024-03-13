import sys

import numpy as np
import pytest
from dmlcloud.util.distributed import shard_indices
from numpy.testing import assert_array_equal


class TestSharding:
    def test_even(self):
        assert_array_equal(shard_indices(10, 0, 2, shuffle=False, drop_remainder=False), [0, 2, 4, 6, 8])
        assert_array_equal(shard_indices(10, 1, 2, shuffle=False, drop_remainder=False), [1, 3, 5, 7, 9])

    def test_uneven(self):
        assert_array_equal(shard_indices(10, 0, 3, shuffle=False, drop_remainder=False), [0, 3, 6, 9])
        assert_array_equal(shard_indices(10, 1, 3, shuffle=False, drop_remainder=False), [1, 4, 7])
        assert_array_equal(shard_indices(10, 2, 3, shuffle=False, drop_remainder=False), [2, 5, 8])

        assert_array_equal(shard_indices(11, 0, 2, shuffle=False, drop_remainder=False), [0, 2, 4, 6, 8, 10])
        assert_array_equal(shard_indices(11, 1, 2, shuffle=False, drop_remainder=False), [1, 3, 5, 7, 9])

    def test_dropping(self):
        assert_array_equal(shard_indices(10, 0, 3, shuffle=False, drop_remainder=True), [0, 3, 6])
        assert_array_equal(shard_indices(10, 1, 3, shuffle=False, drop_remainder=True), [1, 4, 7])
        assert_array_equal(shard_indices(10, 2, 3, shuffle=False, drop_remainder=True), [2, 5, 8])

        assert_array_equal(shard_indices(11, 0, 2, shuffle=False, drop_remainder=True), [0, 2, 4, 6, 8])
        assert_array_equal(shard_indices(11, 1, 2, shuffle=False, drop_remainder=True), [1, 3, 5, 7, 9])

    def test_shuffling(self):
        indices = shard_indices(10, 0, 2, shuffle=True, drop_remainder=False, seed=0)
        assert len(indices) == 5
        assert len(np.unique(indices)) == 5
        assert list(indices) != list(sorted(indices))
        assert (indices >= 0).all() and (indices <= 9).all()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
