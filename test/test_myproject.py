import sys

import pytest


class TestBasic:
    def test_foobar(self):
        assert True


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
