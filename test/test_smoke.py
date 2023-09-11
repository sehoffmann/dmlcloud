import sys

import pytest


class TestSmoke:
    def test_import(self):
        import dml  # noqa: F401

    def test_version(self):
        import dml

        version = dml.__version__
        assert isinstance(version, str)
        assert version > '0.0.0'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
