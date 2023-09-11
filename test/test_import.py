import sys

import pytest


class TestImport:
    def test_import(self):
        import dmlcloud  # noqa: F401

    def test_version(self):
        import dmlcloud

        version = dmlcloud.__version__
        assert isinstance(version, str)
        assert version > '0.0.0'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
