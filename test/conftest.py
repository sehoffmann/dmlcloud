import pytest

from dmlcloud.util.distributed import init_process_group_dummy, deinitialize_torch_distributed


@pytest.fixture
def torch_distributed():
    init_process_group_dummy()
    yield
    deinitialize_torch_distributed()