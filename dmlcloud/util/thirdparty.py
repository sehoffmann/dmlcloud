import importlib
import sys
from types import ModuleType
from typing import Optional


ML_MODULES = [
    'torch',
    'torchvision',
    'torchtext',
    'torchaudio',
    'einops',
    'numpy',
    'pandas',
    'xarray',
    'sklearn',
]


def is_imported(name: str) -> bool:
    return name in sys.modules


def try_import(name: str) -> Optional[ModuleType]:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def try_get_version(name: str) -> Optional[str]:
    module = try_import(name)
    if module is not None:
        return str(module.__version__)
    else:
        return None
