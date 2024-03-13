import importlib
from types import ModuleType
from typing import Optional


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
