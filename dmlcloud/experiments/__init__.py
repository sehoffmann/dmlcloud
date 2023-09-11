from . import cv_classification, mnist

ALL_EXPERIMENTS = [cv_classification, mnist]

__all__ = [
    'ALL_EXPERIMENTS' 'cv_classification',
    'mnist',
]
assert __all__ == sorted(__all__)
