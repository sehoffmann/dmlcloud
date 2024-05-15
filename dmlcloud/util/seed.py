import random

import numpy as np
import torch


def seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def enable_determinism():
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
