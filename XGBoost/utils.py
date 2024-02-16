import random
import numpy as np
import torch


def set_random_seed(seed):
    """
    Set the seed for random number generation in Python, NumPy, and PyTorch.

    Args:
    seed (int): The seed value to use for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False