"""
Module for general utils for loading components from disk and dataset/evaluation handling
"""

import torch as t


def get_device(use_cuda: bool):
    if use_cuda:
        if t.cuda.is_available:
            return t.device('cuda')
        print('[WARNING]: CUDA requested but is unavailable, defaulting to CPU')
        return t.device('cpu')
    return t.device('cpu')
