"""
Module for general utils for loading components from disk and dataset/evaluation handling
"""

import torch as t


def get_last_hidden_states(states, n_directions: int, total_hidden_size: int):
    """
    Do some juggling with the output of the RNNs to get the
        final hidden states of the topmost layers of all the
        directions to feed into attention

    To get it in the same config as q_processed:
        1. Make states batch first
        2. Only keep the last layers for each direction
        3. Concatenate the layer hidden states in one dimension

    :param states: Last hidden states for all layers and directions, of shape:
        [n_layers*n_dirs, batch_size, hidden_size]:
            The first dimension is laid out like:
                layer0dir0, layer0dir1, layer1dir0, layer1dir1
    :param n_directions: Number of directions of the RNN that output these states
    :param total_hidden_size: Total hidden size for one timestep
        (i.e. (n_directions * hidden_size)

    :returns: All hidden states, of shape:
        [batch_size, max_seq_len, hidden_size*n_dirs]

    """
    batch_size = states.size(1)
    states = states.transpose(0, 1)
    states = states[:, -n_directions:, :]
    out = states.contiguous().view(batch_size, total_hidden_size)
    return out


def get_device(use_cuda: bool):
    if use_cuda:
        if t.cuda.is_available:
            return t.device('cuda')
        print('[WARNING]: CUDA requested but is unavailable, defaulting to CPU')
        return t.device('cpu')
    return t.device('cpu')
