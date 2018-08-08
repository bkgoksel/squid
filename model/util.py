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
        print(
            '[WARNING]: CUDA requested but is unavailable, defaulting to CPU')
        return t.device('cpu')
    return t.device('cpu')


def mem_report(print_all: bool=False):
    """
    Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported
    if print_all is True, print size and shape info for each tensor
    """
    import gc

    def _mem_report(tensors, mem_type):
        '''Print the selected tensors of type

        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)

        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation '''
        print('Storage on %s' % (mem_type))
        print('-' * LEN)
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel * element_size / 1024 / 1024  # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            if print_all:
                print('%s\t\t%s\t\t%.2f' % (
                    element_type,
                    size,
                    mem))
        print('-' * LEN)
        print('Total Tensors: %d \tUsed Memory Space: %.2f MBytes' % (total_numel, total_mem))
        print('-' * LEN)

    LEN = 65
    print('=' * LEN)
    print('%s\t%s\t\t\t%s' % ('Element type', 'Size', 'Used MEM(MBytes)'))
    tensors = []
    for obj in gc.get_objects():
        try:
            if t.is_tensor(obj) or (hasattr(obj, 'data') and t.is_tensor(obj.data)):
                tensors.append(obj)
        except Exception:
            pass
    cuda_tensors = [tensor for tensor in tensors if tensor.is_cuda]
    host_tensors = [tensor for tensor in tensors if not tensor.is_cuda]
    _mem_report(cuda_tensors, 'GPU')
    _mem_report(host_tensors, 'CPU')
    print('=' * LEN)
