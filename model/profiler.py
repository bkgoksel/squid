"""
Includes utilities for profiling memory and time usage of model training
"""

import gc
from functools import wraps
from typing import List, Callable, Any, Iterable
import torch as t


def mem_report(print_all: bool = False) -> None:
    """
    Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported
    if print_all is True, print size and shape info for each tensor
    """

    def _mem_report(tensors: Iterable, mem_type: str) -> None:
        """Print the selected tensors of type

        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)

        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation """
        print("Storage on %s" % (mem_type))
        print("-" * LEN)
        total_numel = 0
        total_mem = 0
        visited_data: List[Any] = []
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
                print("%s\t\t%s\t\t%.2f" % (element_type, size, mem))
        print("-" * LEN)
        print(
            "Total Tensors: %d \tUsed Memory Space: %.2f MBytes"
            % (total_numel, total_mem)
        )
        print("-" * LEN)

    LEN = 65
    if print_all:
        print("=" * LEN)
    print("%s\t%s\t\t\t%s" % ("Element type", "Size", "Used MEM(MBytes)"))
    tensors = []
    for obj in gc.get_objects():
        try:
            if t.is_tensor(obj) or (hasattr(obj, "data") and t.is_tensor(obj.data)):
                tensors.append(obj)
        except Exception:
            pass
    cuda_tensors = [tensor for tensor in tensors if tensor.is_cuda]
    host_tensors = [tensor for tensor in tensors if not tensor.is_cuda]
    _mem_report(cuda_tensors, "GPU")
    _mem_report(host_tensors, "CPU")
    if print_all:
        print("=" * LEN)


def memory_profiled(op: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(op)
    def profiled_function(*args: Any, **kwargs: Any) -> Any:
        print("Before op, allocated tensors")
        mem_report()
        res = op(*args, **kwargs)
        print("After op, allocated tensors")
        mem_report()
        return res

    return profiled_function


def autograd_profiled(op: Callable[..., Any], use_cuda: bool) -> Callable[..., Any]:
    @wraps(op)
    def profiled_function(*args: Any, **kwargs: Any) -> Any:
        with t.autograd.profiler.profile(use_cuda=use_cuda) as prof:
            res = op(*args, **kwargs)
            print("Debug run complete, printing CPU profile")
            prof.table(sort_by="cpu_time_total")
            print("Debug run complete, printing CUDA profile")
            prof.table(sort_by="cuda_time_total")
        return res

    return profiled_function
