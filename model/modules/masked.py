"""
Module that deals with sequences that need masking
"""
from enum import Enum
import torch as t
import torch.nn as nn
from typing import NewType, ClassVar

MaskMode = Enum('MaskMode', 'subtract multiply')
MaskTime = Enum('MaskTime', 'pre post')

MaskTensor = NewType('MaskTensor', t.LongTensor)

def mask_sequence(input_batch: t.Tensor, mask_index: float=0) -> MaskTensor:
    """
    Returns a MaskTensor where masked indices are 0 and rest 1
    :param input_batch: Batch first tensor of padded sequences
    :param mask_index: Value that signifies an index that should be masked
    :returns: A MaskTensor, same shape as input_batch
    """
    return MaskTensor((input_batch != mask_index).long())


class MaskedOp(nn.Module):
    """
    Base class for modules that require masking before op
    """
    mask_value: float
    mask_mode: MaskMode
    mask_time: MaskTime
    op: nn.Module

    def __init__(self,
                 op: nn.Module,
                 mask_mode: MaskMode,
                 mask_time: MaskTime,
                 mask_value: float=0) -> None:
        super().__init__()
        self.op = op
        self.mask_mode = mask_mode
        self.mask_time = mask_time
        self.mask_value = mask_value

    def _apply_mask(self, inpt: t.Tensor, mask: MaskTensor) -> Tensor:
        """
        Applies the mask to inpt according to MaskMode
        :param inpt: Input tensor to apply mask to
        :param mask: Mask to apply
        :returns: Result of applying mask to inpt given the Mode
        """
        if self.mask_mode == MaskMode.subtract:
            return inpt + (mask - 1) * self.mask_value
        elif self.mask_mode == MaskMode.multiply:
            return inpt * mask
        else:
            raise Exception('Malformed mask_mode for MaskedOp: %s' % self.mask_mode)

    def forward(self, input_batch: t.Tensor, mask: MaskTensor) -> t.Tensor:
        """
        Subtracts mask from its input and applies op to it
        :param input_batch: Batch of variable length sequences to mask and apply op on
        :param mask: Mask tensor that is 1 for all in-sequence values and 0 for others
        :returns: Output of the supplied op after masking
        """
        if self.mask_time == MaskTime.pre:
            return self.op(self._apply_mask(input_batch, mask))
        elif self.mask_time == MaskTime.post:
            return self._apply_mask(self.op(input_batch), mask)
        else:
            raise Exception('Malformed mask_time for MaskedOp: %s' % self.mask_time)


class MaskedSoftmax(MaskedOp):
    """
    Module that runs softmax on a batch of variable-length sequences
    """

    mask_value: ClassVar[float]=1e10

    def __init__(self, dim: int=1) -> None:
        super().__init__(nn.Softmax(dim=dim),
                         MaskMode.subtract,
                         MaskTime.pre,
                         mask_value=MaskedSoftmax.mask_value)


class MaskedLogSoftmax(MaskedOp):
    """
    Module that runs LogSoftmax on a batch of variable-length sequences
    Wrapper around MaskedOp.
    """

    mask_value: ClassVar[float]=1e10

    def __init__(self, dim: int=1) -> None:
        super().__init__(nn.LogSoftmax(dim=dim),
                         MaskMode.subtract,
                         MaskTime.pre,
                         mask_value=MaskedLogSoftmax.mask_value)


class MaskedLinear(MaskedOp):
    """
    Module that applies a Linear layer to its input then masks the output
    """
    def __init__(self, in_features: int, out_features: int, bias: bool=True) -> None:
        super().__init__(nn.Linear(in_features, out_features, bias),
                         MaskMode.multiply,
                         MaskTime.post)
