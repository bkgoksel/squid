"""
Module that holds classes that can be used for answer prediction
"""
from typing import Set, NamedTuple, Optional, cast
import torch as t
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from model.batcher import QABatch
from model.util import get_last_hidden_states
from model.modules.attention import (
    DocQABidirectionalAttention,
    BidafBidirectionalAttention,
    SelfAttention,
)
from model.modules.masked import MaskedLinear, MaskedLogSoftmax
from model.modules.embeddor import Embeddor

ModelPredictions = NamedTuple(
    "ModelPredictions",
    [("start_logits", t.Tensor), ("end_logits", t.Tensor), ("no_ans_logits", t.Tensor)],
)


class PredictorModel(nn.Module):
    """
    Base class for any Predictor Model
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: QABatch) -> ModelPredictions:
        """
        Predicts (span_start_logits, span_end_logits, has_ans_logits) for a batch of samples
        :param batch: QABatch: a batch of samples returned from a batcher
        :returns: A ModelPredictions object containing start_logits, end_logits, no_ans_prob
        """
        raise NotImplementedError


class ContextualEncoderConfig:
    hidden_size: int
    num_layers: int
    dropout_input: bool
    dropout_prob: float

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout_input: bool,
        dropout_prob: float,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_input = dropout_input
        self.dropout_prob = dropout_prob


class ContextualEncoder(nn.Module):
    """
    Module that encodes an embedded sequence using a GRU
    """

    config: ContextualEncoderConfig
    output_size: int
    dropout: Optional[nn.Dropout]
    GRU: nn.GRU

    def __init__(self, input_dim: int, config: ContextualEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.output_size = 2 * self.config.hidden_size
        if self.config.dropout_prob:
            self.dropout = nn.Dropout(self.config.dropout_prob)
        else:
            self.dropout = None
        self.gru = nn.GRU(
            input_dim,
            self.config.hidden_size,
            self.config.num_layers,
            dropout=self.config.dropout_prob if self.config.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )

    def forward(
        self,
        inpt: t.Tensor,
        lengths: t.LongTensor,
        length_idxs: t.LongTensor,
        orig_idxs: t.LongTensor,
    ) -> t.Tensor:
        """
        Takes in a given padded sequence alongside its length and sorting info and
        encodes it through a bidirectional GRU
        :param inpt: Padded and embedded sequence
        :param lengths: Descending-sorted list of lengths of the sequences in the batch
        :param length_idxs: Indices to sort the input to get the sequences in descending length order
        :param orig_idxs: Indices to sort the length-sorted sequences back to the original order
        :returns: Padded, encoded sequences in original order (batch_len, sequence_len, output_size)
        """
        if self.dropout:
            inpt = self.dropout(inpt)
        inpt = inpt[length_idxs]
        inpt = pack_padded_sequence(inpt, lengths, batch_first=True)
        out, _ = self.gru(inpt)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return out[orig_idxs]
