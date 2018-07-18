"""
Module that holds classes that can be used for answer prediction
"""
from typing import NamedTuple
import torch as t
import torch.nn as nn
from torch.nn.utils.rnn import (PackedSequence,
                                pack_padded_sequence,
                                pad_packed_sequence)

from model.batcher import QABatch
from model.util import get_last_hidden_states
from model.modules.attention import BidirectionalAttention
from model.modules.masked import MaskedLinear
from model.modules.embeddor import Embeddor


ModelPredictions = NamedTuple('ModelPredictions', [
    ('start_logits', t.Tensor),
    ('end_logits', t.Tensor),
    ('no_ans_logits', t.Tensor)
])


class PredictorModel(nn.Module):
    """
    Base class for any Predictor Model
    """
    def __init__(self):
        super().__init__()

    def forward(self, batch: QABatch) -> ModelPredictions:
        """
        Predicts (span_start_logits, span_end_logits, has_ans_logits) for a batch of samples
        :param batch: QABatch: a batch of samples returned from a batcher
        :returns: A ModelPredictions object containing start_logits, end_logits, no_ans_prob
        """
        raise NotImplementedError


GRUConfig = NamedTuple('GRUConfig', [
    ('hidden_size', int),
    ('num_layers', int),
    ('dropout', float),
    ('bidirectional', bool)
])


class PredictorConfig():
    """
    Object that holds config values for a Predictor model
    """

    gru: GRUConfig
    batch_size: int
    n_directions: int

    def __init__(self,
                 gru: GRUConfig,
                 attention_hidden_size: int,
                 batch_size: int) -> None:
        self.gru = gru
        self.n_directions = 1 + int(self.gru.bidirectional)
        self.total_hidden_size = self.n_directions * self.gru.hidden_size
        self.batch_size = batch_size


class ContextualEncoder(nn.Module):
    """
    Module that encodes an embedded sequence using an LSTM
    """

    config: GRUConfig
    GRU: nn.GRU

    def __init__(self, input_dim: int, config: GRUConfig) -> None:
        super().__init__()
        self.config = config
        self.gru = nn.GRU(input_dim,
                          self.config.hidden_size,
                          self.config.num_layers,
                          dropout=self.config.dropout,
                          batch_first=True,
                          bidirectional=self.config.bidirectional)

    def forward(self,
                embedded_in: t.Tensor,
                lengths: t.LongTensor,
                length_idxs: t.LongTensor,
                orig_idxs: t.LongTensor) -> t.Tensor:
        """
        Takes in a given padded sequence alongside its length and sorting info and
        encodes it through a bidirectional GRU
        :param embedded_in: Padded and embedded sequence
        :param lengths: Descending-sorted list of lengths of the sequences in the batch
        :param length_idxs: Indices to sort the input to get the sequences in descending length order
        :param orig_idxs: Indices to sort the length-sorted sequences back to the original order
        :returns: Padded, encoded sequences in original order (batch_len, sequence_len, encoding_size)
        """
        len_sorted = embedded_in[length_idxs]
        packed: PackedSequence = pack_padded_sequence(len_sorted,
                                                      lengths,
                                                      batch_first=True)
        processed_packed, _ = self.gru(packed)
        processed_len_sorted, _ = pad_packed_sequence(processed_packed, batch_first=True)
        return processed_len_sorted[orig_idxs]


class BidafOutput(nn.Module):
    """
    Module that produces prediction logits given a context encoding
    as described in the BiDAF paper
    """
    config: PredictorConfig
    start_modeling_encoder: ContextualEncoder
    end_modeling_encoder: ContextualEncoder
    start_predictor: nn.Linear
    end_predictor: nn.Linear
    no_answer_gru: nn.GRU
    no_answer_predictor: nn.Linear

    def __init__(self, config: PredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.start_modeling_encoder = ContextualEncoder(self.config.total_hidden_size * 4, self.config.gru)
        self.end_modeling_encoder = ContextualEncoder(self.config.total_hidden_size * 4, self.config.gru)
        self.start_predictor = MaskedLinear(self.config.total_hidden_size * 5, 1)
        self.end_predictor = MaskedLinear(self.config.total_hidden_size * 5, 1)
        self.no_answer_gru = nn.GRU(self.config.total_hidden_size * 4,
                                    self.config.gru.hidden_size,
                                    1,
                                    dropout=self.config.gru.dropout,
                                    batch_first=True,
                                    bidirectional=self.config.gru.bidirectional)
        self.no_answer_predictor = nn.Linear(self.config.total_hidden_size, 1)

    def forward(self,
                context_encoding: t.Tensor,
                context_mask: t.LongTensor,
                lengths: t.LongTensor,
                length_idxs: t.LongTensor,
                orig_idxs: t.LongTensor) -> ModelPredictions:
        start_modeled_ctx = self.start_modeling_encoder(context_encoding,
                                                        lengths,
                                                        length_idxs,
                                                        orig_idxs)

        end_modeled_ctx = self.end_modeling_encoder(context_encoding,
                                                    lengths,
                                                    length_idxs,
                                                    orig_idxs)

        start_predictions = self.start_predictor(t.cat([context_encoding, start_modeled_ctx], dim=2), context_mask).squeeze(2)
        end_predictions = self.end_predictor(t.cat([context_encoding, end_modeled_ctx]), context_mask).squeeze(2)

        context_length_sorted = context_encoding[length_idxs]
        context_packed: PackedSequence = pack_padded_sequence(context_length_sorted,
                                                              lengths,
                                                              batch_first=True)

        _, no_answer_out_len_sorted = self.no_answer_gru(context_packed)
        no_answer_out_len_sorted = get_last_hidden_states(no_answer_out_len_sorted, self.config)
        no_answer_out = no_answer_out_len_sorted[orig_idxs]
        no_answer_predictions = self.no_answer_predictor(no_answer_out)

        return ModelPredictions(start_logits=start_predictions,
                                end_logits=end_predictions,
                                no_ans_logits=no_answer_predictions)


class BidafPredictor(PredictorModel):
    """
    Predictor as described in the BiDAF paper
    """

    config: PredictorConfig
    embed: Embeddor
    q_encoder: ContextualEncoder
    ctx_encoder: ContextualEncoder
    attention: BidirectionalAttention
    output_layer: BidafOutput

    def __init__(self, embeddor: Embeddor, config: PredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = embeddor
        self.q_encoder = ContextualEncoder(self.embed.embedding_dim, self.config.gru)
        self.ctx_encoder = ContextualEncoder(self.embed.embedding_dim, self.config.gru)
        self.attention = BidirectionalAttention(self.config.total_hidden_size)
        self.output_layer = BidafOutput(self.config)

    def forward(self, batch: QABatch) -> ModelPredictions:
        """
        Check base class method for docs
        """

        q_embedded = self.embed(batch.question_words, batch.question_chars)
        q_processed = self.q_encoder(q_embedded,
                                     batch.question_lens,
                                     batch.question_len_idxs,
                                     batch.question_orig_idxs)

        ctx_embedded = self.embed(batch.context_words, batch.context_chars)
        ctx_processed = self.ctx_encoder(ctx_embedded,
                                         batch.context_lens,
                                         batch.context_len_idxs,
                                         batch.context_orig_idxs)

        c2q_att, q2c_att = self.attention(ctx_processed, q_processed)

        q_aware_ctx = t.cat([ctx_processed, c2q_att, ctx_processed * c2q_att, ctx_processed * q2c_att], dim=2)

        return self.output_layer(q_aware_ctx,
                                 batch.context_mask,
                                 batch.context_lens,
                                 batch.context_len_idxs,
                                 batch.context_orig_idxs)
