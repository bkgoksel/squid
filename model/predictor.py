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


class GRUConfig:
    single_hidden_size: int
    num_layers: int
    bidirectional: bool
    dropout_prob: float
    total_hidden_size: int
    n_directions: int

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout_prob: float,
        force_unidirectional: bool,
    ) -> None:
        self.single_hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.bidirectional = not force_unidirectional
        self.n_directions = 1 + int(self.bidirectional)
        self.total_hidden_size = self.n_directions * self.single_hidden_size


class DocQAConfig:
    """
    Object that holds config values for a Predictor model
    """

    gru: GRUConfig
    dropout_prob: float
    attention_linear_hidden_size: int
    use_self_attention: bool
    batch_size: int

    def __init__(
        self,
        gru: GRUConfig,
        dropout_prob: float,
        attention_linear_hidden_size: int,
        use_self_attention: bool,
        batch_size: int,
    ) -> None:
        self.gru = gru
        self.dropout_prob = dropout_prob
        self.attention_linear_hidden_size = attention_linear_hidden_size
        self.batch_size = batch_size
        self.use_self_attention = use_self_attention


class ContextualEncoder(nn.Module):
    """
    Module that encodes an embedded sequence using a GRU
    """

    config: GRUConfig
    output_size: int
    dropout: nn.Dropout
    GRU: nn.GRU

    def __init__(self, input_dim: int, config: GRUConfig) -> None:
        super().__init__()
        self.config = config
        self.output_size = self.config.total_hidden_size
        self.dropout = nn.Dropout(self.config.dropout_prob)
        self.gru = nn.GRU(
            input_dim,
            self.config.single_hidden_size,
            self.config.num_layers,
            dropout=self.config.dropout_prob if self.config.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=self.config.bidirectional,
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
        inpt = self.dropout(inpt)
        inpt = inpt[length_idxs]
        inpt = pack_padded_sequence(inpt, lengths, batch_first=True)
        out, _ = self.gru(inpt)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return out[orig_idxs]


class DocQAOutput(nn.Module):
    """
    Module that produces prediction logits given a context encoding
    as described in the DocumentQA paper
    """

    config: GRUConfig
    start_modeling_encoder: ContextualEncoder
    end_modeling_encoder: ContextualEncoder
    start_predictor: nn.Linear
    end_predictor: nn.Linear
    no_answer_gru: nn.GRU
    no_answer_predictor: nn.Linear

    def __init__(self, config: GRUConfig, input_size: int) -> None:
        super().__init__()
        self.config = config
        self.start_modeling_encoder = ContextualEncoder(input_size, self.config)
        self.end_modeling_encoder = ContextualEncoder(
            input_size + self.start_modeling_encoder.output_size, self.config
        )
        self.start_predictor = MaskedLinear(self.start_modeling_encoder.output_size, 1)
        self.end_predictor = MaskedLinear(self.end_modeling_encoder.output_size, 1)
        self.no_answer_gru = nn.GRU(
            input_size,
            self.config.single_hidden_size,
            1,
            batch_first=True,
            bidirectional=self.config.bidirectional,
        )
        self.no_answer_predictor = nn.Linear(self.config.total_hidden_size, 1)
        self.softmax = MaskedLogSoftmax(dim=-1)

    def forward(
        self,
        context_encoding: t.Tensor,
        context_mask: t.LongTensor,
        lengths: t.LongTensor,
        length_idxs: t.LongTensor,
        orig_idxs: t.LongTensor,
    ) -> ModelPredictions:
        start_modeled_ctx = self.start_modeling_encoder(
            context_encoding, lengths, length_idxs, orig_idxs
        )

        end_modeled_ctx = self.end_modeling_encoder(
            t.cat([context_encoding, start_modeled_ctx], dim=2),
            lengths,
            length_idxs,
            orig_idxs,
        )

        start_predictions = self.start_predictor(
            start_modeled_ctx, mask=context_mask
        ).squeeze(2)
        start_predictions = self.softmax(start_predictions, mask=context_mask)
        end_predictions = self.end_predictor(
            end_modeled_ctx, mask=context_mask
        ).squeeze(2)
        end_predictions = self.softmax(end_predictions, mask=context_mask)

        context_length_sorted = context_encoding[length_idxs]
        context_packed: PackedSequence = pack_padded_sequence(
            context_length_sorted, lengths, batch_first=True
        )

        _, no_answer_out_len_sorted = self.no_answer_gru(context_packed)
        no_answer_out_len_sorted = get_last_hidden_states(
            no_answer_out_len_sorted,
            self.config.n_directions,
            self.config.total_hidden_size,
        )
        no_answer_out = no_answer_out_len_sorted[orig_idxs]
        no_answer_predictions = self.no_answer_predictor(no_answer_out)

        return ModelPredictions(
            start_logits=start_predictions,
            end_logits=end_predictions,
            no_ans_logits=no_answer_predictions,
        )


class DocQAPredictor(PredictorModel):
    """
    Predictor as described in the DocumentQA paper
    """

    config: DocQAConfig
    embed: Embeddor
    embedding_encoder: ContextualEncoder
    bi_attention: DocQABidirectionalAttention
    attended_ctx_encoder: ContextualEncoder
    self_attention: SelfAttention
    output_layer: DocQAOutput

    def __init__(self, embeddor: Embeddor, config: DocQAConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = embeddor
        self.embedding_encoder = ContextualEncoder(
            self.embed.embedding_dim, self.config.gru
        )
        self.bi_attention = DocQABidirectionalAttention(
            self.config.gru.total_hidden_size, self.config.attention_linear_hidden_size
        )
        self.attended_ctx_encoder: ContextualEncoder = ContextualEncoder(
            self.bi_attention.final_encoding_size,
            GRUConfig(
                hidden_size=self.bi_attention.final_encoding_size // 2,
                num_layers=2,
                dropout_prob=self.config.gru.dropout_prob,
                force_unidirectional=False,
            ),
        )
        assert (
            self.attended_ctx_encoder.output_size
            == self.bi_attention.final_encoding_size
        )
        self.self_attention: SelfAttention = SelfAttention(
            self.attended_ctx_encoder.output_size,
            self.config.attention_linear_hidden_size,
        )
        self.output_layer = DocQAOutput(
            self.config.gru, self.bi_attention.final_encoding_size
        )

    def forward(self, batch: QABatch) -> ModelPredictions:
        """
        Check base class method for docs
        """

        question = self.embed(batch.question_words, batch.question_chars)
        question = self.embedding_encoder(
            question,
            batch.question_lens,
            batch.question_len_idxs,
            batch.question_orig_idxs,
        )

        context = self.embed(batch.context_words, batch.context_chars)
        context = self.embedding_encoder(
            context, batch.context_lens, batch.context_len_idxs, batch.context_orig_idxs
        )

        context = self.bi_attention(context, question, context_mask=batch.context_mask)

        self_aware_context = self.attended_ctx_encoder(
            context, batch.context_lens, batch.context_len_idxs, batch.context_orig_idxs
        )
        self_aware_context = self.self_attention(
            self_aware_context, self_aware_context, context_mask=batch.context_mask
        )
        context = context + self_aware_context

        return cast(
            ModelPredictions,
            self.output_layer(
                context,
                batch.context_mask,
                batch.context_lens,
                batch.context_len_idxs,
                batch.context_orig_idxs,
            ),
        )


class BidafOutput(nn.Module):
    """
    Module that produces prediction logits given a context encoding
    as described in the BiDAF paper
    """

    config: GRUConfig
    end_modeling_encoder: ContextualEncoder
    start_predictor: nn.Linear
    end_predictor: nn.Linear

    def __init__(
        self, config: GRUConfig, attended_input_size: int, modeled_input_size: int
    ) -> None:
        super().__init__()
        self.config = config
        self.end_modeling_encoder = ContextualEncoder(modeled_input_size, self.config)
        self.start_predictor = MaskedLinear(attended_input_size + modeled_input_size, 1)
        self.end_predictor = MaskedLinear(self.end_modeling_encoder.output_size, 1)
        self.softmax = MaskedLogSoftmax(dim=-1)

    def forward(
        self,
        attended_context: t.Tensor,
        modeled_context: t.Tensor,
        context_mask: t.LongTensor,
        lengths: t.LongTensor,
        length_idxs: t.LongTensor,
        orig_idxs: t.LongTensor,
    ) -> ModelPredictions:
        start_logits = self.start_predictor(
            t.cat([attended_context, modeled_context], dim=-1), mask=context_mask
        ).squeeze(2)

        end_modeled_ctx = self.end_modeling_encoder(
            modeled_context, lengths, length_idxs, orig_idxs
        )
        end_logits = self.start_predictor(
            t.cat([attended_context, end_modeled_ctx], dim=-1), mask=context_mask
        ).squeeze(2)

        start_predictions = self.softmax(start_logits, mask=context_mask)
        end_predictions = self.softmax(end_logits, mask=context_mask)
        return ModelPredictions(
            start_logits=start_predictions,
            end_logits=end_predictions,
            no_ans_logits=None,
        )


class BidafPredictor(PredictorModel):
    """
    Predictor as described in the BiDAF paper
    """

    config: GRUConfig
    embed: Embeddor
    embedding_encoder: ContextualEncoder
    bi_attention: BidafBidirectionalAttention
    modeling_layer: ContextualEncoder
    output_layer: BidafOutput

    def __init__(self, embeddor: Embeddor, config: GRUConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = embeddor
        self.embedding_encoder = ContextualEncoder(
            self.embed.embedding_dim, self.config
        )
        self.bi_attention = BidafBidirectionalAttention(self.config.total_hidden_size)
        self.modeling_layer = ContextualEncoder(
            self.bi_attention.final_encoding_size, self.config
        )
        self.output_layer = BidafOutput(
            self.config,
            self.bi_attention.final_encoding_size,
            self.modeling_layer.output_size,
        )

    def forward(self, batch: QABatch) -> ModelPredictions:
        """
        Check base class method for docs
        """

        question = self.embed(batch.question_words, batch.question_chars)
        question = self.embedding_encoder(
            question,
            batch.question_lens,
            batch.question_len_idxs,
            batch.question_orig_idxs,
        )

        context = self.embed(batch.context_words, batch.context_chars)
        context = self.embedding_encoder(
            context, batch.context_lens, batch.context_len_idxs, batch.context_orig_idxs
        )

        attended_context = self.bi_attention(
            context, question, context_mask=batch.context_mask
        )

        modeled_context = self.modeling_layer(
            attended_context,
            batch.context_lens,
            batch.context_len_idxs,
            batch.context_orig_idxs,
        )

        return cast(
            ModelPredictions,
            self.output_layer(
                attended_context,
                modeled_context,
                batch.context_mask,
                batch.context_lens,
                batch.context_len_idxs,
                batch.context_orig_idxs,
            ),
        )
