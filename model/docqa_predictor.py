"""
Module that holds Bidaf classes that can be used for answer prediction
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
from model.predictor import (
    PredictorModel,
    ContextualEncoderConfig,
    ContextualEncoder,
    ModelPredictions,
)


class DocQAConfig:
    """
    Object that holds config values for a Predictor model
    """

    contextual_encoder_config: ContextualEncoderConfig
    dropout_prob: float
    attention_linear_hidden_size: int
    use_self_attention: bool
    batch_size: int

    def __init__(
        self,
        contextual_encoder_config: ContextualEncoderConfig,
        dropout_prob: float,
        attention_linear_hidden_size: int,
        use_self_attention: bool,
        batch_size: int,
    ) -> None:
        self.contextual_encoder_config = contextual_encoder_config
        self.dropout_prob = dropout_prob
        self.attention_linear_hidden_size = attention_linear_hidden_size
        self.batch_size = batch_size
        self.use_self_attention = use_self_attention


class DocQAOutput(nn.Module):
    """
    Module that produces prediction logits given a context encoding
    as described in the DocumentQA paper
    """

    config: ContextualEncoderConfig
    start_modeling_encoder: ContextualEncoder
    end_modeling_encoder: ContextualEncoder
    start_predictor: nn.Linear
    end_predictor: nn.Linear
    no_answer_gru: nn.GRU
    no_answer_predictor: nn.Linear

    def __init__(self, config: ContextualEncoderConfig, input_size: int) -> None:
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
            self.embed.embedding_dim, self.config.contextual_encoder_config
        )
        self.bi_attention = DocQABidirectionalAttention(
            self.config.contextual_encoder_config.total_hidden_size,
            self.config.attention_linear_hidden_size,
        )
        self.attended_ctx_encoder: ContextualEncoder = ContextualEncoder(
            self.bi_attention.final_encoding_size,
            ContextualEncoderConfig(
                hidden_size=self.bi_attention.final_encoding_size // 2,
                num_layers=2,
                dropout_input=True,
                dropout_prob=self.config.contextual_encoder_config.dropout_prob,
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
            self.config.contextual_encoder_config, self.bi_attention.final_encoding_size
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
