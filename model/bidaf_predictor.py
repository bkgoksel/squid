"""
Module that holds Bidaf classes that can be used for answer prediction
"""
from typing import Set, NamedTuple, Optional, cast
import torch as t
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from model.batcher import QABatch
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


class BidafConfig:
    """
    Object that holds config values for a Predictor model
    """

    contextual_encoder_config: ContextualEncoderConfig
    modeling_layer_config: ContextualEncoderConfig
    output_config: ContextualEncoderConfig

    def __init__(
        self,
        contextual_encoder_config: ContextualEncoderConfig,
        modeling_layer_config: ContextualEncoderConfig,
        output_config: ContextualEncoderConfig,
    ) -> None:
        self.contextual_encoder_config = contextual_encoder_config
        self.modeling_layer_config = modeling_layer_config
        self.output_config = output_config

    @classmethod
    def get_default_bidaf_config(cls) -> "BidafConfig":
        return BidafConfig(
            contextual_encoder_config=ContextualEncoderConfig(
                hidden_size=100, num_layers=1, dropout_input=False, dropout_prob=0.2
            ),
            modeling_layer_config=ContextualEncoderConfig(
                hidden_size=100, num_layers=2, dropout_input=False, dropout_prob=0.2
            ),
            output_config=ContextualEncoderConfig(
                hidden_size=100, num_layers=1, dropout_input=False, dropout_prob=0.2
            ),
        )


class BidafOutput(nn.Module):
    """
    Module that produces prediction logits given a context encoding
    as described in the BiDAF paper
    """

    config: ContextualEncoderConfig
    end_modeling_encoder: ContextualEncoder
    start_predictor: nn.Linear
    end_predictor: nn.Linear

    def __init__(
        self,
        config: ContextualEncoderConfig,
        attended_input_size: int,
        modeled_input_size: int,
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

    config: BidafConfig
    embed: Embeddor
    embedding_encoder: ContextualEncoder
    bi_attention: BidafBidirectionalAttention
    modeling_layer: ContextualEncoder
    output_layer: BidafOutput

    def __init__(self, embeddor: Embeddor, config: BidafConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = embeddor
        self.embedding_encoder = ContextualEncoder(
            self.embed.embedding_dim, self.config.contextual_encoder_config
        )
        self.bi_attention = BidafBidirectionalAttention(
            self.embedding_encoder.output_size
        )
        self.modeling_layer = ContextualEncoder(
            self.bi_attention.final_encoding_size, self.config.modeling_layer_config
        )
        self.output_layer = BidafOutput(
            self.config.output_config,
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
