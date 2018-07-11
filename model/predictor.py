"""
Module that holds classes that can be used for answer prediction
"""
from typing import NamedTuple
import torch as t
import torch.nn as nn
from torch.nn.utils.rnn import (PackedSequence,
                                pack_padded_sequence,
                                pad_packed_sequence)

from model.wv import WordVectors
from model.batcher import QABatch
from model.modules.attention import SimpleAttention, AttentionConfig
from model.modules.masked import MaskedLinear
from model.modules.embeddor import (Embeddor,
                                    WordEmbeddor,
                                    PoolingCharEmbeddor,
                                    ConcatenatingEmbeddor)


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


class BasicPredictorConfig():
    """
    Object that holds config values for a BasicPredictor model
    """

    gru: GRUConfig
    attention: AttentionConfig
    train_vecs: bool
    batch_size: int
    n_directions: int
    char_vocab_size: int
    char_embedding_dimension: int

    def __init__(self,
                 gru: GRUConfig,
                 attention_hidden_size: int,
                 train_vecs: bool,
                 batch_size: int,
                 char_vocab_size: int,
                 char_embedding_dimension: int) -> None:
        self.gru = gru
        self.n_directions = 1 + int(self.gru.bidirectional)
        self.total_hidden_size = self.n_directions * self.gru.hidden_size
        self.attention = AttentionConfig(input_size=self.total_hidden_size,
                                         hidden_size=attention_hidden_size)
        self.batch_size = batch_size


class BasicPredictor(PredictorModel):
    """
    A very simple Predictor for testing
    """

    config: BasicPredictorConfig
    embed: Embeddor
    q_gru: nn.GRU
    q_hidden_state: t.Tensor
    ctx_gru: nn.GRU
    ctx_hidden_state: t.Tensor
    attention: SimpleAttention
    start_predictor: nn.Linear
    end_predictor: nn.Linear
    no_answer_gru: nn.GRU
    no_answer_predictor: nn.Linear

    def __init__(self, embeddor: Embeddor, config: BasicPredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = embeddor
        self.q_gru = nn.GRU(self.embed.embedding_dim,
                            self.config.gru.hidden_size,
                            self.config.gru.num_layers,
                            dropout=self.config.gru.dropout,
                            batch_first=True,
                            bidirectional=self.config.gru.bidirectional)
        self.ctx_gru = nn.GRU(self.embed.embedding_dim,
                              self.config.gru.hidden_size,
                              self.config.gru.num_layers,
                              dropout=self.config.gru.dropout,
                              batch_first=True,
                              bidirectional=self.config.gru.bidirectional)
        self.attention = SimpleAttention(self.config.attention)
        self.start_predictor = MaskedLinear(self.config.total_hidden_size, 1)
        self.end_predictor = MaskedLinear(self.config.total_hidden_size, 1)
        self.no_answer_gru = nn.GRU(self.config.total_hidden_size,
                                    self.config.gru.hidden_size,
                                    1,
                                    dropout=self.config.gru.dropout,
                                    batch_first=True,
                                    bidirectional=self.config.gru.bidirectional)
        self.no_answer_predictor = nn.Linear(self.config.total_hidden_size, 1)

    def forward(self, batch: QABatch) -> ModelPredictions:
        """
        Check base class method for docs
        """

        q_embedded = self.embed(batch.question_words, batch.question_chars)
        q_len_sorted = q_embedded[batch.question_len_idxs]
        q_packed: PackedSequence = pack_padded_sequence(q_len_sorted,
                                                        batch.question_lens,
                                                        batch_first=True)
        _, q_out_len_sorted = self.q_gru(q_packed)
        q_out_len_sorted = BasicPredictor.get_last_hidden_states(q_out_len_sorted, self.config)
        # Put the questions back in their pre-length-sort ordering so the
        # ordering matches with the context encoding
        q_out = q_out_len_sorted[batch.question_orig_idxs]

        ctx_embedded = self.embed(batch.context_words, batch.context_chars)
        ctx_len_sorted = ctx_embedded[batch.context_len_idxs]
        ctx_packed: PackedSequence = pack_padded_sequence(ctx_len_sorted,
                                                          batch.context_lens,
                                                          batch_first=True)
        ctx_processed_packed, _ = self.ctx_gru(ctx_packed)
        ctx_processed_len_sorted, _ = pad_packed_sequence(ctx_processed_packed, batch_first=True)
        # Put the contexts back in their pre-length-sort ordering so the
        # ordering matches with the question encoding
        ctx_processed = ctx_processed_len_sorted[batch.context_orig_idxs]

        attended = self.attention(q_out, ctx_processed, batch.context_mask)
        attended_length_sorted = attended[batch.context_len_idxs]
        attended_packed: PackedSequence = pack_padded_sequence(attended_length_sorted,
                                                               batch.context_lens,
                                                               batch_first=True)

        start_predictions = self.start_predictor(attended, batch.context_mask).squeeze(2)
        end_predictions = self.end_predictor(attended, batch.context_mask).squeeze(2)

        _, no_answer_out_len_sorted = self.no_answer_gru(attended_packed)

        no_answer_out_len_sorted = BasicPredictor.get_last_hidden_states(no_answer_out_len_sorted, self.config)
        no_answer_out = no_answer_out_len_sorted[batch.context_orig_idxs]
        no_answer_predictions = self.no_answer_predictor(no_answer_out)

        return ModelPredictions(start_logits=start_predictions,
                                end_logits=end_predictions,
                                no_ans_logits=no_answer_predictions)

    @staticmethod
    def get_last_hidden_states(states, config: BasicPredictorConfig):
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

        :returns: All hidden states, of shape:
            [batch_size, max_seq_len, hidden_size*n_dirs]

        """
        batch_size = states.size(1)
        states = states.transpose(0, 1)
        states = states[:, -config.n_directions:, :]
        out = states.contiguous().view(batch_size, config.total_hidden_size)
        return out
