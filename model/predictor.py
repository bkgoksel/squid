"""
Module that holds classes that can be used for answer prediction
"""

from wv import WordVectors
from batcher import QABatch
from modules.attention import SimpleAttention, AttentionConfig
from modules.masked import MaskedLinear

from typing import NamedTuple
import torch as t
import torch.nn as nn
from torch.nn.utils.rnn import (PackedSequence,
                                pack_padded_sequence,
                                pad_packed_sequence)


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

    def __init__(self,
                 gru: GRUConfig,
                 attention_hidden_size: int,
                 train_vecs: bool,
                 batch_size: int) -> None:
        self.gru = gru
        self.n_directions = 1 + int(self.gru.bidirectional)
        self.total_hidden_size = self.n_directions * self.gru.hidden_size
        self.attention = AttentionConfig(input_size=self.total_hidden_size,
                                         hidden_size=attention_hidden_size)
        self.train_vecs = train_vecs
        self.batch_size = batch_size


class BasicPredictor(PredictorModel):
    """
    A very simple Predictor for testing
    """

    word_vectors: WordVectors
    config: BasicPredictorConfig
    embed: nn.Embedding
    q_gru: nn.GRU
    q_hidden_state: t.Tensor
    ctx_gru: nn.GRU
    ctx_hidden_state: t.Tensor
    attention: SimpleAttention
    start_predictor: nn.Linear
    end_predictor: nn.Linear
    no_answer_gru: nn.GRU
    no_answer_predictor: nn.Linear

    def __init__(self, word_vectors: WordVectors, config: BasicPredictorConfig) -> None:
        super().__init__()
        self.word_vectors = word_vectors
        self.config = config
        self.embed = nn.Embedding.from_pretrained(t.Tensor(self.word_vectors.vectors),
                                                  freeze=(not self.config.train_vecs))
        self.q_gru = nn.GRU(self.word_vectors.dim,
                            self.config.gru.hidden_size,
                            self.config.gru.num_layers,
                            dropout=self.config.gru.dropout,
                            batch_first=True,
                            bidirectional=self.config.gru.bidirectional)
        self.ctx_gru = nn.GRU(self.word_vectors.dim,
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

        q_embedded = self.embed(batch.questions)
        q_packed: PackedSequence = pack_padded_sequence(q_embedded,
                                                        batch.question_lens,
                                                        batch_first=True)
        _, q_out = self.q_gru(q_packed)
        q_out = self.get_last_hidden_states(q_out)
        # Put the questions back in their pre-length-sort ordering so the
        # ordering matches with the context encoding
        q_out = q_out[batch.question_orig_idxs]

        ctx_embedded = self.embed(batch.contexts)
        ctx_packed: PackedSequence = pack_padded_sequence(ctx_embedded,
                                                          batch.context_lens,
                                                          batch_first=True)
        ctx_processed, _ = self.ctx_gru(ctx_packed)
        ctx_processed, _ = pad_packed_sequence(ctx_processed, batch_first=True)
        # Put the contexts back in their pre-length-sort ordering so the
        # ordering matches with the question encoding
        ctx_processed = ctx_processed[batch.context_orig_idxs]

        original_sorted_context_mask = batch.context_mask[batch.context_orig_idxs]
        attended = self.attention(q_out, ctx_processed, original_sorted_context_mask)
        attended_length_sorted = attended[batch.context_len_idxs]
        attended_packed: PackedSequence = pack_padded_sequence(attended_length_sorted,
                                                               batch.context_lens,
                                                               batch_first=True)

        start_predictions = self.start_predictor(attended, original_sorted_context_mask).squeeze(2)
        end_predictions = self.end_predictor(attended, original_sorted_context_mask).squeeze(2)

        _, no_answer_out = self.no_answer_gru(attended_packed)

        no_answer_out = self.get_last_hidden_states(no_answer_out)
        no_answer_out = no_answer_out[batch.context_orig_idxs]
        no_answer_predictions = self.no_answer_predictor(no_answer_out)

        return ModelPredictions(start_logits=start_predictions,
                                end_logits=end_predictions,
                                no_ans_logits=no_answer_predictions)

    def get_last_hidden_states(self, out):
        """
        Do some juggling with the output of the RNNs to get the
            final hidden states of the topmost layers of all the
            directions to feed into attention
        (should be in same tensor layout as the all hidden states of context)

        'q_out' here is 'u' from the paper

        q_processed: All hidden states, of shape:
            [batch_size, max_seq_len, hidden_size*n_dirs]

        q_out: Last hidden states for all layers and directions, of shape:
            [n_layers*n_dirs, batch_size, hidden_size]:
                The first dimension is laid out like:
                    layer0dir0, layer0dir1, layer1dir0, layer1dir1

        To get it in the same config as q_processed:
            1. Make q_out batch first
            2. Only keep the last layers for each direction
            3. Concatenate the layer hidden states in one dimension
        """
        batch_size = out.size(1)
        out = out.transpose(0, 1)
        out = out[:, -self.config.n_directions:, :]
        out = out.contiguous().view(batch_size, self.config.total_hidden_size)
        return out
