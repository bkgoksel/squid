"""
Module that holds classes that can be used for answer prediction
"""

from corpus import CorpusStats
from wv import WordVectors
from batcher import QABatch
from modules.attention import SimpleAttention, AttentionConfig

from typing import NamedTuple, Tuple
import torch as t
import torch.nn as nn
from torch.nn.utils.rnn import (PackedSequence,
                                pack_padded_sequence,
                                pad_packed_sequence)


class PredictorModel(nn.Module):
    """
    Base class for any Predictor Model
    """
    def __init__(self):
        super().__init__()

    def forward(self, batch: QABatch) -> Tuple[t.LongTensor, t.FloatTensor]:
        """
        Predicts ((span_start, span_end), has_ans_prob) for a batch of samples
        :param batch: QABatch: a batch of samples returned from a batcher
        :returns: Tuple of (LongTensor[span_start, span_end], ans_prob:float32)
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

    def __init__(self,
                 gru: GRUConfig,
                 attention_hidden_size: int,
                 train_vecs: bool,
                 batch_size: int) -> None:
        self.gru = gru
        self.attention = AttentionConfig(input_size=2*self.gru.hidden_size,
                                         hidden_size=attention_hidden_size)
        self.train_vecs = train_vecs
        self.batch_size = batch_size


class BasicPredictor(PredictorModel):
    """
    A very simple Predictor for testing
    """

    corpus_stats: CorpusStats
    word_vectors: WordVectors
    config: BasicPredictorConfig
    embed: nn.Embedding
    q_gru: nn.GRU
    q_hidden_state: t.Tensor
    ctx_gru: nn.GRU
    ctx_hidden_state: t.Tensor
    attention: SimpleAttention

    def __init__(self, word_vectors: WordVectors, corpus_stats: CorpusStats, config: BasicPredictorConfig) -> None:
        super().__init__()
        self.corpus_stats = corpus_stats
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
        self.q_hidden_state = t.zeros((self.config.gru.num_layers*(1 + int(self.config.gru.bidirectional)),
                                            self.config.batch_size,
                                            self.config.gru.hidden_size))
        self.ctx_gru = nn.GRU(self.word_vectors.dim,
                              self.config.gru.hidden_size,
                              self.config.gru.num_layers,
                              dropout=self.config.gru.dropout,
                              batch_first=True,
                              bidirectional=self.config.gru.bidirectional)
        self.ctx_hidden_state = t.zeros((self.config.gru.num_layers*(1 + int(self.config.gru.bidirectional)),
                                              self.config.batch_size,
                                              self.config.gru.hidden_size))
        self.attention = SimpleAttention(self.config.attention)

    def forward(self, batch: QABatch) -> Tuple[t.LongTensor, t.FloatTensor]:
        """
        Check base class method for docs
        """

        q_embedded = self.embed(batch.questions)
        q_packed: PackedSequence = pack_padded_sequence(q_embedded,
                                                        batch.question_lens,
                                                        batch_first=True)
        # Don't update q_hidden_state as batches are independent
        q_processed, q_out = self.q_gru(q_packed, self.q_hidden_state)

        # Juggle with RNN output to get concatenated hidden states
        q_out = q_out.transpose(0,1)
        q_out = q_out[:, -(1 + int(self.config.gru.bidirectional)):,:]
        q_out = q_out.contiguous().view((self.config.batch_size, self.config.gru.hidden_size*(1 + int(self.config.gru.bidirectional))))

        q_processed, q_lens = pad_packed_sequence(q_processed, batch_first=True)
        """
        q_processed: [batch_sample, seq_elem, hidden*dirs]
        q_out: [batch_sample, layers*dirs, hidden]
            layer0dir0, layer0dir1, layer1dir0, layer1dir1

        processed to :

        q_processed: [batch_sample, seq_elem, hidden*dirs]
        q_out: [batch_samples, dirs, hidden]

        q_out[sample1, 0, 0]

        q_out[s,2,0] = q_processed[s,(q_lens[s]-1),0]
        q_out[s,3,0] = q_processed[s, 0, 256]

        """

        ctx_embedded = self.embed(batch.questions)
        ctx_packed: PackedSequence = pack_padded_sequence(ctx_embedded,
                                                       batch.question_lens,
                                                       batch_first=True)
        # Don't update ctx_hidden_state as batches are independent
        ctx_processed, _ = self.ctx_gru(ctx_packed, self.ctx_hidden_state)
        ctx_processed, ctx_lens = pad_packed_sequence(ctx_processed, batch_first=True)

        attended = self.attention(q_out, ctx_processed)

        return None, None
