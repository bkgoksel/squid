"""
Module that holds classes that can be used for answer prediction
"""

from corpus import CorpusStats
from qa import SampleBatch
from wv import WordVectors
from typing import NamedTuple, Tuple, List
import torch as t
import torch.nn as nn


class PredictorModel(nn.Module):
    """
    Base class for any Predictor Model
    """
    def __init__(self):
        super().__init__()

    def forward(self, batch: SampleBatch) -> Tuple[t.Tensor, t.float32]:
        """
        Predicts ((span_start, span_end), has_ans_prob) for a batch of samples
        :param batch: SampleBatch: a batch of samples returned from a batcher
        :returns: Tuple of (IntTensor[span_start, span_end], ans_prob:float32)
        """
        raise NotImplementedError


GRUConfig = NamedTuple('GRUConfig', [
    ('hidden_size', int),
    ('num_layers', int),
    ('dropout', float),
    ('bidirectional', bool)
])

BasicPredictorConfig = NamedTuple('BasicPredictorConfig', [
    ('gru', GRUConfig),
    ('train_vecs', bool)
])


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

    def __init__(self, word_vectors: WordVectors, corpus_stats: CorpusStats, config: BasicPredictorConfig) -> None:
        super().__init__()
        self.corpus_stats = corpus_stats
        self.word_vectors = word_vectors
        self.config = config
        self.embed = nn.Embedding.from_pretrained(self.word_vectors.vectors,
                                                  freeze=(not self.config.train_vecs))
        self.q_gru = nn.GRU(self.corpus_stats.max_q_len,
                            self.config.gru.hidden_size,
                            self.config.gru.num_layers,
                            dropout=self.config.gru.dropout,
                            bidirectional=self.config.gru.bidirectional)
        self.ctx_gru = nn.GRU(self.corpus_stats.max_context_len,
                              self.config.gru.hidden_size,
                              self.config.gru.num_layers,
                              dropout=self.config.gru.dropout,
                              bidirectional=self.config.gru.bidirectional)

    def forward(self, batch: SampleBatch) -> Tuple[t.Tensor, t.float32]:
        """
        Check base class method for docs
        """
        questions = t.Tensor(batch.questions)
        contexts = t.Tensor(batch.contexts)

        q_embedded = self.embed(questions)
        ctx_embedded = self.embed(contexts)

        q_processed, self.q_hidden_state = self.q_gru(q_embedded, self.q_hidden_state)
        ctx_processed, self.ctx_hidden_state = self.ctx_gru(ctx_embedded, self.ctx_hidden_state)

        return None, None
