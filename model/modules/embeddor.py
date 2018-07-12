"""
Module that holds classes built for embedding text
"""
from typing import Dict, List, NamedTuple, Optional

import torch as t
import torch.nn as nn

from model.wv import WordVectors


WordEmbeddorConfig = NamedTuple('WordEmbeddorConfig', [
    ('vectors', WordVectors),
    ('train_vecs', bool)
])


PoolingCharEmbeddorConfig = NamedTuple('PoolingCharEmbeddorConfig', [
    ('char_vocab_size', int),
    ('embedding_dimension', int)
])


EmbeddorConfig = NamedTuple('EmbeddorConfig', [
    ('word_embeddor', Optional[WordEmbeddorConfig]),
    ('char_embeddor', Optional[PoolingCharEmbeddorConfig])
])


class Embeddor(nn.Module):
    """
    Base class for embeddors
    """
    embedding_dim: int

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, words: t.LongTensor, chars: t.LongTensor) -> t.Tensor:
        """
        :param words: Words of the batch organized in a tensor in the following shape:
            (batch_size, max_num_words)
        :param chars: Characters of a batch organized in a tensor in the following shape:
            (batch_size, max_num_words, max_num_chars)
        :returns: Embeddings for each word of shape:
            (batch_size, max_num_words, embedding_dim)
        """
        raise NotImplementedError


class WordEmbeddor(Embeddor):
    """
    Module that embeds words using pretrained word vectors
    """
    embed: nn.Embedding

    def __init__(self, word_vectors: WordVectors, train_vecs: bool) -> None:
        super().__init__(word_vectors.dim)
        self.embed = nn.Embedding.from_pretrained(t.Tensor(word_vectors.vectors),
                                                  freeze=(not train_vecs))

    def forward(self, words: t.LongTensor, chars: t.LongTensor) -> t.Tensor:
        """
        :param words: Words of the batch organized in a tensor in the following shape:
            (batch_size, max_num_words)
        :param chars: Characters of a batch, unused by this module
        :returns: Pretrainde word embeddings for each word of shape:
            (batch_size, max_num_words, embedding_dim)
        """
        return self.embed(words)


class PoolingCharEmbeddor(Embeddor):
    """
    Module that embeds words by training character-level embeddings and max pooling over them
    """
    embed: nn.Embedding

    def __init__(self, char_vocab_size: int, embedding_dimension: int) -> None:
        super().__init__(embedding_dimension)
        self.embed = nn.Embedding(char_vocab_size + 1, embedding_dimension, padding_idx=0)

    def forward(self, words: t.LongTensor, chars: t.LongTensor) -> t.Tensor:
        """
        :param words: Words of the batch, unused by this module
        :param chars: Characters of a batch organized in a tensor in the following shape:
            (batch_size, max_num_words, max_num_chars)
        :returns: Character-level embeddings for each word of shape:
            (batch_size, max_num_words, embedding_dim)
        """
        embeddings = self.embed(chars)
        pooled, _ = embeddings.max(2)
        return pooled


class ConcatenatingEmbeddor(Embeddor):
    """
    Module that takes multiple Embeddors and concatenates their outputs to produce final embeddings
    """
    embeddors: List[Embeddor]

    def __init__(self, *args) -> None:
        super().__init__(sum(embeddor.embedding_dim for embeddor in args))
        self.embeddors = list(args)

    def forward(self, words: t.LongTensor, chars: t.LongTensor) -> t.Tensor:
        """
        :param words: Words of the batch organized in a tensor in the following shape:
            (batch_size, max_num_words)
        :param chars: Characters of a batch organized in a tensor in the following shape:
            (batch_size, max_num_words, max_num_chars)
        :returns: Concatenated embeddings from all the given embeddors
            (batch_size, max_num_words, embedding_dim)
        """
        embeddings = [embeddor(words, chars) for embeddor in self.embeddors]
        return t.cat(embeddings, dim=2)


def make_embeddor(config: EmbeddorConfig) -> Embeddor:
    """
    Makes an embeddor given an embeddor config
    :param config: An EmbeddorConfig object decsribing the embeddor to be made
    :returns: An Embeddor module as described by the config
    """
    embeddors = []
    assert(config.word_embeddor or config.char_embeddor)
    if config.word_embeddor:
        embeddors.append(WordEmbeddor(config.word_embeddor.vectors, config.word_embeddor.train_vecs))
    if config.char_embeddor:
        embeddors.append(PoolingCharEmbeddor(config.char_embeddor.char_vocab_size, config.char_embeddor.embedding_dimension))
    return ConcatenatingEmbeddor(*embeddors)

