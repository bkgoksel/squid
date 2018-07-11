"""
Module that holds classes built for embedding text
"""
from typing import Dict

import torch as t
import torch.nn as nn

from model.wv import WordVectors


class WordEmbeddor(nn.Module):
    """
    Module that embeds words using pretrained word vectors
    """
    embed: nn.Embedding

    def __init__(self, word_vectors: WordVectors, train_vecs: bool) -> None:
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(t.Tensor(word_vectors.vectors),
                                                  freeze=(not train_vecs))

    def forward(self, text: t.LongTensor) -> t.Tensor:
        return self.embed(text)


class PoolingCharEmbeddor(nn.Module):
    """
    Module that embeds words by training character-level embeddings and max pooling over them
    """
    embed: nn.Embedding

    def __init__(self, char_vocab_size: int, embedding_dimension: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(char_vocab_size, embedding_dimension, padding_idx=0)

    def forward(self, chars: t.LongTensor) -> t.Tensor:
        """
        :param chars: Characters of a batch organized in a tensor in the following shape:
            (batch_size, max_num_words, max_num_chars)
        1. Embeds the chars ->
            (batch, word, char_idx, embedding_size)
        2. Pools the embeddings ->
            (batch, word, embedding_size)

        """
        embeddings = self.embed(chars)
        pooled, _ = embeddings.max(2)
        return pooled


class WordCharPoolCombinedEmbeddor(nn.Module):
    """
    Module that embeds each word using a concatenation of
    its word embeddings and char-level max-pooled embeddings
    """
    word_embeddor: WordEmbeddor
    char_embeddor: PoolingCharEmbeddor

    def __init__(self,
                 word_vectors: WordVectors,
                 train_vecs: bool,
                 char_vocab_size: int,
                 char_embedding_dimension: int) -> None:
        super().__init__()
        self.word_embeddor = WordEmbeddor(word_vectors, train_vecs)
        self.char_embeddor = PoolingCharEmbeddor(char_vocab_size, char_embedding_dimension)

    def forward(self, word_encoding: t.LongTensor, char_encoding: t.LongTensor) -> t.Tensor:
        word_embeddings = self.word_embeddor(word_encoding)
        char_embeddings = self.char_embeddor(char_encoding)
        combined_embeddings = t.cat([word_embeddings, char_embeddings], dim=2)
        return combined_embeddings
