"""
Module that holds classes built for embedding text
"""

import torch as t
import torch.nn as nn

from model.wv import WordVectors


class WordEmbeddor(nn.Module):
    """
    Module that embeds words using pretrained word vectors
    """
    def __init__(self, word_vectors: WordVectors, train_vecs: bool) -> None:
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(t.Tensor(word_vectors.vectors),
                                                  freeze=(not train_vecs))

    def forward(self, text: t.LongTensor):
        return self.embed(text)


class CharEmbeddor(nn.Module):
    """
    Module that embeds words using their characters
    """
    pass
