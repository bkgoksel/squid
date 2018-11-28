"""
Module that holds classes built for embedding text
"""
from typing import Any, Dict, List, NamedTuple, Optional

import torch as t
import torch.nn as nn

from model.wv import WordVectors


WordEmbeddorConfig = NamedTuple(
    "WordEmbeddorConfig", [("vectors", WordVectors), ("train_vecs", bool)]
)


PoolingCharEmbeddorConfig = NamedTuple(
    "PoolingCharEmbeddorConfig",
    [("char_vocab_size", int), ("embedding_dimension", int)],
)


EmbeddorConfig = NamedTuple(
    "EmbeddorConfig",
    [
        ("highway_layers", int),
        ("word_embeddor", Optional[WordEmbeddorConfig]),
        ("char_embeddor", Optional[PoolingCharEmbeddorConfig]),
    ],
)


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

    def __init__(
        self, word_vectors: WordVectors, train_vecs: bool, device: Any = t.device("cpu")
    ) -> None:
        super().__init__(word_vectors.dim)
        embedding_matrix = t.Tensor(word_vectors.vectors).to(device)
        self.embed = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=(not train_vecs)
        )
        self.to(device)

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

    def __init__(
        self,
        char_vocab_size: int,
        embedding_dimension: int,
        device: Any = t.device("cpu"),
    ) -> None:
        super().__init__(embedding_dimension)
        self.vocab_size = char_vocab_size + 1
        self.embed = nn.Embedding(
            char_vocab_size + 1, embedding_dimension, padding_idx=0
        )
        self.to(device)

    def forward(self, words: t.LongTensor, chars: t.LongTensor) -> t.Tensor:
        """
        :param words: Words of the batch, unused by this module
        :param chars: Characters of a batch organized in a tensor in the following shape:
            (batch_size, max_num_words, max_num_chars)
        :returns: Character-level embeddings for each word of shape:
            (batch_size, max_num_words, embedding_dim)
        """
        batch_size, max_num_words, max_num_chars = chars.size()
        # Flatten the word length dimension to make Tensor 2D for embedding
        chars = chars.view(-1, max_num_chars)
        embeddings = self.embed(chars)
        embeddings = embeddings.view(
            batch_size, max_num_words, max_num_chars, self.embedding_dim
        )
        pooled, _ = embeddings.max(2)
        return pooled


class ConcatenatingEmbeddor(Embeddor):
    """
    Module that takes multiple Embeddors and concatenates their outputs to produce final embeddings
    """

    embeddors: List[Embeddor]

    def __init__(self, embeddors: List[Embeddor]) -> None:
        super().__init__(sum(embeddor.embedding_dim for embeddor in embeddors))
        self.embeddors = nn.ModuleList(embeddors)

    def forward(self, words: t.LongTensor, chars: t.LongTensor) -> t.Tensor:
        """
        :param words: Words of the batch organized in a tensor in the following shape:
            (batch_size, max_num_words)
        :param chars: Characters of a batch organized in a tensor in the following shape:
            (batch_size, max_num_words, max_num_chars)
        :returns: Concatenated embeddings from all the given embeddors
            (batch_size, max_num_words, embedding_dim)
        """
        return t.cat([embeddor(words, chars) for embeddor in self.embeddors], dim=2)


class HighwayEmbeddor(ConcatenatingEmbeddor):
    """
    Module that takes multiple Embeddors and runs a Highway network on their concatenated outputs
    """

    embeddors: List[Embeddor]
    n_layers: int
    normal_layer: nn.ModuleList
    gate_layer: nn.ModuleList
    relu: nn.ReLU
    sigmoid: nn.Sigmoid

    def __init__(self, embeddors: List[Embeddor], n_layers: int = 1) -> None:
        super().__init__(embeddors)
        self.n_layers = n_layers
        self.normal_layer = nn.ModuleList(
            [nn.Linear(self.embedding_dim, self.embedding_dim) for _ in range(n_layers)]
        )
        self.gate_layer = nn.ModuleList(
            [nn.Linear(self.embedding_dim, self.embedding_dim) for _ in range(n_layers)]
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, words: t.LongTensor, chars: t.LongTensor) -> t.Tensor:
        """
        :param words: Words of the batch organized in a tensor in the following shape:
            (batch_size, max_num_words)
        :param chars: Characters of a batch organized in a tensor in the following shape:
            (batch_size, max_num_words, max_num_chars)
        :returns: Concatenated and highway networked embeddings from all the given embeddors
            (batch_size, max_num_words, embedding_dim)
        """
        embeddings = super().forward(words, chars)
        for i in range(self.n_layers):
            normal_layer_ret = self.relu(self.normal_layer[i](embeddings))
            gate = self.sigmoid(self.gate_layer[i](embeddings))

            embeddings = gate * normal_layer_ret + (1 - gate) * embeddings
        return embeddings


def make_embeddor(config: EmbeddorConfig, device: Any) -> Embeddor:
    """
    Makes an embeddor given an embeddor config on the given device
    :param config: An EmbeddorConfig object decsribing the embeddor to be made
    :param device: Torch device to put the embeddor on
    :returns: An Embeddor module as described by the config
    """
    embeddors = []
    assert (
        config.word_embeddor or config.char_embeddor
    ), "At least one of WordEmbeddor and CharEmbeddor needs to be specified"
    if config.word_embeddor:
        embeddors.append(
            WordEmbeddor(
                config.word_embeddor.vectors, config.word_embeddor.train_vecs, device
            )
        )
    if config.char_embeddor:
        embeddors.append(
            PoolingCharEmbeddor(
                config.char_embeddor.char_vocab_size,
                config.char_embeddor.embedding_dimension,
                device,
            )
        )
    if config.highway_layers:
        return HighwayEmbeddor(embeddors, config.highway_layers)
    else:
        return ConcatenatingEmbeddor(embeddors)
