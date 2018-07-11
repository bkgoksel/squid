"""
Module for testing embeddor model utilities
"""

import unittest
from unittest.mock import Mock, MagicMock

import numpy as np
import torch as t
import torch.nn as nn

from model.embeddor import (WordEmbeddor,
                            PoolingCharEmbeddor,
                            WordCharPoolCombinedEmbeddor)
from model.wv import WordVectors


class WordEmbeddorTestCase(unittest.TestCase):
    def setUp(self):
        vocab = ['c1', 'c2', 'c3', 'c4', 'c00']
        char_vocab = set([char for word in vocab for char in word])
        self.token_id_mapping = dict(map(reversed, enumerate(vocab, 1)))
        self.id_token_mapping = dict(enumerate(vocab, 1))
        self.char_mapping = dict(map(reversed, enumerate(char_vocab, 1)))
        self.vectors = MagicMock(WordVectors)
        self.vectors.__getitem__.side_effect = lambda tok: self.token_id_mapping[tok]

    # TODO: Write the test cases for the embeddors

