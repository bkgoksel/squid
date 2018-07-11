"""
Module for testing embeddor model utilities
"""

import unittest
from unittest.mock import Mock, MagicMock

import numpy as np
import torch as t
import torch.nn as nn

from model.modules.embeddor import (WordEmbeddor,
                                    PoolingCharEmbeddor,
                                    WordCharPoolCombinedEmbeddor)
from model.wv import WordVectors


class WordEmbeddorTestCase(unittest.TestCase):
    def setUp(self):
        self.vectors = Mock(WordVectors)
        self.word_embedding_size = 5
        self.word_vocab_len = 3
        self.vectors.vectors = np.ones((self.word_vocab_len, self.word_embedding_size))
        for i in range(self.word_vocab_len):
            self.vectors.vectors[i] *= i

    def test_word_embeddor(self):
        embeddor = WordEmbeddor(self.vectors, False)
        words_list = [[1, 0], [2, 2]]
        words = t.LongTensor(words_list)
        embedded = embeddor(words)
        # Check results
        self.assertEqual(embedded.size(), t.Size([len(words_list),  # batch size
                                                  len(words_list[0]),  # max seq len
                                                  self.word_embedding_size]))
        for sample_idx, sample in enumerate(words_list):
            for word_idx, word in enumerate(sample):
                self.assertTrue(np.all(embedded[sample_idx, word_idx].numpy() == np.ones(self.word_embedding_size) * word))


class PoolingCharEmbeddorTestCase(unittest.TestCase):
    def setUp(self):
        self.char_vocab_size = 3
        self.char_embedding_size = 4

    def test_pooling_char_embeddor(self):
        embeddor = PoolingCharEmbeddor(self.char_vocab_size, self.char_embedding_size)
        chars_list = [[[1, 1], [0, 0]], [[2, 0], [2, 0]]]
        chars = t.LongTensor(chars_list)
        embedded = embeddor(chars)
        # Check results
        self.assertEqual(embedded.size(), t.Size([len(chars_list),  # batch size
                                                  len(chars_list[0]),  # max seq len
                                                  len(chars_list[0][0]),  # max word len
                                                  self.char_embedding_size]))


class WordCharPoolCombinedEmbeddorTestCase(unittest.TestCase):
    def setUp(self):
        self.word_embedding_size = 5
        self.word_vocab_len = 3
        self.char_vocab_size = 3
        self.char_embedding_size = 4

        self.total_embedding_size = self.word_embedding_size + self.char_embedding_size
        self.vectors = Mock(WordVectors)
        self.vectors.vectors = np.ones((self.word_vocab_len, self.word_embedding_size))
        for i in range(self.word_vocab_len):
            self.vectors.vectors[i] *= i

    def test_word_and_char_pooling_combined_embeddor(self):
        embeddor = WordCharPoolCombinedEmbeddor(self.vectors, False, self.char_vocab_size, self.char_embedding_size)

        words_list = [[1, 0], [2, 2]]
        words = t.LongTensor(words_list)

        chars_list = [[[1, 1], [0, 0]], [[2, 0], [2, 0]]]
        chars = t.LongTensor(chars_list)
        embedded = embeddor(words, chars)
        # Check results
        self.assertEqual(embedded.size(), t.Size([len(words_list),  # batch size
                                                  len(words_list[0]),  # max seq len
                                                  self.total_embedding_size]))
        for sample_idx, sample in enumerate(words_list):
            for word_idx, word in enumerate(sample):
                # Word Embeddings are pretrained so should be the same after concat
                self.assertTrue(np.all(embedded[sample_idx, word_idx, :self.word_embedding_size].numpy() == np.ones(self.word_embedding_size) * word))
