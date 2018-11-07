"""
Module for testing dataset representations
"""
import unittest
from unittest.mock import Mock, MagicMock

from nltk.tokenize import WordPunctTokenizer

from typing import List
from model.wv import WordVectors


class WordVectorsTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_reading_simple(self):
        """
        TODO:
            - Read a simple set of vectors from a file:
                - no consume_first_line
                - no add_unk_token
                - no add_pad_token
            - Make sure vectors matches the number
            - Make sure the word_to_idx and idx_to_word match
            - Make sure word has the correct index
        """
        pass

    def test_reading_add_unk(self):
        """
        TODO:
            - Make sure the correct UNK token is appended to the vectors
        """
        pass

    def test_reading_add_padd(self):
        """
        TODO:
            - Make sure the correct PAD token is prepended to the vectors
            - Make sure word indexing is still correct
        """
        pass

    def test_reading_consume_first_line(self):
        """
        TODO:
            - Make sure when consume_first_line is set the first line is ignored
        """
        pass

    def test_serialization(self):
        """
        TODO:
            - Assert that serialized-deserialized vectors are the same as
            the original vectors
        """
        pass
