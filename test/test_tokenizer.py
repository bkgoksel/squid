"""
Module for testing dataset representations
"""
import unittest
from unittest.mock import Mock, MagicMock

from nltk.tokenize import WordPunctTokenizer

from typing import List
from model.tokenizer import Tokenizer, NltkTokenizer, Token


class BatcherTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_nltk_tokenize(self):
        """
        TODO:
            - Mock WordPunctTokenizer to return a predetermined
                set of span ranges for span_tokenize
            - Make sure the Token objects are built correctly
        """
        pass
