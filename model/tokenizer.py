"""
Module to hold tokenizer logic
"""

from typing import List
from nltk.tokenize import WordPunctTokenizer


class Tokenizer():
    """
    Base class for tokenizer wrappers
    """
    def tokenize(self, text: str) -> List[str]:
        """
        Takes a string text and tokenizes it
        :param text: A string that contains some text
        :returns: A list of strings, tokens in the order they
            appear in the text
        """
        raise NotImplementedError


class NltkTokenizer(Tokenizer):

    """
    Tokenizer that uses WordPunctTokenizer from NLTK
    """
    tokenizer: WordPunctTokenizer

    def __init__(self):
        self.tokenizer = WordPunctTokenizer

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)
