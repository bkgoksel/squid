"""
Module to hold tokenizer logic
"""

from typing import List, Tuple, NamedTuple
from nltk.tokenize import WordPunctTokenizer


Token = NamedTuple("Token", [("word", str), ("span", Tuple[int, int])])


class Tokenizer:
    """
    Base class for tokenizer wrappers
    """

    def tokenize(self, text: str) -> List[Token]:
        """
        Takes a string text and tokenizes it
        :param text: A string that contains some text
        :returns: A list of Tokens, tokens in the order they
            appear in the text and their character spans
        """
        raise NotImplementedError


class NltkTokenizer(Tokenizer):

    """
    Tokenizer that uses WordPunctTokenizer from NLTK
    """

    tokenizer: WordPunctTokenizer

    def __init__(self):
        self.tokenizer = WordPunctTokenizer()

    def tokenize(self, text: str) -> List[Token]:
        spans = list(self.tokenizer.span_tokenize(text))
        words = [text[span_start:span_end] for span_start, span_end in spans]
        return [Token(word=word, span=span) for word, span in zip(words, spans)]
