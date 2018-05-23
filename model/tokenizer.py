"""
Module to hold tokenizer logic
"""

from typing import Any

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
