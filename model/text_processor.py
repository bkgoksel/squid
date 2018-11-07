"""
Module to store various text processors
"""

from typing import Set, Dict


class TextProcessor:
    """
    Base class for text processors
    """

    config: Dict[str, bool]

    def __init__(self, config: Dict[str, bool]) -> None:
        self.config = config

    def process(self, text: str) -> str:
        """
        Takes in a bunch of text and processes it according to its configuration
        :param text: String to be processed
        :returns: String of processed text
        """
        return text.lower() if self.config["lowercase"] else text
