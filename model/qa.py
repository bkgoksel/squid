"""
Module that encapsulates the objects to represent contexts, questions and
answers at various points of existence.
"""
from typing import List
from tokenizer import Tokenizer


class Tokenized():
    """
    Base Class for any object that stores tokenized text
    """
    tokens: List[str]
    num_tokens: int

    def __init__(self, text: str, tokenizer: Tokenizer) -> None:
        self.tokens = tokenizer.tokenize(text)
        self.num_tokens = len(self.tokens)


class Answer(Tokenized):
    """
    Base class for an Answer, stores the text and span boundaries
    """
    text: str
    span_start: int
    span_end: int
    tokens: List[str]
    num_tokens: int

    def __init__(self, text: str, span_start: int, tokenizer: Tokenizer) -> None:
        self.text = text
        self.span_start = span_start
        self.span_end = self.span_start + len(self.text)
        super().__init__(text, tokenizer)


class QuestionAnswer(Tokenized):
    """
    Base class for a question paired with its answer.
    Stores the question text and a list of answers
    """
    text: str
    answers: List[Answer]
    tokens: List[str]
    num_tokens: int

    def __init__(self, text: str, answers: List[Answer], tokenizer: Tokenizer) -> None:
        self.text = text
        self.answers = answers
        super().__init__(text, tokenizer)


class ContextQuestionAnswer(Tokenized):
    """
    Base class for a context paragraph and its question-answer pairs
    Stores the context text and a list of question answer pair objects
    """
    text: str
    qas: List[QuestionAnswer]
    tokens: List[str]
    num_tokens: int

    def __init__(self, text: str, qas: List[QuestionAnswer], tokenizer: Tokenizer) -> None:
        self.text = text
        self.qas = qas
        super().__init__(text, tokenizer)
