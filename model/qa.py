"""
Module that encapsulates the objects to represent contexts, questions and
answers at various points of existence.
"""
from typing import Any, List

class Answer():
    """
    Base class for an Answer, stores the text and span boundaries
    """
    text: str
    span_start: int
    span_end: int

    def __init__(self, text: str, span_start: int) -> None:
        self.text = text
        self.span_start = span_start
        self.span_end = self.span_start + len(self.text)

class QuestionAnswer():
    """
    Base class for a question paired with its answer.
    Stores the question text and a list of answers
    """
    text: str
    answers: List[Answer]

    def __init__(self, text: str, answers: List[Answer]) -> None:
        self.text = text
        self.answers = answers

class ContextQuestionAnswer(object):
    """
    Base class for a context paragraph and its question-answer pairs
    Stores the context text and a list of question answer pair objects
    """
    text: str
    qas: List[QuestionAnswer]

    def __init__(self, context_text: str, qas: List[QuestionAnswer]) -> None:
        self.context_text = context_text
        self.qas = qas


class TokenizedAnswer(Answer):
    """
    Class to store all answer data as well as a list of tokens from the text
    """
    tokens: List[str]
    num_tokens: int
    text: str
    span_start: int
    span_end: int

    def __init__(self, text: str, span_start: int, tokenizer: Any) -> None:
        super().__init__(text, span_start)
        self.tokens = tokenizer.tokenize(self.text)
        self.num_tokens = len(self.tokens)

class TokenizedQA(QuestionAnswer):
    """
    Class to store all QA data as well as the tokenized answers and question
    text tokens
    """
    tokens: List[str]
    num_tokens: int
    text: str
    answers: List[TokenizedAnswer]

    def __init__(self, text: str, answers: List[TokenizedAnswer], tokenizer: Any) -> None:
        super().__init__(text, answers)
        self.tokens = tokenizer.tokenize(self.text)
        self.num_tokens = len(self.tokens)

class TokenizedContextQA(ContextQuestionAnswer):
    """
    Class to store the context text, its tokens and the related question-answer pairs
    """
    tokens: List[str]
    num_tokens: int
    text: str
    qas: List[TokenizedQA]

    def __init__(self, context_text: str, qas: List[TokenizedQA], tokenizer: Any) -> None:
        super().__init__(context_text, qas)
        self.tokens = tokenizer.tokenize(self.context_text)
        self.num_tokens = len(self.tokens)
