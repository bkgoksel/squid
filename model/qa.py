"""
Module that encapsulates the objects to represent contexts, questions and
answers at various points of existence.
"""
import numpy as np
from typing import List, Any, Set
from tokenizer import Tokenizer
from wv import WordVectors


class Tokenized():
    """
    Base Class for any object that stores tokenized text
    """
    tokens: List[str]

    def __init__(self, text: str, tokenizer: Tokenizer) -> None:
        self.tokens = tokenizer.tokenize(text)


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

    def __eq__(self, other) -> bool:
        """
        Two answers are equal if their spans and text are equal
        """
        return (self.span_start == other.span_start and
                self.span_end == other.span_end and
                self.text == other.text)

    def __hash__(self):
        """
        Use the answer components to hash the answer
        """
        return hash("%d_%d_%s" % (self.span_start, self.span_end, self.text))


class QuestionAnswer(Tokenized):
    """
    Base class for a question paired with its answer.
    Stores the question text and a list of answers
    """
    text: str
    answers: Set[Answer]
    tokens: List[str]

    def __init__(self, text: str, answers: Set[Answer], tokenizer: Tokenizer) -> None:
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

    def __init__(self, text: str, qas: List[QuestionAnswer], tokenizer: Tokenizer) -> None:
        self.text = text
        self.qas = qas
        super().__init__(text, tokenizer)


class EncodedAnswer():
    """
    Class to store an Answer's encoding
    """
    span_start: int
    span_end: int
    encoding: Any  # numpy array

    def __init__(self, answer: Answer, word_vectors: WordVectors) -> None:
        self.span_start = answer.span_start
        self.span_end = answer.span_end


class EncodedQuestionAnswer():
    """
    Class for a Question's encoding
    Paired with its encoded answers
    """
    encoding: Any  # numpy array
    answers: List[EncodedAnswer]

    def __init__(self, qa: QuestionAnswer, word_vectors: WordVectors) -> None:
        self.encoding = np.array([word_vectors[tk] for tk in qa.tokens])
        self.answers = [EncodedAnswer(ans, word_vectors) for ans in qa.answers]


class EncodedContextQuestionAnswer():
    """
    Class for a context paragraph and its question-answer pairs
    Stores them in encoded form
    """
    encoding: Any  # numpy array
    qas: List[EncodedQuestionAnswer]

    def __init__(self, ctx: ContextQuestionAnswer, word_vectors: WordVectors) -> None:
        self.encoding = np.array([word_vectors[tk] for tk in ctx.tokens])
        self.qas = [EncodedQuestionAnswer(qa, word_vectors) for qa in ctx.qas]


class EncodedSample():
    """
    Stores a single model sample (context, question, answers)
    """
    question: Any  # numpy array
    context: Any  # numpy array
    has_answer: bool
    span_starts: Any  # numpy array
    span_ends: Any  # numpy array

    def __init__(self, ctx_encoding: Any, qa: EncodedQuestionAnswer) -> None:
        self.context = ctx_encoding
        self.question = qa.encoding
        self.has_answer = bool(qa.answers)
        self.span_starts = np.zeros_like(self.context)
        self.span_ends = np.zeros_like(self.context)
        """
        for answer in qa.answers:
            self.span_starts = answer.span_start
            self.answer_spans[i, 1] = answer.span_end
        """
        # TODO: Implement answer idx -> ctx token matching
        raise NotImplementedError
