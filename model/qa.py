"""
Module that encapsulates the objects to represent contexts, questions and
answers at various points of existence.
"""
import numpy as np
from typing import List, Any, Dict
from tokenizer import Tokenizer


class Tokenized():
    """
    Base Class for any object that stores tokenized text
    """
    tokens: List[str]

    def __init__(self, text: str, tokenizer: Tokenizer) -> None:
        self.tokens = tokenizer.tokenize(text)


class Answer(Tokenized):
    """
    Base class for an Answer, stores the text and span boundaries
    """
    text: str
    span_start: int
    span_end: int
    tokens: List[str]

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

    def __init__(self, answer: Answer, word_to_idx: Dict[str, int]) -> None:
        self.span_start = answer.span_start
        self.span_end = answer.span_end
        self.encoding = np.array([word_to_idx[tk] for tk in answer.tokens])


class EncodedQuestionAnswer():
    """
    Class for a Question's encoding
    Paired with its encoded answers
    """
    encoding: Any  # numpy array
    answers: List[EncodedAnswer]

    def __init__(self, qa: QuestionAnswer, word_to_idx: Dict[str, int]) -> None:
        self.encoding = np.array([word_to_idx[tk] for tk in qa.tokens])
        self.answers = [EncodedAnswer(ans, word_to_idx) for ans in qa.answers]


class EncodedContextQuestionAnswer():
    """
    Class for a context paragraph and its question-answer pairs
    Stores them in encoded form
    """
    encoding: Any  # numpy array
    qas: List[EncodedQuestionAnswer]

    def __init__(self, ctx: ContextQuestionAnswer, word_to_idx: Dict[str, int]) -> None:
        self.encoding = np.array([word_to_idx[tk] for tk in ctx.tokens])
        self.qas = [EncodedQuestionAnswer(qa, word_to_idx) for qa in ctx.qas]


class EncodedSample():
    """
    Stores a single model sample (context, question, answers)
    """
    question: Any  # numpy array
    context: Any  # numpy array
    has_answer: bool
    answer_spans: Any  # numpy array

    def __init__(self, ctx_encoding: Any, qa: EncodedQuestionAnswer) -> None:
        self.context = ctx_encoding
        self.question = qa.encoding
        self.has_answers = bool(qa.answers)
        if self.has_answers:
            self.answer_spans = np.empty((len(qa.answers), 2), np.int32)
            for i, answer in enumerate(qa.answers):
                self.answer_spans[i, 0] = answer.span_start
                self.answer_spans[i, 1] = answer.span_end
        else:
            self.answer_spans = np.zeros((1, 2), np.int32)


class SampleBatch():
    """
    Stores a batch of samples
    """
    questions: Any  # numpy array
    contexts: Any  # numpy array
    has_answers: bool
    answer_spans: Any  # numpy array

    def __init__(self, samples: List[EncodedSample]) -> None:
        self.questions = np.stack([sample.question for sample in samples], axis=1)
        self.contexts = np.stack([sample.context for sample in samples], axis=1)
        self.has_answers = np.stack([sample.has_answer for sample in samples], axis=0)
        self.answer_spans = np.stack([sample.answer_spans for sample in samples], axis=0)
