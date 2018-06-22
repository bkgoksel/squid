"""
Module for testing data encoding
"""
import unittest
from unittest.mock import Mock, MagicMock

import numpy as np

from typing import List
from model.text_processor import TextProcessor
from model.tokenizer import Tokenizer, Token
from model.qa import (Answer,
                      QuestionAnswer,
                      ContextQuestionAnswer,
                      EncodedAnswer,
                      EncodedQuestionAnswer,
                      EncodedContextQuestionAnswer,
                      EncodedSample)
from model.wv import WordVectors


class EncodingTestCase(unittest.TestCase):
    def setUp(self):
        self.processor = Mock(TextProcessor)
        self.processor.side_effect = lambda txt: txt.lower()

        def split_tokenize(txt: str):
            toks = txt.split()
            starts = [3 * start for start in range(len(toks))]
            ends = [(3 * end - 1) for end in range(1, len(toks) + 1)]
            return [Token(word=tok[0], span=(tok[1], tok[2])) for tok in zip(toks, starts, ends)]

        self.tokenizer = Mock(Tokenizer)
        self.tokenizer.tokenize.side_effect = lambda txt: split_tokenize(txt)
        self.vectors = MagicMock(WordVectors)

    def test_answer_init(self):
        """
        Tests that Answer object are initialized correctly
        """
        answer_text = 'a0 a1 a2'
        answer_start = 0
        answer_obj: Answer = Answer(answer_text, answer_start, self.tokenizer, self.processor)
        self.assertEqual(answer_obj.text, answer_text)
        self.assertEqual(answer_obj.span_start, answer_start)
        self.assertEqual(answer_obj.span_end, answer_start + len(answer_text))

    def test_qa_init(self):
        """
        Tests that QuestionAnswer objects are initialized successfully
        """
        answers = [Answer('a00 a01', 0, self.tokenizer, self.processor), Answer('a10', 2, self.tokenizer, self.processor)]
        question_id = 'qid_0'
        question_text = 'q0 q1 q2 q3'
        question_tokens = self.tokenizer.tokenize(question_text)

        question_obj: QuestionAnswer = QuestionAnswer(question_id, question_text, answers, self.tokenizer, self.processor)

        self.assertEqual(question_obj.question_id, question_id)
        self.assertEqual(question_obj.text, question_text)
        self.assertEqual(question_obj.answers, answers)
        self.assertEqual(question_obj.tokens, question_tokens)

    def test_context_qa_init(self):
        """
        Tests that ContextQuestionAnswer objects are initialized properly
        """
        qas: List[QuestionAnswer] = [QuestionAnswer('qa_%d' % i,
                                                    'question %d' % i,
                                                    [Answer('c%d' % i, 3 * i, self.tokenizer, self.processor)],
                                                    self.tokenizer,
                                                    self.processor)
                                     for i in range(2)]
        context_text = 'c0 c1 c2'
        context_tokens = self.tokenizer.tokenize(context_text)
        cqa_obj: ContextQuestionAnswer = ContextQuestionAnswer(
            context_text, qas, self.tokenizer, self.processor)
        self.assertEqual(qas, cqa_obj.qas)
        self.assertEqual(context_text, cqa_obj.text)
        self.assertEqual(context_tokens, cqa_obj.tokens)

    def test_answer_encoding(self):
        """
        Tests that the EncodedAnswer class maps answer spans to post-tokenization
        token indices correctly
        """
        context_text: str = "c0 c1 c2 a0 a1 c5 c6"
        context_tokens: List[Token] = self.tokenizer.tokenize(context_text)
        answer: Answer = Answer('a0 a1', 9, self.tokenizer, self.processor)
        encoded_answer: EncodedAnswer = EncodedAnswer(answer, context_tokens)
        self.assertEqual(encoded_answer.span_start, 3)
        self.assertEqual(encoded_answer.span_end, 4)

    def test_encoded_qa(self):
        """
        Tests that EncodedQuestionAnswer objects are initialized correctly
        This takes a QuestionAnswer objects and builds a
        """
        token_id_mapping = {'c0': 0, 'c1': 1, 'c2': 2, 'a0': 3, 'a1': 4}

        context_text: str = "c0 c1 c2 a0 a1"
        context_tokens = self.tokenizer.tokenize(context_text)

        answers = [Answer('a0 a1', 9, self.tokenizer, self.processor), Answer('a0', 9, self.tokenizer, self.processor)]
        encoded_answers = [EncodedAnswer(ans, context_tokens) for ans in answers]

        question_id = 'qid_0'
        question_text = 'c0 c1 c2'
        question_tokens = self.tokenizer.tokenize(question_text)
        question_encoding = np.array([token_id_mapping[tok.word] for tok in question_tokens])

        question_obj: QuestionAnswer = QuestionAnswer(question_id, question_text, answers, self.tokenizer, self.processor)

        self.vectors.__getitem__.side_effect = lambda tok: token_id_mapping[tok]

        encoded_qa_obj: EncodedQuestionAnswer = EncodedQuestionAnswer(
            question_obj, self.vectors, context_tokens)

        self.assertEqual(encoded_qa_obj.question_id, question_id)
        self.assertTrue(np.allclose(encoded_qa_obj.encoding, question_encoding))
        self.assertEqual(encoded_qa_obj.answers, encoded_answers)

    def test_encoded_cqa(self):
        """
        Tests that the EncodedContextQuestionAnswer object encodes the context
        correctly
        """
        token_id_mapping = {'c0': 0, 'c1': 1, 'c2': 2, 'a0': 3, 'a1': 4}

        context_text: str = "c0 c1 c2 a0 a1"
        context_tokens = self.tokenizer.tokenize(context_text)
        context_encoding = np.array([token_id_mapping[tok.word] for tok in context_tokens])

        answers = [Answer('a0 a1', 9, self.tokenizer, self.processor), Answer('a0', 9, self.tokenizer, self.processor)]

        question_id = 'qid_0'
        question_text = 'c0 c1 c2'

        question_obj: QuestionAnswer = QuestionAnswer(question_id, question_text, answers, self.tokenizer, self.processor)
        cqa_obj: ContextQuestionAnswer = ContextQuestionAnswer(context_text,
                                                               [question_obj],
                                                               self.tokenizer,
                                                               self.processor)

        self.vectors.__getitem__.side_effect = lambda tok: token_id_mapping[tok]

        encoded_cqa_obj: EncodedContextQuestionAnswer = EncodedContextQuestionAnswer(
            cqa_obj, self.vectors)
        self.assertTrue(np.allclose(encoded_cqa_obj.encoding, context_encoding))

    def test_encoded_sample(self):
        """
        Tests that the EncodedSample object builds the question, context
        and answer_span arrays correctly given a context_encoding and
        EncodedQuestionAnswer objects
        """
        token_id_mapping = {'c0': 0, 'c1': 1, 'c2': 2, 'a0': 3, 'a1': 4}

        context_text: str = "c0 c1 c2 a0 a1"
        context_tokens = self.tokenizer.tokenize(context_text)
        context_encoding = np.array([token_id_mapping[tok.word] for tok in context_tokens])

        answers = [Answer('a0 a1', 9, self.tokenizer, self.processor), Answer('a0', 9, self.tokenizer, self.processor)]
        answer_starts = np.array([0, 0, 0, 1, 0])
        answer_ends = np.array([0, 0, 0, 1, 1])

        question_id = 'qid_0'
        question_text = 'c0 c1 c2'
        question_tokens = self.tokenizer.tokenize(question_text)
        question_encoding = np.array([token_id_mapping[tok.word] for tok in question_tokens])

        question_obj: QuestionAnswer = QuestionAnswer(question_id, question_text, answers, self.tokenizer, self.processor)

        self.vectors.__getitem__.side_effect = lambda tok: token_id_mapping[tok]

        encoded_qa_obj: EncodedQuestionAnswer = EncodedQuestionAnswer(
            question_obj, self.vectors, context_tokens)
        encoded_sample = EncodedSample(context_encoding, encoded_qa_obj)

        self.assertEqual(encoded_sample.question_id, question_id)
        self.assertTrue(np.allclose(encoded_sample.question, question_encoding))
        self.assertTrue(np.allclose(encoded_sample.context, context_encoding))
        self.assertTrue(encoded_sample.has_answer)
        self.assertTrue(np.allclose(encoded_sample.span_starts, answer_starts))
        self.assertTrue(np.allclose(encoded_sample.span_ends, answer_ends))
