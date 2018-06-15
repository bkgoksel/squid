"""
Module for testing dataset representations
"""
import unittest
from unittest.mock import Mock, MagicMock
import numpy as np
import torch as t

from typing import List
from model.batcher import (QABatch,
                           collate_batch,
                           pad_and_sort)
from model.qa import (Answer,
                      QuestionId,
                      QuestionAnswer,
                      EncodedQuestionAnswer,
                      EncodedContextQuestionAnswer,
                      EncodedSample)
from model.tokenizer import Tokenizer, Token
from model.wv import WordVectors


class BatcherTestCase(unittest.TestCase):
    def setUp(self):
        token_id_mapping = dict(map(reversed, enumerate(['c0', 'c1', 'c2', 'c3', 'c4'])))

        def split_tokenize(txt: str):
            toks = txt.split()
            starts = [3 * start for start in range(len(toks))]
            ends = [(3 * end - 1) for end in range(1, len(toks) + 1)]
            return [Token(word=tok[0], span=(tok[1], tok[2])) for tok in zip(toks, starts, ends)]

        self.tokenizer = Mock(Tokenizer)
        self.tokenizer.tokenize.side_effect = lambda txt: split_tokenize(txt)
        self.vectors = MagicMock(WordVectors)
        self.vectors.__getitem__.side_effect = lambda tok: token_id_mapping[tok.word]

    def make_sample(self, context_text: str, answers: List[Answer], question_id: str, question_text: str) -> EncodedSample:
        context_tokens = self.tokenizer.tokenize(context_text)
        context_encoding = np.array([self.vectors[tok] for tok in context_tokens])

        question_obj: QuestionAnswer = QuestionAnswer(question_id, question_text, answers, self.tokenizer)

        encoded_qa_obj: EncodedQuestionAnswer = EncodedQuestionAnswer(
            question_obj, self.vectors, context_tokens)
        encoded_sample = EncodedSample(context_encoding, encoded_qa_obj)
        return encoded_sample

    def test_collate_batch_simple(self):
        """
        Tests that collate batch includes all question ids in original order
        """
        samples: List[EncodedSample] = [
            self.make_sample('c0', [], 'q0', 'c1'),
            self.make_sample('c0', [], 'q1', 'c1'),
            self.make_sample('c0', [], 'q2', 'c1'),
        ]
        batch: QABatch = collate_batch(samples)
        self.assertEqual(batch.question_ids, [QuestionId('q0'),
                                              QuestionId('q1'),
                                              QuestionId('q2')])

    def test_collate_batch_q_len_sorting(self):
        """
        Tests that question lengths in batch are sorted, and
        len_idxs indices map to correct indices
        """
        samples: List[EncodedSample] = [
            self.make_sample('c0', [], 'q0', 'c1'),
            self.make_sample('c0', [], 'q1', 'c1 c2 c3'),
            self.make_sample('c0', [], 'q2', 'c1 c2'),
        ]

        batch: QABatch = collate_batch(samples)
        self.assertEqual(batch.question_lens, t.LongTensor([3, 2, 1]))
        self.assertEqual(batch.question_len_idxs, t.LongTensor([1, 2, 0]))

    def test_collate_batch_ctx_len_sorting(self):
        """
        Tests that question lengths in batch are sorted, and
        len_idxs indices map to correct indices
        """
        samples: List[EncodedSample] = [
            self.make_sample('c0 c1 c2', [], 'q0', 'c1'),
            self.make_sample('c0', [], 'q1', 'c1'),
            self.make_sample('c0 c1', [], 'q2', 'c1'),
        ]

        batch: QABatch = collate_batch(samples)
        self.assertEqual(batch.context_lens, t.LongTensor([3, 2, 1]))
        self.assertEqual(batch.context_len_idxs, t.LongTensor([0, 2, 1]))

    def test_collate_batch_idxs(self):
        """
        Tests that (q|c)[len_sorted][orig_idxs] == q|c
        """
        samples: List[EncodedSample] = [
            self.make_sample('c0 c1 c2', [], 'q0', 'c0 c1'),
            self.make_sample('c0', [], 'q1', 'c0 c1 c2'),
            self.make_sample('c0 c1', [], 'q2', 'c1'),
        ]

        batch: QABatch = collate_batch(samples)

        self.assertEqual(batch.context_len_idxs, t.LongTensor([0, 2, 1]))
        self.assertEqual(batch.question_len_idxs, t.LongTensor([1, 0, 2]))
        self.assertEqual(batch.questions, (batch.questions[batch.question_len_idxs])[batch.question_orid_idxs])
        self.assertEqual(batch.contexts, (batch.contexts[batch.context_len_idxs])[batch.context_orid_idxs])

    def test_pad_and_sort(self):
        pass
