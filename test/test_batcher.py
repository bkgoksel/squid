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
                      EncodedSample)
from model.text_processor import TextProcessor
from model.tokenizer import Tokenizer, Token
from model.wv import WordVectors


class BatcherTestCase(unittest.TestCase):
    def setUp(self):
        vocab = ['c1', 'c2', 'c3', 'c4', 'c00']
        char_vocab = set([char for word in vocab for char in word])
        self.token_id_mapping = dict(map(reversed, enumerate(vocab, 1)))
        self.id_token_mapping = dict(enumerate(vocab, 1))
        self.char_mapping = dict(map(reversed, enumerate(char_vocab, 1)))

        def split_tokenize(txt: str):
            toks = txt.split()
            starts = [3 * start for start in range(len(toks))]
            ends = [(3 * end - 1) for end in range(1, len(toks) + 1)]
            return [Token(word=tok[0], span=(tok[1], tok[2])) for tok in zip(toks, starts, ends)]

        self.tokenizer = Mock(Tokenizer)
        self.tokenizer.tokenize.side_effect = lambda txt: split_tokenize(txt)
        self.processor = Mock(TextProcessor)
        self.processor.process.side_effect = lambda txt: txt
        self.vectors = MagicMock(WordVectors)
        self.vectors.__getitem__.side_effect = lambda tok: self.token_id_mapping[tok]

    def make_sample(self, context_text: str, answers: List[Answer], question_id: str, question_text: str) -> EncodedSample:
        context_tokens = self.tokenizer.tokenize(context_text)
        context_word_encoding = np.array([self.vectors[tok.word] for tok in context_tokens])
        context_char_encoding = [np.array([self.char_mapping[char] for char in tok.word]) for tok in context_tokens]

        question_obj: QuestionAnswer = QuestionAnswer(question_id, question_text, answers, self.tokenizer, self.processor)

        encoded_qa_obj: EncodedQuestionAnswer = EncodedQuestionAnswer(
            question_obj, self.vectors, self.char_mapping, context_tokens)
        encoded_sample = EncodedSample(context_word_encoding, context_char_encoding, encoded_qa_obj)
        return encoded_sample

    def test_pad_and_sort_single_seq(self):
        seq = [np.array([1])]
        batch, orig_idxs, length_idxs, lengths = pad_and_sort(seq)
        self.assertTrue(np.all(batch.numpy() == np.array([1])),
                        'Batch: {0}, expected: {1}'.format(batch, [1]))
        self.assertTrue(np.all(orig_idxs.numpy() == np.array([0])),
                        'Length idxs: {0}, expected: {1}'.format(length_idxs, [0]))
        self.assertTrue(np.all(length_idxs.numpy() == np.array([0])),
                        'Orig idxs: {0}, expected: {1}'.format(orig_idxs, [0]))
        self.assertTrue(np.all(lengths.numpy() == np.array([1])),
                        'Lengths: {0}, expected: {1}'.format(lengths, [1]))

    def test_pad_and_sort_regular(self):
        seqs = [
            np.array([1, 1, 1]),
            np.array([2]),
            np.array([3, 3]),
        ]
        batch, orig_idxs, length_idxs, lengths = pad_and_sort(seqs)
        self.assertTrue(np.all(batch.numpy() ==
                               np.stack([[1, 1, 1], [3, 3, 0], [2, 0, 0]])),
                        'Batch: {0}, expected: {1}'.format(batch, [[1, 1, 1], [3, 3, 0], [2, 0, 0]]))
        self.assertTrue(np.all(length_idxs.numpy() ==
                               np.array([0, 2, 1])),
                        'Length idxs: {0}, expected: {1}'.format(length_idxs, [0, 2, 1]))
        self.assertTrue(np.all(orig_idxs.numpy() ==
                               np.array([0, 2, 1])),
                        'Orig idxs: {0}, expected: {1}'.format(orig_idxs, [0, 2, 1]))
        self.assertTrue(np.all(lengths.numpy() ==
                               np.array([3, 2, 1])),
                        'Lengths: {0}, expected: {1}'.format(lengths, [3, 2, 1]))
        self.assertTrue(np.all(batch[orig_idxs].numpy() ==
                               np.stack([[1, 1, 1], [2, 0, 0], [3, 3, 0]])),
                        'After orig idxs: {0}, expected: {1}'.format(batch[orig_idxs], [[1, 1, 1], [2, 0, 0], [3, 3, 0]]))
        self.assertTrue(np.all(batch[orig_idxs][length_idxs].numpy() ==
                               batch),
                        'After orig idxs and len idxs: {0}, expected: {1}'.format(batch[orig_idxs][length_idxs], batch))

    def test_pad_and_sort_edge_case(self):
        seqs = [
            np.array([1]),
            np.array([1, 2, 3]),
            np.array([1, 2]),
        ]
        batch, orig_idxs, length_idxs, lengths = pad_and_sort(seqs)
        self.assertTrue(np.all(batch.numpy() ==
                               np.stack([[1, 2, 3], [1, 2, 0], [1, 0, 0]])),
                        'Batch: {0}, expected: {1}'.format(batch, [[1, 2, 3], [1, 2, 0], [1, 0, 0]]))
        self.assertTrue(np.all(length_idxs.numpy() ==
                               np.array([1, 2, 0])),
                        'Length idxs: {0}, expected: {1}'.format(length_idxs, [1, 2, 0]))
        self.assertTrue(np.all(orig_idxs.numpy() ==
                               np.array([2, 0, 1])),
                        'Orig idxs: {0}, expected: {1}'.format(orig_idxs, [2, 0, 1]))
        self.assertTrue(np.all(lengths.numpy() ==
                               np.array([3, 2, 1])),
                        'Lengths: {0}, expected: {1}'.format(lengths, [3, 2, 1]))
        self.assertTrue(np.all(batch[orig_idxs].numpy() ==
                               np.stack([[1, 0, 0], [1, 2, 3], [1, 2, 0]])),
                        'After orig idxs: {0}, expected: {1}'.format(batch[orig_idxs], [[1, 0, 0], [1, 2, 3], [1, 2, 0]]))
        self.assertTrue(np.all(batch[orig_idxs][length_idxs].numpy() ==
                               batch),
                        'After orig idxs and len idxs: {0}, expected: {1}'.format(batch[orig_idxs][length_idxs], batch))

    def test_collate_batch_simple_words(self):
        """
        Tests that collate batch includes all question ids in original order
        """
        samples: List[EncodedSample] = [
            self.make_sample('c1', [], 'q0', 'c1'),
            self.make_sample('c1', [], 'q1', 'c2'),
            self.make_sample('c1', [], 'q2', 'c3'),
        ]
        batch: QABatch = collate_batch(samples)
        self.assertEqual(batch.question_ids, [QuestionId('q0'),
                                              QuestionId('q1'),
                                              QuestionId('q2')])
        self.assertTrue(np.all(batch.question_words.numpy() ==
                               np.stack([[1], [2], [3]])),
                        'Batch questions: {0} Expected: {1}'.format(batch.question_words, [[1], [2], [3]]))

    def test_collate_batch_q_len_sorting(self):
        """
        Tests that question lengths in batch are sorted, and
        len_idxs indices map to correct indices
        """
        samples: List[EncodedSample] = [
            self.make_sample('c1', [], 'q0', 'c1'),
            self.make_sample('c1', [], 'q1', 'c1 c2 c3'),
            self.make_sample('c1', [], 'q2', 'c1 c2'),
        ]

        batch: QABatch = collate_batch(samples)
        self.assertTrue(np.all(batch.question_words.numpy() ==
                               np.stack([[1, 0, 0], [1, 2, 3], [1, 2, 0]])),
                        'Batch questions: {0} Expected: {1}'.format(batch.question_words, [[1, 0, 0], [1, 2, 3], [1, 2, 0]]))
        self.assertTrue(np.allclose(batch.question_lens, [3, 2, 1]),
                        'Question lens: {0} expected: {1}'.format(batch.question_lens, [3, 2, 1]))
        self.assertTrue(np.allclose(batch.question_len_idxs, [1, 2, 0]),
                        'Question len idxs: {0} expected: {1}'.format(batch.question_len_idxs, [1, 2, 0]))

    def test_collate_batch_ctx_len_sorting(self):
        """
        Tests that question lengths in batch are sorted, and
        len_idxs indices map to correct indices
        """
        samples: List[EncodedSample] = [
            self.make_sample('c1 c2 c3', [], 'q0', 'c1'),
            self.make_sample('c1', [], 'q1', 'c1'),
            self.make_sample('c1 c2', [], 'q2', 'c1'),
        ]

        batch: QABatch = collate_batch(samples)
        self.assertTrue(np.allclose(batch.context_lens, [3, 2, 1]))
        self.assertTrue(np.allclose(batch.context_len_idxs, [0, 2, 1]))

    def test_collate_batch_idxs(self):
        """
        Tests that (q|c)[len_sorted][orig_idxs] == q|c
        """
        samples: List[EncodedSample] = [
            self.make_sample('c1 c2 c3', [], 'q0', 'c1 c2'),
            self.make_sample('c1', [], 'q1', 'c1 c2 c3'),
            self.make_sample('c1 c2', [], 'q2', 'c1'),
        ]

        batch: QABatch = collate_batch(samples)

        self.assertTrue(np.allclose(batch.context_len_idxs, [0, 2, 1]))
        self.assertTrue(np.allclose(batch.question_len_idxs, [1, 0, 2]))
        self.assertTrue(np.allclose(batch.question_words, (batch.question_words[batch.question_len_idxs])[batch.question_orig_idxs]))
        self.assertTrue(np.allclose(batch.context_words, (batch.context_words[batch.context_len_idxs])[batch.context_orig_idxs]))

    def test_collate_batch_question_chars(self):
        """
        Tests that collate batch includes all question word characters that are parsed correctly
        """
        samples: List[EncodedSample] = [
            self.make_sample('c1', [], 'q0', 'c1'),
            self.make_sample('c2', [], 'q1', 'c2'),
            self.make_sample('c3', [], 'q2', 'c3'),
        ]
        batch: QABatch = collate_batch(samples)
        self.assertEqual(len(batch.question_chars), 3)
        for idx, sample in enumerate(batch.question_chars):
            word = self.id_token_mapping[batch.question_words[idx].item()]
            self.assertEqual(sample[0].shape, t.Size([2]))
            self.assertEqual(sample[0][0], self.char_mapping[word[0]])
            self.assertEqual(sample[0][1].item(), self.char_mapping[word[1]])

    def test_collate_batch_context_chars(self):
        """
        Tests that collate batch includes all context word characters that are parsed correctly
        """
        samples: List[EncodedSample] = [
            self.make_sample('c1', [], 'q0', 'c1'),
            self.make_sample('c2', [], 'q1', 'c1'),
            self.make_sample('c3', [], 'q2', 'c1'),
        ]
        batch: QABatch = collate_batch(samples)
        self.assertEqual(len(batch.context_chars), 3)
        for idx, sample in enumerate(batch.context_chars):
            word = self.id_token_mapping[batch.context_words[idx].item()]
            self.assertEqual(sample[0].shape, t.Size([2]))
            self.assertEqual(sample[0][0], self.char_mapping[word[0]])
            self.assertEqual(sample[0][1].item(), self.char_mapping[word[1]])

    def test_collate_batch_different_question_word_lens(self):
        """
        Tests that collate batch deals with questions with different
        question word lengths
        """
        samples: List[EncodedSample] = [
            self.make_sample('c1', [], 'q0', 'c00'),
            self.make_sample('c1', [], 'q1', 'c1'),
        ]
        batch: QABatch = collate_batch(samples)
        self.assertEqual(batch.question_chars.shape, t.Size([2, 1, 3]))

    def test_collate_batch_different_question_word_numbers_and_lens(self):
        """
        Tests that collate batch deals with questions with different
        question word lengths and numbers
        """
        samples: List[EncodedSample] = [
            self.make_sample('c1', [], 'q0', 'c1'),
            self.make_sample('c1', [], 'q1', 'c00'),
            self.make_sample('c1', [], 'q2', 'c1 c00'),
        ]
        batch: QABatch = collate_batch(samples)
        self.assertEqual(batch.question_chars.shape, t.Size([3, 2, 3]))

    def test_collate_batch_different_context_word_lens(self):
        """
        Tests that collate batch deals with questions with different
        context word lengths
        """
        samples: List[EncodedSample] = [
            self.make_sample('c00', [], 'q0', 'c1'),
            self.make_sample('c1', [], 'q1', 'c1'),
        ]
        batch: QABatch = collate_batch(samples)
        self.assertEqual(batch.context_chars.shape, t.Size([2, 1, 3]))

    def test_collate_batch_different_context_word_numbers_and_lens(self):
        """
        Tests that collate batch deals with questions with different
        context word lengths and numbers
        """
        samples: List[EncodedSample] = [
            self.make_sample('c1', [], 'q0', 'c1'),
            self.make_sample('c00', [], 'q1', 'c1'),
            self.make_sample('c1 c00', [], 'q2', 'c1'),
        ]
        batch: QABatch = collate_batch(samples)
        self.assertEqual(batch.context_chars.shape, t.Size([3, 2, 3]))
