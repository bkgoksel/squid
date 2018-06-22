""" Module for testing dataset representations
"""
import unittest
from unittest.mock import Mock
import json
import tempfile
from typing import List

from model.text_processor import TextProcessor
from model.tokenizer import Tokenizer, Token
from model.qa import (Answer,
                      QuestionAnswer,
                      ContextQuestionAnswer,
                      QuestionId)

from model.corpus import (Corpus,
                          CorpusStats,
                          EncodedCorpus,
                          SampleCorpus,
                          QADataset)


class RawCorpusTestCase(unittest.TestCase):
    def setUp(self):
        self.tempfile = tempfile.NamedTemporaryFile()

        def split_tokenize(txt: str):
            toks = txt.split()
            starts = [3 * start for start in range(len(toks))]
            ends = [(3 * end - 1) for end in range(1, len(toks) + 1)]
            return [Token(word=tok[0], span=(tok[1], tok[2])) for tok in zip(toks, starts, ends)]

        self.tokenizer = Mock(Tokenizer)
        self.tokenizer.tokenize.side_effect = lambda txt: split_tokenize(txt)
        self.processor = Mock(TextProcessor)
        self.processor.process.side_effect = lambda txt: txt

    def tearDown(self):
        self.tempfile.close()

    def test_simple_single_question_answer(self):
        """
        Test that read_context_qas(data_file, tokenizer)
        reads a sample dataset correctly
        TODO:
            - Use small simple JSON inputs for different edge cases
            - Mock Answer, QuestionAnswer and ContextQuestionAnswer
                classes to make sure they're all created with correct
                parameters
        """
        input_dict = {
            'data': [
                {
                    'paragraphs': [
                        {
                            'context': 'c0 c1 c2.c3 c4\'c5',
                            'qas': [
                                {
                                    'answers': [
                                        {
                                            'answer_start': 0,
                                            'text': 'c0 c1'
                                        }
                                    ],
                                    'id': '0x0001',
                                    'question': 'q00 q01 q02 q03?'
                                },
                            ]
                        }
                    ]
                },
            ]
        }
        answer: Answer = Answer('c0 c1', 0)
        qa: QuestionAnswer = QuestionAnswer(QuestionId('0x0001'), 'q00 q01 q02 q03?', [answer], self.tokenizer, self.processor)
        cqa: ContextQuestionAnswer = ContextQuestionAnswer('c0 c1 c2.c3 c4\'c5', [qa], self.tokenizer, self.processor)
        json.dump(input_dict, self.tempfile)
        cqas: List[ContextQuestionAnswer] = Corpus.read_context_qas(self.tempfile.name, self.tokenizer, self.processor)
        self.assertEqual(cqas, [cqa])

    def test_no_answer(self):
        pass

    def test_multiple_questions(self):
        pass

    def test_multiple_answers(self):
        pass

    def test_multiple_docs(self):
        pass

    def test_duplicate_answers(self):
        pass

    def test_compute_vocab(self):
        pass

    def test_compute_stats(self):
        pass

    def test_get_single_answer_text(self):
        pass

    def test_get_answer_texts(self):
        """
        Mock the single version to make sure it's called once per question
        """
        pass


class EncodedCorpusTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_encode(self):
        pass


class SampleCorpusTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_make_samples(self):
        pass


class QADatasetTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_len(self):
        """
        Test that it reports its correct length
        """
        pass

    def test_get_answer_texts(self):
        """
        Test that it calls corpus' method correctly
        """
        pass


"""
    def test_corpus_init(self):
        Tests that the Corpus object is created
        correctly after reading from JSON file
        input_set = {
                'data': [
                    {
                        'paragraphs': [
                            {
                                'context': 'c0 c1 c2.c3 c4\'c5',
                                'qas': [
                                    {
                                        'answers': [
                                            {
                                                'answer_start': 0,
                                                'text': 'c0 c1'
                                                },
                                            {
                                                'answer_start': 0,
                                                'text': 'c0'
                                                }
                                            ],
                                        'id': '0x0001',
                                        'question': 'q00 q01 q02 q03?'
                                        },
                                    {
                                        'answers': [
                                            {
                                                'answer_start': 6,
                                                'text': 'c2.c3'
                                                }
                                            ],
                                        'id': '0x0002',
                                        'question': 'q10 q11 q12?'
                                        },
                                    {
                                        'answers': [],
                                        'id': '0x0003',
                                        'question': 'q20 q21 q22 q23?'
                                        },
                                    ]
                                }
                            ]
                        },
                    {
                        }
                    ]
                }
        json.dump(input_set, self.tempfile)
        corpus: Corpus = Corpus.from_raw(self.tempfile.name)
"""
