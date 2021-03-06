"""
Module for testing dataset representations
"""

import json
import tempfile
from typing import List
import unittest
from unittest.mock import Mock

from model.text_processor import TextProcessor
from model.tokenizer import Tokenizer, Token
from model.qa import Answer, QuestionAnswer, ContextQuestionAnswer, QuestionId

from model.corpus import (
    Corpus,
    CorpusStats,
    EncodedCorpus,
    SampleCorpus,
    QADataset,
    TrainDataset,
    EvalDataset,
)


class RawCorpusTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempfile = tempfile.NamedTemporaryFile(mode="w")

        def split_tokenize(txt: str) -> List[Token]:
            toks = txt.split()
            starts = [3 * start for start in range(len(toks))]
            ends = [(3 * end - 1) for end in range(1, len(toks) + 1)]
            return [
                Token(word=tok[0], span=(tok[1], tok[2]))
                for tok in zip(toks, starts, ends)
            ]

        self.tokenizer = Mock(Tokenizer)
        self.tokenizer.tokenize.side_effect = lambda txt: split_tokenize(txt)
        self.processor = Mock(TextProcessor)
        self.processor.process.side_effect = lambda txt: txt

    def tearDown(self) -> None:
        self.tempfile.close()

    def test_simple_single_question_answer(self) -> None:
        """
        Test that read_context_qas(data_file, tokenizer)
        reads a sample dataset correctly
        TODO:
            - Mock Answer, QuestionAnswer and ContextQuestionAnswer
                classes to make sure they're all created with correct
                parameters
        """
        input_dict = {
            "data": [
                {
                    "paragraphs": [
                        {
                            "context": "c0 c1 c2.c3 c4'c5",
                            "qas": [
                                {
                                    "answers": [{"answer_start": 0, "text": "c0 c1"}],
                                    "id": "0x0001",
                                    "question": "q00 q01 q02 q03?",
                                }
                            ],
                        }
                    ]
                }
            ]
        }
        answer: Answer = Answer("c0 c1", 0, self.tokenizer, self.processor)
        qa: QuestionAnswer = QuestionAnswer(
            QuestionId("0x0001"),
            "q00 q01 q02 q03?",
            set([answer]),
            self.tokenizer,
            self.processor,
        )
        cqa: ContextQuestionAnswer = ContextQuestionAnswer(
            "c0 c1 c2.c3 c4'c5", [qa], self.tokenizer, self.processor
        )
        json.dump(input_dict, self.tempfile)
        self.tempfile.flush()
        cqas: List[ContextQuestionAnswer] = Corpus.read_context_qas(
            self.tempfile.name, self.tokenizer, self.processor, False
        )
        self.assertEqual(cqas, [cqa])

    def test_no_answer(self) -> None:
        input_dict = {
            "data": [
                {
                    "paragraphs": [
                        {
                            "context": "c0 c1 c2.c3 c4'c5",
                            "qas": [
                                {
                                    "answers": [],
                                    "id": "0x0001",
                                    "question": "q00 q01 q02 q03?",
                                }
                            ],
                        }
                    ]
                }
            ]
        }
        json.dump(input_dict, self.tempfile)
        self.tempfile.flush()
        cqas: List[ContextQuestionAnswer] = Corpus.read_context_qas(
            self.tempfile.name, self.tokenizer, self.processor, False
        )
        self.assertEqual(len(cqas), 1)
        self.assertEqual(len(cqas[0].qas), 1)
        self.assertEqual(len(cqas[0].qas[0].answers), 0)

    def test_multiple_questions(self) -> None:
        pass

    def test_multiple_answers(self) -> None:
        pass

    def test_multiple_docs(self) -> None:
        pass

    def test_duplicate_answers(self) -> None:
        pass

    def test_compute_vocab(self) -> None:
        pass

    def test_compute_stats(self) -> None:
        pass

    def test_get_single_answer_text(self) -> None:
        pass

    def test_get_answer_texts(self) -> None:
        """
        Mock the single version to make sure it's called once per question
        """
        pass


class EncodedCorpusTestCase(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_encode(self) -> None:
        pass


class SampleCorpusTestCase(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_make_samples(self) -> None:
        pass


class QADatasetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_len(self) -> None:
        """
        Test that it reports its correct length
        """
        pass

    def test_get_answer_texts(self) -> None:
        """
        Test that it calls corpus' method correctly
        """
        pass


class TrainDatasetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        pass


class EvalDatasetTestCase(unittest.TestCase):
    def setUp(self) -> None:
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
