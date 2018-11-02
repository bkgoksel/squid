"""
Integration tests that make sure the JSON -> QABatch pipeline work as expected
"""
import json
from tempfile import NamedTemporaryFile
from typing import Dict, Tuple, Any

import torch as t
from torch.utils.data import DataLoader

from model.batcher import get_collator
from model.corpus import QADataset, TrainDataset, EvalDataset
from model.tokenizer import NltkTokenizer
from model.processor import TextProcessor
from model.vw import WordVectors

SAMPLE_TRAIN_DATASET = {
    "version": "1.0",
    "data": [
        {
            "paragraphs": [
                {
                    "context": "a bb",
                    "qas": [
                        {
                            "question": "a a a",
                            "id": "0",
                            "answers": [{"text": "a", "answer_start": 0}],
                        },
                        {
                            "question": "b b",
                            "id": "1",
                            "answers": [
                                {"text": "bb", "answer_start": 2},
                                {"text": "bb", "answer_start": 2},
                            ],
                        },
                        {"question": "c", "id": "2", "answers": []},
                    ],
                },
                {
                    "context": "a b ccc",
                    "qas": [
                        {
                            "question": "d d dd?",
                            "id": "3",
                            "answers": [{"text": "ccc", "answer_start": 4}],
                        }
                    ],
                },
            ]
        }
    ],
}

TRAIN_QID_TO_Q_WORDS = {
    "0": t.LongTensor([2, 2, 2]),
    "1": t.LongTensor([1, 1]),
    "2": t.LongTensor([1]),
    "3": t.LongTensor([1, 1, 1]),
}

TRAIN_QID_TO_Q_LENS = {"0": 3, "1": 2, "2": 1, "3": 3}

TRAIN_QID_TO_CTX_WORDS = {
    "0": t.LongTensor([2, 3]),
    "1": t.LongTensor([2, 3]),
    "2": t.LongTensor([2, 3]),
    "3": t.LongTensor([2, 1, 3]),
}

TRAIN_QID_TO_CTX_LENS = {"0": 2, "1": 2, "2": 2, "3": 3}

TRAIN_QID_TO_ANS_STARTS = {
    "0": t.LongTensor([1, 0]),
    "1": t.LongTensor([0, 1]),
    "2": t.LongTensor([0, 0]),
    "3": t.LongTensor([0, 0, 1]),
}

TRAIN_QID_TO_ANS_ENDS = {
    "0": t.LongTensor([1, 0]),
    "1": t.LongTensor([0, 1]),
    "2": t.LongTensor([0, 0]),
    "3": t.LongTensor([0, 0, 1]),
}

SAMPLE_DEV_DATASET = {
    "version": "1.0",
    "data": [
        {
            "paragraphs": [
                {
                    "context": "a bd de",
                    "qas": [
                        {
                            "question": "d d dd?",
                            "id": "1",
                            "answers": [{"text": "dd", "answer_start": 4}],
                        }
                    ],
                }
            ]
        }
    ],
}

SAMPLE_WORD_VECTORS = """
a 0.1 0.1 0.1
bb 0.2 0.2 0.2
ccc 0.3 0.3 0.3
"""


def prepare_files():
    train_file = NamedTemporaryFile(delete=False)
    json.dump(SAMPLE_TRAIN_DATASET, train_file)
    train_file.flush()
    dev_file = NamedTemporaryFile(delete=False)
    json.dump(SAMPLE_DEV_DATASET, dev_file)
    dev_file.flush()
    vectors_file = NamedTemporaryFile(delete=False)
    vectors_file.write(SAMPLE_WORD_VECTORS)
    vectors_file.flush()
    return train_file, dev_file, vectors_file


def check_dataset(
    dataset_json, dataset_obj, loader, tokenizer, processor, multi_answer
):
    for batch in loader:
        question_ids = batch.question_ids
        for idx, qid in enumerate(question_ids):
            assertTensorEqual(batch.question_words[idx], TRAIN_QID_TO_Q_WORDS[qid])
            assertTensorEqual(batch.question_chars[idx], TRAIN_QID_TO_Q_CHARS[qid])
            assertTensorEqual(batch.question_lens[idx], TRAIN_QID_TO_Q_LENS[qid])
            assertTensorEqual(batch.question_mask[idx], TRAIN_QID_TO_Q_MASK[qid])
            # assertTensorEqual(batch.question_len_idxs[idx], TRAIN_QID_TO_Q_LEN_IDXS[qid])
            # assertTensorEqual(batch.question_orig_idxs[idx], TRAIN_QID_TO_Q_ORIG_IDXS[qid])
            assertTensorEqual(batch.context_words[idx], TRAIN_QID_TO_CTX_WORDS[qid])
            assertTensorEqual(batch.context_chars[idx], TRAIN_QID_TO_CTX_CHARS[qid])
            assertTensorEqual(batch.context_lens[idx], TRAIN_QID_TO_CTX_LENS[qid])
            assertTensorEqual(batch.context_mask[idx], TRAIN_QID_TO_CTX_MASK[qid])
            assertTensorEqual(
                batch.answer_span_starts[idx], TRAIN_QID_TO_ANS_STARTS[qid]
            )
            assertTensorEqual(batch.answer_span_ends[idx], TRAIN_QID_TO_ANS_ENDS[qid])


def main():
    train_file, dev_file, vectors_file = prepare_files()
    tokenizer = NltkTokenizer()
    processor = TextProcessor({"lowercase": True})
    multi_answer = False
    vectors = WordVectors.load_vectors(vectors_file.name)
    train_dataset = TrainDataset.load_dataset(
        train_file.name, vectors, tokenizer, processor, multi_answer
    )
    dev_dataset = EvalDataset.load_dataset(
        dev_file.name,
        train_dataset.token_mapping,
        train_dataset.char_mapping,
        tokenizer,
        processor,
    )
    train_loader = DataLoader(train_dataset, 16, collate_fn=get_collator(0, 0))
    check_dataset(
        SAMPLE_TRAIN_DATASET,
        train_dataset,
        train_loader,
        tokenizer,
        processor,
        multi_answer,
    )
