"""
Module that handles batching logic
"""

from typing import List, Iterable, Tuple, NamedTuple
import numpy as np
import torch as t
from torch.nn.utils.rnn import pad_sequence
from qa import EncodedSample

QABatch = NamedTuple('QABatch', [
    ('questions', t.LongTensor),
    ('question_lens', t.LongTensor),
    ('contexts', t.LongTensor),
    ('context_lens', t.LongTensor),
    ('has_answers', t.LongTensor),
    ('answer_spans', List[t.LongTensor])
])

def collate_batch(batch: List[EncodedSample]) -> QABatch:
    """
    Takes a list of EncodedSample objects and creates a PyTorch batch
    :param batch: List[EncodedSample] QA samples
    :returns: a QABatch
    """

    batch_size = len(batch)
    has_answers = t.LongTensor([sample.has_answer for sample in batch])
    answer_spans = [t.LongTensor(sample.answer_spans) for sample in batch]

    questions = [sample.question for sample in batch]
    contexts = [sample.context for sample in batch]

    questions, question_lens = pad_and_sort(questions)
    contexts, context_lens = pad_and_sort(contexts)

    return QABatch(questions=questions,
                   question_lens=question_lens,
                   contexts=contexts,
                   context_lens=context_lens,
                   has_answers=has_answers,
                   answer_spans=answer_spans)

def pad_and_sort(seq: List[Iterable]) -> Tuple[t.LongTensor, t.LongTensor]:
    """
    Pads a list of sequences with 0's to make them all the same
    length as the longest sequence
    :param seq: A list of sequences
    :returns: Tuple[t.LongTensor, t.LongTensor]
        All sequences padded to maximum length and sorted by length
        and their respective unpadded lengths
    """
    lens = t.LongTensor([el.shape[0] for el in seq])
    lens, idxs = lens.sort(0, descending=True)
    seq = np.array(seq)
    seq = seq[idxs]
    seq = [t.LongTensor(el) for el in seq]
    batch = pad_sequence(seq, batch_first=True)
    return batch, lens
