"""
Module that handles batching logic
"""

from typing import List, Any, Tuple, NamedTuple
import numpy as np
import torch as t
from torch.nn.utils.rnn import pad_sequence

from modules.masked import mask_sequence
from qa import EncodedSample

QABatch = NamedTuple('QABatch', [
    ('questions', t.LongTensor),
    ('question_lens', t.LongTensor),
    ('question_idxs', t.LongTensor),
    ('question_mask', t.LongTensor),
    ('contexts', t.LongTensor),
    ('context_lens', t.LongTensor),
    ('context_idxs', t.LongTensor),
    ('context_mask', t.LongTensor),
    ('answer_span_starts', t.LongTensor),
    ('answer_span_ends', t.LongTensor)
])


def collate_batch(batch: List[EncodedSample]) -> QABatch:
    """
    Takes a list of EncodedSample objects and creates a PyTorch batch
    :param batch: List[EncodedSample] QA samples
    :returns: a QABatch
    """
    questions = []
    contexts = []
    answer_span_starts = []
    answer_span_ends = []

    for sample in batch:
        questions.append(sample.question)
        contexts.append(sample.context)
        answer_span_starts.append(sample.span_starts)
        answer_span_ends.append(sample.span_ends)

    questions, question_idxs, question_lens = pad_and_sort(questions)
    question_mask = mask_sequence(questions)

    contexts, context_idxs, context_lens = pad_and_sort(contexts)
    context_mask = mask_sequence(contexts)

    answer_span_starts = np.array(answer_span_starts)
    answer_span_ends = np.array(answer_span_ends)

    return QABatch(questions=questions,
                   question_lens=question_lens,
                   question_idxs=question_idxs,
                   question_mask=question_mask,
                   contexts=contexts,
                   context_lens=context_lens,
                   context_idxs=context_idxs,
                   context_mask=context_mask,
                   answer_span_starts=answer_span_starts,
                   answer_span_ends=answer_span_ends)


def pad_and_sort(seq: List[Any]) -> Tuple[t.LongTensor, t.LongTensor, t.LongTensor]:
    """
    Pads a list of sequences with 0's to make them all the same
    length as the longest sequence
    :param seq: A list of sequences
    :returns:
        - Batch of padded sequences
        - Original(pre-sort) indices of elements (meaning batch[orig_idxs] == seq)
        - lengths of sequences
    """
    lens = t.LongTensor([el.shape[0] for el in seq])
    lens, idxs = lens.sort(0, descending=True)
    seq = np.array(seq)
    seq = seq[idxs]
    seq = [t.LongTensor(el) for el in seq]
    batch = pad_sequence(seq, batch_first=True)
    _, orig_idxs = idxs.sort()
    return batch, orig_idxs, lens
