"""
Module that handles batching logic
"""

from typing import List, Any, Tuple, NamedTuple
import numpy as np
import torch as t
from torch.nn.utils.rnn import pad_sequence

from modules.masked import mask_sequence
from qa import EncodedSample, QuestionId

"""
Holds a batch of samples in a form that's easy for the model to use
len_idxs and orig_idxs allow for length-sorted or original orderings
of the respective texts i.e.
questions[question_len_idxs] = length_sorted_questions
length_sorted_questions[question_orig_idxs] = questions

masks, lengths and question_ids come in original ordering
"""
QABatch = NamedTuple('QABatch', [
    ('questions', t.LongTensor),
    ('question_lens', t.LongTensor),
    ('question_len_idxs', t.LongTensor),
    ('question_orig_idxs', t.LongTensor),
    ('question_mask', t.LongTensor),
    ('question_ids', List[QuestionId]),
    ('contexts', t.LongTensor),
    ('context_lens', t.LongTensor),
    ('context_len_idxs', t.LongTensor),
    ('context_orig_idxs', t.LongTensor),
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
    question_ids = []
    contexts = []
    answer_span_starts = []
    answer_span_ends = []

    for sample in batch:
        questions.append(sample.question)
        question_ids.append(sample.question_id)
        contexts.append(sample.context)
        answer_span_starts.append(sample.span_starts)
        answer_span_ends.append(sample.span_ends)

    questions, question_len_idxs, question_orig_idxs, question_lens = pad_and_sort(questions)
    questions = questions[question_orig_idxs]
    question_lens = question_lens[question_orig_idxs]
    question_mask = mask_sequence(questions)

    contexts, context_len_idxs, context_orig_idxs, context_lens = pad_and_sort(contexts)
    contexts = contexts[context_orig_idxs]
    context_lens = context_lens[context_orig_idxs]
    context_mask = mask_sequence(contexts)

    answer_span_starts, _, _, _ = pad_and_sort(answer_span_starts)
    answer_span_starts = answer_span_starts[context_orig_idxs]

    answer_span_ends, _, _, _ = pad_and_sort(answer_span_ends)
    answer_span_ends = answer_span_ends[context_orig_idxs]

    return QABatch(questions=questions,
                   question_lens=question_lens,
                   question_len_idxs=question_len_idxs,
                   question_orig_idxs=question_orig_idxs,
                   question_mask=question_mask,
                   question_ids=question_ids,
                   contexts=contexts,
                   context_lens=context_lens,
                   context_len_idxs=context_len_idxs,
                   context_orig_idxs=context_orig_idxs,
                   context_mask=context_mask,
                   answer_span_starts=answer_span_starts,
                   answer_span_ends=answer_span_ends)


def pad_and_sort(seq: List[Any]) -> Tuple[t.LongTensor, t.LongTensor, t.LongTensor, t.LongTensor]:
    """
    Pads a list of sequences with 0's to make them all the same
    length as the longest sequence
    :param seq: A list of sequences
    :returns:
        - Batch of padded sequences
        - Original-to-length sort indices (meaning seq[length_idxs] == batch)
        - Length-sorted-to-original sort indices (meaning batch[orig_idxs] == seq)
        - lengths of sequences
    """
    lengths = t.LongTensor([el.shape[0] for el in seq])
    lengths, length_idxs = lengths.sort(0, descending=True)
    seq = np.array(seq)
    seq = seq[length_idxs]
    seq = [t.LongTensor(el) for el in seq]
    batch = pad_sequence(seq, batch_first=True)
    _, orig_idxs = length_idxs.sort()
    return batch, orig_idxs, length_idxs, lengths
