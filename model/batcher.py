"""
Module that handles batching logic
"""

from typing import List, Any, Tuple, NamedTuple
import numpy as np
import torch as t
from torch.nn.utils.rnn import pad_sequence

from model.modules.masked import mask_sequence
from model.qa import EncodedSample, QuestionId

"""
Holds a batch of samples in a form that's easy for the model to use
len_idxs and orig_idxs allow for length-sorted or original orderings
of the respective texts i.e.
question_words[question_len_idxs] = length_sorted_questions
length_sorted_questions[question_orig_idxs] = question_words

masks, and question_ids come in original ordering
lengths come sorted
"""
QABatch = NamedTuple('QABatch', [
    ('question_words', t.LongTensor),
    ('question_chars', t.LongTensor),
    ('question_lens', t.LongTensor),
    ('question_len_idxs', t.LongTensor),
    ('question_orig_idxs', t.LongTensor),
    ('question_mask', t.LongTensor),
    ('question_ids', List[QuestionId]),
    ('context_words', t.LongTensor),
    ('context_chars', t.LongTensor),
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
    For chars:
        context_chars[batch, word, char_idx] -> (batch_len, max_ctx_len, max_word_len)
    :param batch: List[EncodedSample] QA samples
    :returns: a QABatch
    """
    # TODO: Do the char encoding in the batch
    question_words = []
    question_chars = []
    question_ids = []
    context_words = []
    context_chars = []
    answer_span_starts = []
    answer_span_ends = []

    for sample in batch:
        question_words.append(sample.question_words)
        question_chars.append(sample.question_chars)
        question_ids.append(sample.question_id)
        context_words.append(sample.context_words)
        context_chars.append(sample.context_chars)
        answer_span_starts.append(sample.span_starts)
        answer_span_ends.append(sample.span_ends)

    question_words, question_orig_idxs, question_len_idxs, question_lens = pad_and_sort(question_words)
    question_words = question_words[question_orig_idxs]
    question_mask = mask_sequence(question_words)

    context_words, context_orig_idxs, context_len_idxs, context_lens = pad_and_sort(context_words)
    context_words = context_words[context_orig_idxs]
    context_mask = mask_sequence(context_words)

    answer_span_starts, _, _, _ = pad_and_sort(answer_span_starts)
    answer_span_starts = answer_span_starts[context_orig_idxs]

    answer_span_ends, _, _, _ = pad_and_sort(answer_span_ends)
    answer_span_ends = answer_span_ends[context_orig_idxs]

    return QABatch(question_words=question_words,
                   question_chars=question_chars,
                   question_lens=question_lens,
                   question_len_idxs=question_len_idxs,
                   question_orig_idxs=question_orig_idxs,
                   question_mask=question_mask,
                   question_ids=question_ids,
                   context_words=context_words,
                   context_chars=context_chars,
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
