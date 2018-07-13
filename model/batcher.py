"""
Module that handles batching logic
"""

from typing import List, Any, Tuple
import numpy as np
import torch as t
from torch.nn.utils.rnn import pad_sequence

from model.modules.masked import mask_sequence
from model.qa import EncodedSample, QuestionId


class QABatch():
    """
    Holds a batch of samples in a form that's easy for the model to use
    len_idxs and orig_idxs allow for length-sorted or original orderings
    of the respective texts i.e.
    question_words[question_len_idxs] = length_sorted_questions
    length_sorted_questions[question_orig_idxs] = question_words

    masks, and question_ids come in original ordering
    lengths come sorted
    """
    question_ids: List[QuestionId]
    question_words: t.LongTensor
    question_chars: t.LongTensor
    question_lens: t.LongTensor
    question_len_idxs: t.LongTensor
    question_orig_idxs: t.LongTensor
    question_mask: t.LongTensor
    context_words: t.LongTensor
    context_chars: t.LongTensor
    context_lens: t.LongTensor
    context_len_idxs: t.LongTensor
    context_orig_idxs: t.LongTensor
    context_mask: t.LongTensor
    answer_span_starts: t.LongTensor
    answer_span_ends: t.LongTensor

    def __init__(self,
                 question_ids: List[QuestionId],
                 question_words: t.LongTensor,
                 question_chars: t.LongTensor,
                 question_lens: t.LongTensor,
                 question_len_idxs: t.LongTensor,
                 question_orig_idxs: t.LongTensor,
                 question_mask: t.LongTensor,
                 context_words: t.LongTensor,
                 context_chars: t.LongTensor,
                 context_lens: t.LongTensor,
                 context_len_idxs: t.LongTensor,
                 context_orig_idxs: t.LongTensor,
                 context_mask: t.LongTensor,
                 answer_span_starts: t.LongTensor,
                 answer_span_ends: t.LongTensor) -> None:
        self.question_ids = question_ids
        self.question_words = question_words
        self.question_chars = question_chars
        self.question_lens = question_lens
        self.question_len_idxs = question_len_idxs
        self.question_orig_idxs = question_orig_idxs
        self.question_mask = question_mask
        self.context_words = context_words
        self.context_chars = context_chars
        self.context_lens = context_lens
        self.context_len_idxs = context_len_idxs
        self.context_orig_idxs = context_orig_idxs
        self.context_mask = context_mask
        self.answer_span_starts = answer_span_starts
        self.answer_span_ends = answer_span_ends

    def to(self, device: Any):
        """
        Moves all Tensors to device, calls .to on all tensors in batch
        :param device: a PyTorch device
        :returns: self
        """
        self.question_words = self.question_words.to(device)
        self.question_chars = self.question_chars.to(device)
        self.question_lens = self.question_lens.to(device)
        self.question_len_idxs = self.question_len_idxs.to(device)
        self.question_orig_idxs = self.question_orig_idxs.to(device)
        self.question_mask = self.question_mask.to(device)
        self.context_words = self.context_words.to(device)
        self.context_chars = self.context_chars.to(device)
        self.context_lens = self.context_lens.to(device)
        self.context_len_idxs = self.context_len_idxs.to(device)
        self.context_orig_idxs = self.context_orig_idxs.to(device)
        self.context_mask = self.context_mask.to(device)
        self.answer_span_starts = self.answer_span_starts.to(device)
        self.answer_span_ends = self.answer_span_ends.to(device)

        return self


def collate_batch(batch: List[EncodedSample]) -> QABatch:
    """
    Takes a list of EncodedSample objects and creates a PyTorch batch
    For chars:
        context_chars[batch, word, char_idx] -> (batch_len, max_ctx_len, max_word_len)
    :param batch: List[EncodedSample] QA samples
    :returns: a QABatch
    """
    question_words_list = []
    question_chars_list = []
    question_ids = []
    context_words_list = []
    context_chars_list = []
    answer_span_starts = []
    answer_span_ends = []

    batch_size = len(batch)
    max_question_word_len = 0
    max_ctx_word_len = 0

    for sample in batch:
        question_words_list.append(sample.question_words)
        question_ids.append(sample.question_id)
        context_words_list.append(sample.context_words)
        answer_span_starts.append(sample.span_starts)
        answer_span_ends.append(sample.span_ends)
        question_chars_list.append(sample.question_chars)
        max_question_word_len = max(max_question_word_len, max(word.size for word in sample.question_chars))
        context_chars_list.append(sample.context_chars)
        max_ctx_word_len = max(max_ctx_word_len, max(word.size for word in sample.context_chars))

    question_words, question_orig_idxs, question_len_idxs, question_lens = pad_and_sort(question_words_list)
    question_words = question_words[question_orig_idxs]
    question_mask = mask_sequence(question_words)

    # TODO: Is there a more efficient way of doing this?
    max_question_len = question_lens[0]
    question_chars = np.zeros((batch_size, max_question_len, max_question_word_len))
    for batch_idx, q_chars in enumerate(question_chars_list):
        for word_idx, word in enumerate(q_chars):
            question_chars[batch_idx, word_idx, :word.size] = word
    question_chars = t.LongTensor(question_chars)

    context_words, context_orig_idxs, context_len_idxs, context_lens = pad_and_sort(context_words_list)
    context_words = context_words[context_orig_idxs]
    context_mask = mask_sequence(context_words)

    # TODO: Is there a more efficient way of doing this?
    max_context_len = context_lens[0]
    context_chars = np.zeros((batch_size, max_context_len, max_ctx_word_len))
    for batch_idx, c_chars in enumerate(context_chars_list):
        for word_idx, word in enumerate(c_chars):
            context_chars[batch_idx, word_idx, :word.size] = word
    context_chars = t.LongTensor(context_chars)

    answer_span_starts, _, _, _ = pad_and_sort(answer_span_starts)
    answer_span_starts = answer_span_starts[context_orig_idxs]

    answer_span_ends, _, _, _ = pad_and_sort(answer_span_ends)
    answer_span_ends = answer_span_ends[context_orig_idxs]

    assert max_question_word_len > 0, 'Empty word in a question in batch'
    assert max_question_len > 0, 'Empty question in batch'
    assert max_ctx_word_len > 0, 'Empty word in a context in batch'
    assert max_context_len > 0, 'Empty context in batch'

    return QABatch(question_ids=question_ids,
                   question_words=question_words,
                   question_chars=question_chars,
                   question_lens=question_lens,
                   question_len_idxs=question_len_idxs,
                   question_orig_idxs=question_orig_idxs,
                   question_mask=question_mask,
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
    if len(seq) == 1:
        batch = t.LongTensor(seq)
        orig_idxs = t.LongTensor([0])
        length_idxs = t.LongTensor([0])
        lengths = t.LongTensor([len(seq[0])])
        return batch, orig_idxs, length_idxs, lengths
    lengths = t.LongTensor([el.shape[0] for el in seq])
    lengths, length_idxs = lengths.sort(0, descending=True)
    seq = [t.LongTensor(seq[i]) for i in length_idxs]
    batch = pad_sequence(seq, batch_first=True)
    _, orig_idxs = length_idxs.sort()
    return batch, orig_idxs, length_idxs, lengths
