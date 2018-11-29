"""
Module that describes evaluator classes. In general these take
ModelPredictions logits from Predictor classes and output final predictions
as well as losses for training.
"""
from typing import Any, Dict, Tuple

import numpy as np
import torch as t
import torch.nn as nn

from model.batcher import QABatch
from model.predictor import ModelPredictions
from model.qa import QuestionId
from model.modules.masked import MaskedOp, MaskTime, MaskMode


class Evaluator(nn.Module):
    """
    Base class for evaluators
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: QABatch, model_predictions: ModelPredictions) -> t.Tensor:
        raise NotImplementedError


class SingleClassLossEvaluator(Evaluator):
    """
    Simple Evaluator for data where there is
    a single correct answer.
    Uses NLLoss after masking the input logits, expects LogSoftmax'ed logits
    """

    loss_op: MaskedOp

    def __init__(self) -> None:
        super().__init__()
        self.loss_op = MaskedOp(
            nn.NLLLoss(), MaskMode.subtract, MaskTime.pre, mask_value=1e30
        )

    def forward(self, batch: QABatch, model_predictions: ModelPredictions) -> t.Tensor:
        # TODO: make sure this is correct
        answer_starts = batch.answer_span_starts.argmax(1)
        answer_ends = batch.answer_span_ends.argmax(1)
        start_loss = self.loss_op(
            model_predictions.start_logits, answer_starts, mask=batch.context_mask
        )
        end_loss = self.loss_op(
            model_predictions.end_logits, answer_ends, mask=batch.context_mask
        )
        return start_loss + end_loss


class MultiClassLossEvaluator(Evaluator):
    """
    Simple Evaluator that computes multi-class loss on all
    starting and ending points for multi-answer training
    Uses BCEWithLogitsLoss after masking the logits
    """

    loss_op: MaskedOp

    def __init__(self) -> None:
        super().__init__()
        self.loss_op = MaskedOp(
            nn.BCEWithLogitsLoss(), MaskMode.subtract, MaskTime.pre, mask_value=1e30
        )

    def forward(self, batch: QABatch, model_predictions: ModelPredictions) -> t.Tensor:
        start_loss = self.loss_op(
            model_predictions.start_logits,
            batch.answer_span_starts.float(),
            mask=batch.context_mask,
        )
        end_loss = self.loss_op(
            model_predictions.end_logits,
            batch.answer_span_ends.float(),
            mask=batch.context_mask,
        )
        return start_loss + end_loss


def get_answer_token_idxs(
    batch: QABatch, model_predictions: ModelPredictions
) -> Dict[QuestionId, Tuple[Any, ...]]:
    """
    Given a ModelPredictions object and text QABatch object for the batch that the predictions
    are from, return a QuestionId -> (answer span start token idx, answe span end token idx) mapping.
    """
    qid_to_ans: Dict[QuestionId, Tuple[Any, ...]] = {}
    all_start_logits = model_predictions.start_logits.to(t.device("cpu")).numpy()
    all_end_logits = model_predictions.end_logits.to(t.device("cpu")).numpy()
    for qid, context_len, start_logits, end_logits in zip(
        batch.question_ids, batch.context_lens, all_start_logits, all_end_logits
    ):
        best_score = 0.0
        best_ans = (0, 0)
        for start_idx in range(context_len):
            start_logit = start_logits[start_idx]
            for end_idx in range(start_idx, context_len):
                end_logit = end_logits[end_idx]
                if start_logit + end_logit >= best_score:
                    best_ans = (start_idx, end_idx)
        qid_to_ans[qid] = best_ans
    """
    answer_starts = (
        t.max(model_predictions.start_logits, 1)[1].to(t.device("cpu")).numpy()
    )
    answer_ends = t.max(model_predictions.end_logits, 1)[1].to(t.device("cpu")).numpy()
    answers = np.column_stack([answer_starts, answer_ends]).tolist()
    qid_to_answer = {
        qid: tuple(answer) for qid, answer in zip(batch.question_ids, answers)
    }
    return qid_to_answer
    """
    return qid_to_ans
