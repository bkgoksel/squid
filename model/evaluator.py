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


class Evaluator(nn.Module):
    """
    Base class for evaluators
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self,
                batch: QABatch,
                model_predictions: ModelPredictions) -> t.Tensor:
        raise NotImplementedError


class SingleClassLossEvaluator(Evaluator):
    """
    Simple Evaluator that outputs the minimum of all
    masked cross entropy losses for each valid starting
    and ending point
    """

    loss_op: nn.BCEWithLogitsLoss

    def __init__(self) -> None:
        super().__init__()
        self.loss_op = nn.BCEWithLogitsLoss()

    def forward(self,
                batch: QABatch,
                model_predictions: ModelPredictions) -> t.Tensor:
        correct_starts = t.nonzero(batch.answer_span_starts).numpy()
        start_losses = []
        for start_idx in correct_starts:
            fake_start = t.zeros_like(batch.answer_span_starts)
            fake_start[0, start_idx] = 1
            start_losses.append(self.loss_op(model_predictions.start_logits, fake_start.float()))
        correct_ends = t.nonzero(batch.answer_span_ends).numpy()
        end_losses = []
        for end_idx in correct_ends:
            fake_end = t.zeros_like(batch.answer_span_ends)
            fake_end[0, end_idx] = 1
            end_losses.append(self.loss_op(model_predictions.end_logits, fake_end.float()))
        return min(start_losses) + min(end_losses)


class MultiClassLossEvaluator(Evaluator):
    """
    Simple Evaluator that computes multi-class loss on all
    starting and ending points
    """

    loss_op: nn.BCEWithLogitsLoss

    def __init__(self) -> None:
        super().__init__()
        self.loss_op = nn.BCEWithLogitsLoss()

    def forward(self,
                batch: QABatch,
                model_predictions: ModelPredictions) -> t.Tensor:
        start_loss = self.loss_op(model_predictions.start_logits, batch.answer_span_starts.float())
        end_loss = self.loss_op(model_predictions.end_logits, batch.answer_span_ends.float())
        return start_loss + end_loss


def get_answer_token_idxs(batch: QABatch,
                          model_predictions: ModelPredictions) -> Dict[QuestionId, Tuple[Any, ...]]:
    """
    Given a ModelPredictions object and text QABatch object for the batch that the predictions
    are from, return a QuestionId -> (answer span start token idx, answe span end token idx) mapping.
    """
    answer_starts = t.max(model_predictions.start_logits, 1)[1].to(t.device('cpu')).numpy()
    answer_ends = t.max(model_predictions.end_logits, 1)[1].to(t.device('cpu')).numpy()
    answers = np.column_stack([answer_starts, answer_ends]).tolist()
    qid_to_answer = {qid: tuple(answer) for qid, answer in zip(batch.question_ids, answers)}
    return qid_to_answer
