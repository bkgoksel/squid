"""
Module that describes evaluator classes. In general these take
ModelPredictions logits from Predictor classes and output final predictions
as well as losses for training.
"""
from typing import Any, Dict, Tuple

import numpy as np
import torch as t
import torch.nn as nn

from batcher import QABatch
from predictor import ModelPredictions
from qa import QuestionId


class SingleClassLossEvaluator(nn.Module):
    """
    Simple Evaluator that outputs the minimum of all
    masked cross entropy losses for each valid starting
    and ending point
    """

    loss_op: nn.BCEWithLogitsLoss

    def __init__(self) -> None:
        super().__init__()
        self.loss_op = nn.CrossEntropyWithLogitsLoss()

    def forward(self,
                batch: QABatch,
                model_predictions: ModelPredictions) -> t.Tensor:
        raise NotImplementedError


class MultiClassLossEvaluator(nn.Module):
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
    answer_starts = t.max(model_predictions.start_logits, 1)[1].numpy()
    answer_ends = t.max(model_predictions.end_logits, 1)[1].numpy()
    answers = np.column_stack([answer_starts, answer_ends]).tolist()
    qid_to_answer = {qid: tuple(answer) for qid, answer in zip(batch.question_ids, answers)}
    return qid_to_answer
