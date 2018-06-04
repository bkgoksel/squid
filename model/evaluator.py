"""
Module that describes evaluator classes. In general these take
ModelPredictions logits from Predictor classes and output final predictions
as well as losses for training.
"""
from typing import Any, Optional, Tuple

import torch as t
import torch.nn as nn

from batcher import QABatch
from predictor import ModelPredictions


class EvaluatorModel(nn.Module):
    """
    Base class for any Evaluator Model
    """

    def __init__(self):
        super().__init__()

    def forward(self,
                batch: QABatch,
                model_predictions: ModelPredictions) -> Tuple[t.Tensor, Optional[Any]]:
            """
            Computes a Loss tensor given a Batch and model predictions
j           Optionally outputs a list of FinalPrediction objects
            """
            raise NotImplementedError


class LossEvaluator(EvaluatorModel):
    """
    Simple Evaluator that outputs masked cross entropy loss
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


class AnswerEvaluator(EvaluatorModel):
    """
    Evaluator that generates answers from predictions and computes F1 and EM scores
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self,
                batch: QABatch,
                model_predictions: ModelPredictions) -> t.Tensor:
        span_start, span_end = self.get_answer(model_predictions)
        return 0

    def get_answers(self, model_predictions: ModelPredictions) -> Any:
        """
        Returns the most likely answer spans given the predictions for each question
        :param model_predictions: A ModelPredictions object
        :returns: A Torch LongTensor of shape [batch_size,1,1]
        """
        _, answer_starts = t.max(model_predictions.start_logits, 1)
        _, answer_ends = t.max(model_predictions.end_logits, 1)
        return answer_starts, answer_ends
