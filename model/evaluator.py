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
            Optionally outputs a list of FinalPrediction objects
            """
            raise NotImplementedError


class BasicEvaluator(EvaluatorModel):
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
        start_loss = self.loss_op(model_predictions.start_logits, batch.answer_span_starts)
        end_loss = self.loss_op(model_predictions.end_logits, batch.answer_span_ends)
        return start_loss + end_loss
