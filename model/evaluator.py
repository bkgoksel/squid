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
from qa import FinalPrediction


class EvaluatorModel(nn.Module):
    """
    Base class for any Evaluator Model
    """

    def __init__(self):
        super().__init__()

    def forward(self,
                batch: QABatch,
                model_predictions: ModelPredictions) -> Tuple[t.Tensor, Optional[Iterable[FinalPrediction]]]:
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
        self.loss_op = BCEWithLogitsLoss()

    def forward(self,
                batch: QABatch,
                model_predictions: ModelPredictions) -> t.Tensor:
        """
        """
        pass

    def compute_loss(logits, target, lens):
        """
        Computes masked cross entropy loss
        :param logits: Prediction logits from model, of shape:
            (batch_len, max_seq_len)
        """
        pass
