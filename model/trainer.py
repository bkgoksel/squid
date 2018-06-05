"""
Module that holds the training harness
"""

from torch.utils.data import DataLoader
import torch.optim as optim

from wv import WordVectors
from corpus import QADataset
from batcher import QABatch, collate_batch
from predictor import PredictorModel, BasicPredictor, BasicPredictorConfig, ModelPredictions
from evaluator import LossEvaluator, AnswerEvaluator


def train_model(train_dataset: QADataset,
                dev_dataset: QADataset,
                vectors: WordVectors,
                num_epochs: int,
                batch_size: int,
                predictor_config: BasicPredictorConfig) -> None:
    predictor: PredictorModel = BasicPredictor(vectors, predictor_config)
    train_evaluator: LossEvaluator = LossEvaluator()
    trainable_parameters = filter(lambda p: p.requires_grad, predictor.parameters())
    optimizer: optim.Optimizer = optim.Adam(trainable_parameters)
    loader: DataLoader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_batch)
    for epoch in range(num_epochs):
        for batch_num, batch in enumerate(loader):
            optimizer.zero_grad()
            predictions: ModelPredictions = predictor(batch)
            loss = train_evaluator(batch, predictions)
            loss.backward()
            optimizer.step()
            running_loss = loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_num + 1, running_loss))
        # print('Epoch %d over, evaluating on dev set' % (epoch + 1))
        # eval_model(dev_dataset, predictor)


def eval_model(dataset: QADataset,
               predictor: PredictorModel,
               batch_size: int=16) -> None:
    loader: DataLoader = DataLoader(dataset, batch_size, collate_fn=collate_batch)
    answer_evaluator: AnswerEvaluator = AnswerEvaluator()
    batch: QABatch
    for batch_num, batch in enumerate(loader):
        predictions: ModelPredictions = predictor(batch)
        evaluation = answer_evaluator(predictions, batch)
        print(evaluation)
