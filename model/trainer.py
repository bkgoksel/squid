"""
Module that holds the training harness
"""

import json
from typing import Any, Dict

import torch as t
from torch.utils.data import DataLoader
import torch.optim as optim

from model.corpus import (QADataset,
                          TrainDataset,
                          EvalDataset)
from model.qa import QuestionId
from model.batcher import QABatch, collate_batch
from model.predictor import (PredictorModel,
                             BasicPredictor,
                             BasicPredictorConfig,
                             ModelPredictions)

from model.util import get_device

from model.modules.embeddor import (Embeddor,
                                    EmbeddorConfig,
                                    make_embeddor)

import model.evaluator as evaluator
from model.evaluator import (Evaluator,
                             MultiClassLossEvaluator,
                             SingleClassLossEvaluator)

import scripts.evaluate_v1_1 as evaluate_v1_1
import scripts.evaluate_v2_0 as evaluate_v2_0


def train_model(train_dataset: TrainDataset,
                dev_dataset: EvalDataset,
                learning_rate: float,
                num_epochs: int,
                batch_size: int,
                predictor_config: BasicPredictorConfig,
                embeddor_config: EmbeddorConfig,
                use_cuda: bool=False,
                fit_one_batch: bool=False) -> PredictorModel:
    """
    Trains a BasicPredictor model on the given train set with given params and returns
    the trained model instance

    :param train_dataset: A Processed TrainDataset object of training data
    :param dev_dataset: A Processed EvalDataset object of dev data
    :param learning_rate: LR for Adam optimizer
    :param num_epochs: Number of epochs to train for
    :param batch_size: Size of each training batch
    :param predictor_config: A BasicPredictorConfig object specifying parameters of the model
    :param embeddor_config: An EmbeddorConfig object that specifies the embeddings layer
    :param use_cuda: If True use CUDA (default False)
    :param fit_one_batch: If True train on a single batch (default False)

    :returns: A Trained PredictorModel object
    """

    device = get_device(use_cuda)
    embeddor: Embeddor = make_embeddor(embeddor_config, device)
    predictor: PredictorModel = BasicPredictor(embeddor, predictor_config).to(device)
    train_evaluator: Evaluator
    if fit_one_batch:
        # Take the minimum loss so the model can achieve 0 loss for questions with
        # multiple correct answers
        train_evaluator = SingleClassLossEvaluator().to(device)
    else:
        train_evaluator = MultiClassLossEvaluator().to(device)
    trainable_parameters = filter(lambda p: p.requires_grad, set(predictor.parameters()) | set(embeddor.parameters()))
    optimizer: optim.Optimizer = optim.Adam(trainable_parameters, lr=learning_rate)
    loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    batches = [next(iter(loader)).to(device)] if fit_one_batch else loader
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_num, batch in enumerate(batches):
            optimizer.zero_grad()
            batch.to(device)
            predictions: ModelPredictions = predictor(batch)
            loss = train_evaluator(batch, predictions)
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            epoch_loss += batch_loss
            print('[%d, %d] loss: %.3f' % (epoch + 1, batch_num + 1, batch_loss))
        epoch_loss = epoch_loss / len(loader)
        print('=== EPOCH %d done. Average loss: %.3f' % (epoch + 1, epoch_loss))
        if epoch and epoch % 10 == 9:
            validate(dev_dataset, predictor, train_evaluator, use_cuda, batch_size)
    return predictor


def validate(dataset: QADataset,
             predictor: PredictorModel,
             evaluator: Any,
             use_cuda: bool,
             batch_size: int=16) -> None:
    print('=== EPOCH %d: Measuring QA performance on the dev set')
    try:
        dev_perf = evaluate_on_squad_dataset(dataset, predictor, use_cuda, batch_size)
        print('=== Dev set performance: {}'.format(json.dumps(dev_perf)))
    except Exception as err:
        print('Error when trying to get full evaluation: {}'.format(err))
    print('=== EPOCH %d: Measuring loss on the dev set')
    dev_loss = get_dataset_loss(dataset, predictor, evaluator, use_cuda, batch_size)
    print('=== Dev set loss: {}'.format(dev_loss))


def get_dataset_loss(dataset: QADataset,
                     predictor: PredictorModel,
                     evaluator: Any,
                     use_cuda: bool,
                     batch_size: int=16) -> float:
    device = get_device(use_cuda)
    loader: DataLoader = DataLoader(dataset, batch_size, collate_fn=collate_batch)
    total_loss = 0.0
    batch: QABatch
    for batch in loader:
        with t.no_grad():
            batch.to(device)
            predictions: ModelPredictions = predictor(batch)
            total_loss += evaluator(batch, predictions).item()
    return total_loss


def answer_dataset(dataset: QADataset,
                   predictor: PredictorModel,
                   use_cuda: bool,
                   batch_size: int=16) -> Dict[QuestionId, str]:
    device = get_device(use_cuda)
    loader: DataLoader = DataLoader(dataset, batch_size, collate_fn=collate_batch)
    batch: QABatch
    qid_to_answer: Dict[QuestionId, str] = dict()
    for batch_num, batch in enumerate(loader):
        with t.no_grad():
            batch.to(device)
            predictions: ModelPredictions = predictor(batch)
            qid_to_answer.update(evaluator.get_answer_token_idxs(batch, predictions))
    return dataset.get_answer_texts(qid_to_answer)


def evaluate_on_squad_dataset(dataset: QADataset,
                              predictor: PredictorModel,
                              use_cuda: bool,
                              batch_size: int=16) -> Dict[str, str]:
    answer_dict = answer_dataset(dataset, predictor, use_cuda, batch_size)
    with open(dataset.source_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset_version = dataset_json['version']
        if dataset_version == '1.1.':
            eval_fn = evaluate_v1_1.evaluate
        elif dataset_version == '2.0':
            eval_fn = evaluate_v2_0.evaluate
        dataset_dict = dataset_json['data']
    return eval_fn(dataset_dict, answer_dict)
