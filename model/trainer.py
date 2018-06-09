"""
Module that holds the training harness
"""

from pickle import UnpicklingError
from typing import Dict

from torch.utils.data import DataLoader
import torch.optim as optim

from wv import WordVectors
from corpus import Corpus, QADataset
from qa import QuestionId
from batcher import QABatch, collate_batch
from predictor import PredictorModel, BasicPredictor, BasicPredictorConfig, ModelPredictions
from tokenizer import Tokenizer, NltkTokenizer

import evaluator
from evaluator import LossEvaluator


def train_model(train_dataset: QADataset,
                dev_dataset: QADataset,
                vectors: WordVectors,
                num_epochs: int,
                batch_size: int,
                predictor_config: BasicPredictorConfig) -> PredictorModel:
    predictor: PredictorModel = BasicPredictor(vectors, predictor_config)
    train_evaluator: LossEvaluator = LossEvaluator()
    trainable_parameters = filter(lambda p: p.requires_grad, predictor.parameters())
    optimizer: optim.Optimizer = optim.Adam(trainable_parameters)
    loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    print('%d/%d = %d' % (len(train_dataset), batch_size, len(loader)))
    for epoch in range(num_epochs):
        for batch_num, batch in enumerate(loader):
            optimizer.zero_grad()
            predictions: ModelPredictions = predictor(batch)
            loss = train_evaluator(batch, predictions)
            loss.backward()
            optimizer.step()
            running_loss = loss.item()
            print('[%d, %d] loss: %.3f' % (epoch + 1, batch_num + 1, running_loss))
    return predictor


def answer_dataset(dataset: QADataset,
                   predictor: PredictorModel,
                   batch_size: int=16) -> Dict[QuestionId, str]:
    loader: DataLoader = DataLoader(dataset, batch_size, collate_fn=collate_batch)
    batch: QABatch
    qid_to_answer: Dict[QuestionId, str] = dict()
    for batch_num, batch in enumerate(loader):
        predictions: ModelPredictions = predictor(batch)
        qid_to_answer.update(evaluator.get_answer_token_idxs(batch, predictions))
    return dataset.get_answer_texts(qid_to_answer)


def load_vectors(filename: str) -> WordVectors:
    try:
        vectors = WordVectors.from_disk(filename)
    except (IOError, UnpicklingError) as e:
        vectors = WordVectors.from_text_vectors(filename)
    return vectors


def load_dataset(filename: str, vectors: WordVectors) -> QADataset:
    corpus: Corpus
    try:
        corpus = Corpus.from_disk(filename)
    except (IOError, UnpicklingError) as e:
        tokenizer: Tokenizer = NltkTokenizer()
        corpus = Corpus.from_raw(filename, tokenizer)
    return QADataset(corpus, vectors)

