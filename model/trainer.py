"""
Module that holds the training harness
"""

from pickle import UnpicklingError
from typing import Dict

from torch.utils.data import DataLoader
import torch.optim as optim

from model.wv import WordVectors
from model.corpus import Corpus, QADataset
from model.qa import QuestionId
from model.batcher import QABatch, collate_batch
from model.predictor import (PredictorModel,
                             BasicPredictor,
                             BasicPredictorConfig,
                             ModelPredictions)
from model.text_processor import TextProcessor
from model.tokenizer import Tokenizer, NltkTokenizer

from model.modules.embeddor import (Embeddor,
                                    EmbeddorConfig,
                                    make_embeddor)

import model.evaluator as evaluator
from model.evaluator import MultiClassLossEvaluator


def train_model(train_dataset: QADataset,
                dev_dataset: QADataset,
                learning_rate: float,
                num_epochs: int,
                batch_size: int,
                predictor_config: BasicPredictorConfig,
                embeddor_config: EmbeddorConfig,
                fit_one_batch: bool=False) -> PredictorModel:
    """
    Trains a BasicPredictor model on the given train set with given params and returns
    the trained model instance

    :param train_dataset: A Processed QADataset object of training data
    :param dev_dataset: A Processed QADataset object of dev data
    :param learning_rate: LR for Adam optimizer
    :param num_epochs: Number of epochs to train for
    :param batch_size: Size of each training batch
    :param predictor_config: A BasicPredictorConfig object specifying parameters of the model
    :param embeddor_config: An EmbeddorConfig object that specifies the embeddings layer
    :param fit_one_batch: If True train on a single batch (default False)

    :returns: A Trained PredictorModel object
    """

    embeddor: Embeddor = make_embeddor(embeddor_config)
    predictor: PredictorModel = BasicPredictor(embeddor, predictor_config)
    train_evaluator: MultiClassLossEvaluator = MultiClassLossEvaluator()
    trainable_parameters = filter(lambda p: p.requires_grad, set(predictor.parameters()) | set(embeddor.parameters()))
    optimizer: optim.Optimizer = optim.Adam(trainable_parameters, lr=learning_rate)
    loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    print('%d/%d = %d' % (len(train_dataset), batch_size, len(loader)))
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batches = [next(iter(loader))] if fit_one_batch else loader
        for batch_num, batch in enumerate(batches):
            optimizer.zero_grad()
            predictions: ModelPredictions = predictor(batch)
            loss = train_evaluator(batch, predictions)
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            epoch_loss += batch_loss
            print('[%d, %d] loss: %.3f' % (epoch + 1, batch_num + 1, batch_loss))
        epoch_loss = epoch_loss / len(loader)
        print('=== EPOCH %d done. Average loss: %.3f' % (epoch + 1, epoch_loss))
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
        processor: TextProcessor = TextProcessor({'lowercase': True})
        corpus = Corpus.from_raw(filename, tokenizer, processor)
    return QADataset(corpus, vectors)
