"""
Module that holds the training harness
"""

from torch.utils.data import DataLoader
import torch.optim as optim

from wv import WordVectors
from corpus import Corpus, QADataset
from tokenizer import Tokenizer, NltkTokenizer
from batcher import collate_batch
from predictor import PredictorModel, BasicPredictor, BasicPredictorConfig, GRUConfig, ModelPredictions
from evaluator import EvaluatorModel, BasicEvaluator

def train_model(corpus: Corpus,
                vectors: WordVectors,
                num_epochs: int,
                batch_size: int,
                predictor_config: BasicPredictorConfig) -> None:
    dataset: QADataset = QADataset(corpus, vectors)
    predictor: PredictorModel = BasicPredictor(vectors, dataset.stats, predictor_config)
    evaluator: EvaluatorModel = BasicEvaluator()
    trainable_parameters = filter(lambda p: p.requires_grad, predictor.parameters())
    optimizer: optim.Optimizer = optim.Adam(trainable_parameters)
    loader: DataLoader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_batch)
    for epoch in range(num_epochs):
        for batch_num, batch in enumerate(loader):
            optimizer.zero_grad()
            predictions: ModelPredictions = predictor(batch)
            loss = evaluator(batch, predictions)
            loss.backward()
            optimizer.step()
            running_loss = loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_num + 1, running_loss))
