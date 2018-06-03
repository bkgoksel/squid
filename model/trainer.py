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

vector_file: str = 'data/word-vectors/glove/glove.6B.100d.txt'
data_file: str = 'data/original/tiny-dev.json'
corpus_file: str = 'data/saved/tiny-dev'
saved_vector_file: str = 'data/saved/word-vectors/glove.6B.100d'
batch_size = 32
num_epochs = 16
predictor_config = BasicPredictorConfig(gru=GRUConfig(hidden_size=512,
                                                      num_layers=2,
                                                      dropout=0.2,
                                                      bidirectional=True),
                                        attention_hidden_size=256,
                                        train_vecs=False,
                                        batch_size=batch_size)

# print("Building corpus")
# tokenizer: Tokenizer = NltkTokenizer()
# corpus: Corpus = Corpus.from_raw(data_file, tokenizer)
# corpus.save(corpus_file)
print("Reading corpus from disk")
corpus: Corpus = Corpus.from_disk(corpus_file)
print("Corpus done, stats: %s" % str(corpus.stats))

# print("Building Word Vector representation")
# vectors: WordVectors = WordVectors.from_text_vectors(vector_file)
# vectors.save(saved_vector_file)
print("Reading word vecs from disk")
vectors: WordVectors = WordVectors.from_disk(saved_vector_file)

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
