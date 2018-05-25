"""
Module that holds the training harness
"""

from torch.utils.data import DataLoader

from wv import WordVectors
from corpus import Corpus, QADataset
from tokenizer import Tokenizer, NltkTokenizer
from batcher import QABatch, collate_batch
# from predictor import PredictorModel, BasicPredictor
# from evaluator import EvaluatorModel, BasicEvaluator

vector_file: str = 'data/wordvecs/fake.txt'
data_file: str = 'data/original/tiny-dev.json'
corpus_file: str = 'data/saved/tiny-dev-pos'
saved_vector_file: str = 'data/saved/vectors/fake'
batch_size = 3

# tokenizer: Tokenizer = NltkTokenizer()
# corpus: Corpus = Corpus.from_raw(data_file, tokenizer)
corpus: Corpus = Corpus.from_disk(corpus_file)
# vectors: WordVectors = WordVectors.from_text_vectors(vector_file)
# vectors.save('data/saved/vectors/fake')
vectors: WordVectors = WordVectors.from_disk(saved_vector_file)
dataset: QADataset = QADataset(corpus, vectors)
predictor: PredictorModel = BasicPredictor(vectors, dataset.stats)
# evaluator: EvaluatorModel = BasicEvaluator()
loader: DataLoader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_batch)
batch: QABatch = next(iter(loader))
