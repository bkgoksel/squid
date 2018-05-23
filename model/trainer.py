"""
Module that holds the training harness
"""

from wv import WordVectors
from corpus import Corpus, SampleCorpus
from tokenizer import Tokenizer, NltkTokenizer
from batcher import RandomBatcher
from predictor import PredictorModel, BasicPredictor
from evaluator import EvaluatorModel, BasicEvaluator

vector_file: str = 'glove.6B.100d.txt'
data_file: str = 'data/original/train.json'
batch_size = 256

tokenizer: Tokenizer = NltkTokenizer()
raw_corpus = Corpus.from_raw(data_file, tokenizer)
vectors: WordVectors = WordVectors.from_text_vectors(vector_file)
sample_corpus: SampleCorpus = SampleCorpus(raw_corpus, vectors)
predictor: PredictorModel = BasicPredictor(vectors, sample_corpus.stats)
evaluator: EvaluatorModel = BasicEvaluator()
batcher: RandomBatcher = RandomBatcher(sample_corpus, batch_size)
