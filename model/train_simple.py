import argparse

from pickle import UnpicklingError
from trainer import train_model
from predictor import BasicPredictorConfig, GRUConfig
from tokenizer import Tokenizer, NltkTokenizer
from corpus import Corpus, QADataset
from wv import WordVectors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-file', type=str, default='data/original/train.json')
    parser.add_argument('--dev-file', type=str, default='data/original/dev.json')
    parser.add_argument('--word-vector-file', type=str, default='data/word-vectors/glove/glove.6B.100d.txt')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-epochs', type=int, default=25)
    parser.add_argument('--lstm-hidden-size', type=int, default=512)
    parser.add_argument('--lstm-num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--attention-hidden-size', type=int, default=512)

    return parser.parse_known_args()[0]


def main() -> None:
    args = parse_args()
    predictor_config = BasicPredictorConfig(gru=GRUConfig(hidden_size=args.lstm_hidden_size,
                                                          num_layers=args.lstm_num_layers,
                                                          dropout=args.dropout,
                                                          bidirectional=True),
                                            attention_hidden_size=args.attention_hidden_size,
                                            train_vecs=False,
                                            batch_size=args.batch_size)
    vectors: WordVectors = load_vectors(args.word_vector_file)
    train_dataset: QADataset = load_dataset(args.train_file, vectors)
    dev_dataset: QADataset = load_dataset(args.dev_file, vectors)
    train_model(train_dataset, dev_dataset, vectors, args.num_epochs, args.batch_size, predictor_config)


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


if __name__ == '__main__':
    main()
