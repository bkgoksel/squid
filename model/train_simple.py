import argparse
import json

import trainer
from predictor import BasicPredictorConfig, GRUConfig, PredictorModel
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
    parser.add_argument('--answer-train-set', action='store_true', help='if specified generate answers to the train set')

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
    vectors: WordVectors = trainer.load_vectors(args.word_vector_file)
    train_dataset: QADataset = trainer.load_dataset(args.train_file, vectors)
    dev_dataset: QADataset = trainer.load_dataset(args.dev_file, vectors)
    model: PredictorModel = trainer.train_model(train_dataset, dev_dataset, vectors, args.num_epochs, args.batch_size, predictor_config)
    if args.answer_train_set:
        train_answers = trainer.answer_dataset(train_dataset, model)
        with open('train-pred.json', 'w') as f:
            json.dump(train_answers, f)


if __name__ == '__main__':
    main()
