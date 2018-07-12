import argparse
import json

import model.trainer as trainer
from model.predictor import (BasicPredictorConfig,
                             GRUConfig,
                             PredictorModel)
from model.modules.embeddor import (EmbeddorConfig,
                                    WordEmbeddorConfig,
                                    PoolingCharEmbeddorConfig)
from model.corpus import QADataset
from model.wv import WordVectors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-file', type=str, default='data/original/train.json')
    parser.add_argument('--dev-file', type=str, default='data/original/dev.json')
    parser.add_argument('--word-vector-file', type=str, default='data/word-vectors/glove/glove.6B.100d.txt')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-epochs', type=int, default=25)
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--char-embedding-size', type=int, default=200, help='Set to 0 to disable char-level embeddings')
    parser.add_argument('--lstm-hidden-size', type=int, default=512)
    parser.add_argument('--lstm-num-layers', type=int, default=2)
    parser.add_argument('--lstm-unidirectional', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--attention-hidden-size', type=int, default=512)
    parser.add_argument('--answer-train-set', action='store_true', help='if specified generate answers to the train set')
    parser.add_argument('--fit-one-batch', action='store_true', help='if specified try to fit a single batch')
    parser.add_argument('--config-file', type=str, default='', help='if specified load config from this json file (overwrites cli args)')

    return parser.parse_known_args()[0]


def main() -> None:
    args = parse_args()
    vectors: WordVectors = trainer.load_vectors(args.word_vector_file)
    train_dataset: QADataset = trainer.load_dataset(args.train_file, vectors)
    dev_dataset: QADataset = trainer.load_dataset(args.dev_file, vectors)
    predictor_config = BasicPredictorConfig(gru=GRUConfig(hidden_size=args.lstm_hidden_size,
                                                          num_layers=args.lstm_num_layers,
                                                          dropout=args.dropout,
                                                          bidirectional=(not args.lstm_unidirectional)),
                                            attention_hidden_size=args.attention_hidden_size,
                                            batch_size=args.batch_size)
    word_embedding_config = WordEmbeddorConfig(vectors=vectors, train_vecs=False)
    if args.char_embedding_size:
        char_embedding_config = PoolingCharEmbeddorConfig(embedding_dimension=args.char_embedding_size,
                                                          char_vocab_size=train_dataset.corpus.stats.char_vocab_size)
    else:
        char_embedding_config = None

    embeddor_config = EmbeddorConfig(word_embeddor=word_embedding_config,
                                     char_embeddor=char_embedding_config)
    print('Training with config: %s \n vectors: %s \n training file: %s \n dev file: %s \n' %
          (predictor_config, args.word_vector_file, args.train_file, args.dev_file))
    model: PredictorModel = trainer.train_model(train_dataset,
                                                dev_dataset,
                                                args.lr,
                                                args.num_epochs,
                                                args.batch_size,
                                                predictor_config,
                                                embeddor_config,
                                                args.fit_one_batch)
    if args.answer_train_set:
        train_answers = trainer.answer_dataset(train_dataset, model)
        with open('train-pred.json', 'w') as f:
            json.dump(train_answers, f)


if __name__ == '__main__':
    main()
