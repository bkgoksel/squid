import argparse

from trainer import train_model
from predictor import BasicPredictorConfig, GRUConfig
from tokenizer import Tokenizer, NltkTokenizer
from corpus import Corpus
from wv import WordVectors

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default='data/original/train.json')
    parser.add_argument('--word-vector-file', type=str, default='data/word-vectors/glove/glove.6B.100d.txt')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-epochs', type=int, default=25)
    parser.add_argument('--lstm-hidden-size', type=int, default=512)
    parser.add_argument('--lstm-num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--attention-hidden-size', type=int, default=512)

    return parser.parse_known_args()[0]

def main():
    args = parse_args()
    predictor_config = BasicPredictorConfig(gru=GRUConfig(hidden_size=args.lstm_hidden_size,
                                                          num_layers=args.lstm_num_layers,
                                                          dropout=args.dropout,
                                                          bidirectional=True),
                                            attention_hidden_size=args.attention_hidden_size,
                                            train_vecs=False,
                                            batch_size=args.batch_size)
    corpus : Corpus
    try:
        corpus = Corpus.from_disk(args.train_file)
    except:
        tokenizer: Tokenizer = NltkTokenizer()
        corpus = Corpus.from_raw(args.train_file, tokenizer)

    vectors: WordVectors
    try:
        vectors: WordVectors = WordVectors.from_disk(args.word_vector_file)
    except:
        vectors = WordVectors.from_text_vectors(args.word_vector_file)

    train_model(corpus, vectors, args.num_epochs, args.batch_size, predictor_config)

if __name__ == '__main__':
    main()
