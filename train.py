import argparse
import json
from typing import Tuple

import torch as t

import model.trainer as trainer
from model.tokenizer import Tokenizer, NltkTokenizer
from model.text_processor import TextProcessor
from model.predictor import PredictorConfig, GRUConfig, PredictorModel, DocQAPredictor
from model.modules.embeddor import (
    Embeddor,
    EmbeddorConfig,
    make_embeddor,
    WordEmbeddorConfig,
    PoolingCharEmbeddorConfig,
)
from model.corpus import TrainDataset, EvalDataset
from model.util import get_device
from model.wv import WordVectors

DEFAULT_ARGS = {
    "train_file": "data/original/train.json",
    "dev_file": "data/original/dev.json",
    "word_vector_file": "data/word-vectors/glove/glove.6B.100d.txt",
    "batch_size": 40,
    "num_epochs": 15,
    "lr": 1e-4,
    "char_embedding_size": 50,
    "rnn_hidden_size": 100,
    "rnn_num_layers": 1,
    "max_context_size": 250,
    "max_question_size": 100,
    "dropout": 0.2,
    "config_file": "",
    "run_name": "train-run",
}


def parse_args() -> argparse.Namespace:
    # Don't format args
    # fmt: off
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-file", type=str, default=DEFAULT_ARGS["train_file"])
    parser.add_argument("--dev-file", type=str, default=DEFAULT_ARGS["dev_file"])
    parser.add_argument("--word-vector-file", type=str, default=DEFAULT_ARGS["word_vector_file"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_ARGS["batch_size"])
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_ARGS["num_epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULT_ARGS["lr"])
    parser.add_argument("--char-embedding-size", type=int, default=DEFAULT_ARGS["char_embedding_size"], help="Set to 0 to disable char-level embeddings")
    parser.add_argument("--max-context-size", type=int, default=DEFAULT_ARGS["max_context_size"], help="Trim all context values to this length during training (0 for unlimited)")
    parser.add_argument("--max-question-size", type=int, default=DEFAULT_ARGS["max_question_size"], help="Trim all context values to this length during training (0 for unlimited)")
    parser.add_argument("--rnn-hidden-size", type=int, default=DEFAULT_ARGS["rnn_hidden_size"])
    parser.add_argument("--rnn-num-layers", type=int, default=DEFAULT_ARGS["rnn_num_layers"])
    parser.add_argument("--rnn-unidirectional", action="store_true")
    parser.add_argument("--dropout", type=float, default=DEFAULT_ARGS["dropout"])
    parser.add_argument("--answer-train-set", action="store_true", help="if specified generate answers to the train set")
    parser.add_argument("--debug", action="store_true", help="if specified debug by fitting a single batch and profiling")
    parser.add_argument("--multi-answer", action="store_true", help="if specified don't truncate answer spans down to one")
    parser.add_argument("--no-self-attention", action="store_true", help="if specified don't use self attention")
    parser.add_argument("--disable-cuda", action="store_true", help="if specified don\t use CUDA even if available")
    parser.add_argument("--config-file", type=str, default=DEFAULT_ARGS["config_file"], help="if specified load config from this json file (overwrites cli args)")
    parser.add_argument("--run-name", type=str, default=DEFAULT_ARGS["run_name"], help="name of run (also used for model saving and initialization)",)
    # fmt: on

    return parser.parse_known_args()[0]


def initialize_model(
    args: argparse.Namespace, train_dataset: TrainDataset, vectors: WordVectors
) -> PredictorModel:
    """
    Given the command line args and the dataset and vectors to train on,
    initializes a new model
    :param args: argparse Namespace coming from the CLI
    :param train_dataset: TrainDataset object to compute vocabulary on
    :param vectors: WordVectors object to use in the model
    :returns: A new PredictorModel
    """
    device = get_device(args.use_cuda)
    predictor_config = PredictorConfig(
        gru=GRUConfig(
            hidden_size=args.rnn_hidden_size,
            num_layers=args.rnn_num_layers,
            dropout=args.dropout,
            bidirectional=(not args.rnn_unidirectional),
        ),
        batch_size=args.batch_size,
        use_self_attention=(not args.no_self_attention),
    )
    word_embedding_config = WordEmbeddorConfig(
        vectors=vectors, token_mapping=train_dataset.token_mapping, train_vecs=False
    )
    if args.char_embedding_size:
        char_embedding_config = PoolingCharEmbeddorConfig(
            embedding_dimension=args.char_embedding_size,
            char_vocab_size=train_dataset.corpus.stats.char_vocab_size,
        )
    else:
        char_embedding_config = None

    embeddor_config = EmbeddorConfig(
        word_embeddor=word_embedding_config, char_embeddor=char_embedding_config
    )
    embeddor: Embeddor = make_embeddor(embeddor_config, device)
    return DocQAPredictor(embeddor, predictor_config).to(device)


def get_model(
    args: argparse.Namespace
) -> Tuple[PredictorModel, TrainDataset, EvalDataset]:
    """
    Given the CLI args, tries to load a model from disk if possible,
    otherwise creates a new model.
    Returns the model alongside the Datasets to be used for training
    :param args: CLI args to get the model and datasets
    :returns: a Tuple of the model, the training dataset and the dev dataset
    """
    device = get_device(args.use_cuda)

    tokenizer: Tokenizer = NltkTokenizer()
    processor: TextProcessor = TextProcessor({"lowercase": True})
    vectors: WordVectors = WordVectors.load_vectors(args.word_vector_file)

    train_dataset: TrainDataset = TrainDataset.load_dataset(
        args.train_file, vectors, tokenizer, processor, args.multi_answer
    )
    dev_dataset: EvalDataset = EvalDataset.load_dataset(
        args.dev_file,
        train_dataset.token_mapping,
        train_dataset.char_mapping,
        tokenizer,
        processor,
    )
    try:
        print(f"Attempting to load model to train from {args.run_name}")
        model: PredictorModel = t.load(args.run_name).to(device)
    except IOError as e:
        print(f"Can't load model: {e}, initializing from scratch")
        model = initialize_model(args, train_dataset, vectors)

    return model, train_dataset, dev_dataset


def get_training_config(args: argparse.Namespace) -> trainer.TrainingConfig:
    """
    Parse the command line args builds a TrainingConfig object
    :param args: argparse namespace object from the CLI invocation
    :returns: A well formatted TrainingConfig object that can be used
        for training the model
    """
    return trainer.TrainingConfig(
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_question_size=args.max_question_size,
        max_context_size=args.max_context_size,
        use_cuda=not args.disable_cuda,
        loader_num_workers=args.loader_num_workers,
        model_checkpoint_path=args.run_name,
    )


def main() -> None:
    args = parse_args()
    model, train_dataset, dev_dataset = get_model(args)
    training_config = get_training_config(args)

    trainer.train_model(
        model, train_dataset, dev_dataset, training_config, debug=args.debug
    )
    if args.answer_train_set:
        train_answers = trainer.answer_dataset(train_dataset, model, args.use_cuda)
        with open("train-pred.json", "w") as f:
            json.dump(train_answers, f)
    dev_answers = trainer.answer_dataset(dev_dataset, model, args.use_cuda)
    with open("dev-pred.json", "w") as f:
        json.dump(dev_answers, f)
    print("Final evaluation on dev")
    eval_results = trainer.evaluate_on_squad_dataset(
        dev_dataset, model, args.use_cuda, 64
    )
    print(eval_results)
    print("Saving model to {args.run_name}")
    t.save(model, args.run_name)


if __name__ == "__main__":
    main()
