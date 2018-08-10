import argparse
import copy
import json
from typing import Dict, Any


class TrainArgs:
    train_file: str
    dev_file: str
    word_vector_file: str
    batch_size: int
    num_epochs: int
    lr: float
    char_embedding_size: int
    rnn_hidden_size: int
    rnn_num_layers: int
    max_context_size: int
    max_question_size: int
    loader_num_workers: int
    dropout: float
    run_name: str

    DEFAULT_ARGS = {
        "train_file": "data/original/train.json",
        "dev_file": "data/original/dev.json",
        "word_vector_file": "data/word-vectors/glove/glove.6B.100d.txt",
        "batch_size": 40,
        "num_epochs": 15,
        "lr": 1e-4,
        "char_embedding_size": 100,
        "rnn_hidden_size": 100,
        "rnn_num_layers": 1,
        "max_context_size": 200,
        "max_question_size": 100,
        "loader_num_workers": 2,
        "dropout": 0.2,
        "config_file": "train-config.json",
        "run_name": "train-run",
    }

    def __init__(self, arg_dict: Dict[str, Any]) -> None:
        self.train_file = arg_dict.get("train_file", self.DEFAULT_ARGS["train_file"])
        self.dev_file = arg_dict.get("dev_file", self.DEFAULT_ARGS["dev_file"])
        self.word_vector_file = arg_dict.get(
            "word_vector_file", self.DEFAULT_ARGS["word_vector_file"]
        )
        self.batch_size = arg_dict.get("batch_size", self.DEFAULT_ARGS["batch_size"])
        self.num_epochs = arg_dict.get("num_epochs", self.DEFAULT_ARGS["num_epochs"])
        self.lr = arg_dict.get("lr", self.DEFAULT_ARGS["lr"])
        self.char_embedding_size = arg_dict.get(
            "char_embedding_size", self.DEFAULT_ARGS["char_embedding_size"]
        )
        self.rnn_hidden_size = arg_dict.get(
            "rnn_hidden_size", self.DEFAULT_ARGS["rnn_hidden_size"]
        )
        self.rnn_num_layers = arg_dict.get(
            "rnn_num_layers", self.DEFAULT_ARGS["rnn_num_layers"]
        )
        self.max_context_size = arg_dict.get(
            "max_context_size", self.DEFAULT_ARGS["max_context_size"]
        )
        self.max_question_size = arg_dict.get(
            "max_question_size", self.DEFAULT_ARGS["max_question_size"]
        )
        self.loader_num_workers = arg_dict.get(
            "loader_num_workers", self.DEFAULT_ARGS["loader_num_workers"]
        )
        self.dropout = arg_dict.get("dropout", self.DEFAULT_ARGS["dropout"])
        self.run_name = arg_dict.get("run_name", self.DEFAULT_ARGS["run_name"])

    @classmethod
    def get_args(cls) -> "TrainArgs":
        """
        Combines the default arguments, config file (if exists) and command line args
        to build a final args dict.
        Precedence is as following:
            CLI args > config file > default arguments
        """
        args = copy.copy(cls.DEFAULT_ARGS)
        cli_args = cls.parse_cli_args()
        try:
            config_file_args = json.load(open(cli_args.config_file, "r"))
        except (IOError, json.JSONDecodeError) as e:
            print(f"Cannot load {cli_args.config_file}: {e}, continuing")
            config_file_args = {}
        args.update(config_file_args)
        args.update(vars(cli_args))
        return TrainArgs(args)

    @staticmethod
    def parse_cli_args() -> argparse.Namespace:
        # Don't format args
        # fmt: off
        parser = argparse.ArgumentParser()
        parser.add_argument("--run-name", type=str, help="name of run (also used for model saving and initialization)",)
        parser.add_argument("--config-file", type=str, help="load config from this json file other command line args override this config")
        parser.add_argument("--train-file", type=str)
        parser.add_argument("--dev-file", type=str)
        parser.add_argument("--word-vector-file", type=str)
        parser.add_argument("--batch-size", type=int)
        parser.add_argument("--num-epochs", type=int)
        parser.add_argument("--lr", type=float)
        parser.add_argument("--char-embedding-size", help="Set to 0 to disable char-level embeddings")
        parser.add_argument("--max-context-size", help="Trim all context values to this length during training (0 for unlimited)")
        parser.add_argument("--max-question-size", type=int, help="Trim all context values to this length during training (0 for unlimited)")
        parser.add_argument("--rnn-hidden-size", type=int)
        parser.add_argument("--rnn-num-layers", type=int)
        parser.add_argument("--rnn-unidirectional", action="store_true")
        parser.add_argument("--dropout", type=float)
        parser.add_argument("--loader-num-workers", type=int, help="number of worker processes to use for DataLoader")
        parser.add_argument("--debug", action="store_true", help="if specified debug by fitting a single batch and profiling")
        parser.add_argument("--multi-answer", action="store_true", help="if specified don't truncate answer spans down to one")
        parser.add_argument("--no-self-attention", action="store_true", help="if specified don't use self attention")
        parser.add_argument("--disable-cuda", action="store_true", help="if specified don't use CUDA even if available")
        # fmt: on
        return parser.parse_known_args()[0]
