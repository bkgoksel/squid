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
    weight_decay: float
    max_grad_norm: float
    char_embedding_size: int
    attention_linear_hidden_size: int
    rnn_hidden_size: int
    rnn_num_layers: int
    max_context_size: int
    max_question_size: int
    loader_num_workers: int
    dropout: float
    run_name: str
    rnn_unidirectional: bool
    debug: bool
    multi_answer: bool
    no_self_attention: bool
    disable_cuda: bool

    DEFAULT_ARGS = {
        "train_file": "data/original/train.json",
        "dev_file": "data/original/dev.json",
        "word_vector_file": "data/word-vectors/glove/glove.6B.100d.txt",
        "batch_size": 45,
        "num_epochs": 30,
        "lr": 1e-3,
        "weight_decay": 0,
        "max_grad_norm": 10,
        "char_embedding_size": 50,
        "attention_linear_hidden_size": 200,
        "rnn_hidden_size": 200,
        "rnn_num_layers": 2,
        "max_context_size": 400,
        "max_question_size": 100,
        "loader_num_workers": 2,
        "dropout": 0.1,
        "config_file": "train-config.json",
        "run_name": "train-run",
        "rnn_unidirectional": False,
        "debug": False,
        "multi_answer": False,
        "no_self_attention": False,
        "disable_cuda": False,
    }

    def __init__(self, arg_dict: Dict[str, Any]) -> None:
        # Don't format args
        # fmt: off
        self.train_file = arg_dict["train_file"]
        self.dev_file = arg_dict["dev_file"]
        self.word_vector_file = arg_dict["word_vector_file"]
        self.batch_size = arg_dict["batch_size"]
        self.num_epochs = arg_dict["num_epochs"]
        self.lr = arg_dict["lr"]
        self.weight_decay = arg_dict["weight_decay"]
        self.max_grad_norm = arg_dict["max_grad_norm"]
        self.char_embedding_size = arg_dict["char_embedding_size"]
        self.attention_linear_hidden_size = arg_dict["attention_linear_hidden_size"]
        self.rnn_hidden_size = arg_dict["rnn_hidden_size"]
        self.rnn_num_layers = arg_dict["rnn_num_layers"]
        self.max_context_size = arg_dict["max_context_size"]
        self.max_question_size = arg_dict["max_question_size"]
        self.loader_num_workers = arg_dict["loader_num_workers"]
        self.dropout = arg_dict["dropout"]
        self.run_name = arg_dict["run_name"]
        self.rnn_unidirectional = arg_dict["rnn_unidirectional"]
        self.debug = arg_dict["debug"]
        self.multi_answer = arg_dict["multi_answer"]
        self.no_self_attention = arg_dict["no_self_attention"]
        self.disable_cuda = arg_dict["disable_cuda"]
        # fmt: on

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
        parser.add_argument("--weight-decay", type=float, help="weight decay (L2 penalty) to use during training")
        parser.add_argument("--max-grad-norm", type=float, help="Maximum norm to use for gradient clipping (default None-> no gradient clipping)")
        parser.add_argument("--char-embedding-size", help="Set to 0 to disable char-level embeddings")
        parser.add_argument("--max-context-size", help="Trim all context values to this length during training (0 for unlimited)")
        parser.add_argument("--max-question-size", type=int, help="Trim all context values to this length during training (0 for unlimited)")
        parser.add_argument("--attention-linear-hidden-size", type=int)
        parser.add_argument("--rnn-hidden-size", type=int)
        parser.add_argument("--rnn-num-layers", type=int)
        parser.add_argument("--dropout", type=float)
        parser.add_argument("--loader-num-workers", type=int, help="number of worker processes to use for DataLoader")
        parser.add_argument("--rnn_unidirectional ", action="store_true", help="if specified make all RNNs unidirectional instead of bidirectional")
        parser.add_argument("--debug", action="store_true", help="if specified debug by fitting a single batch and profiling")
        parser.add_argument("--multi-answer", action="store_true", help="if specified don't truncate answer spans down to one")
        parser.add_argument("--no-self-attention", action="store_true", help="if specified don't use self attention")
        parser.add_argument("--disable-cuda", action="store_true", help="if specified don't use CUDA even if available")
        # fmt: on
        return parser.parse_known_args()[0]

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
        config_file = (
            cls.DEFAULT_ARGS["config_file"]
            if cli_args.config_file is None
            else cli_args.config_file
        )
        cli_arg_dict = dict(
            (arg, val) for (arg, val) in vars(cli_args).items() if val is not None
        )
        try:
            config_file_args = json.load(open(config_file, "r"))
        except (IOError, json.JSONDecodeError) as e:
            print(f"Cannot load {cli_args.config_file}: {e}, continuing")
            config_file_args = {}
        args.update(config_file_args)
        args.update(cli_arg_dict)
        return TrainArgs(args)
