import json
from typing import Tuple, Optional

import torch as t

from model.trainer import Trainer
from model.train_parser import TrainArgs
from model.tokenizer import Tokenizer, NltkTokenizer
from model.text_processor import TextProcessor
from model.predictor import (
    DocQAConfig,
    GRUConfig,
    PredictorModel,
    DocQAPredictor,
    BidafPredictor,
)
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


def initialize_model(
    args: TrainArgs, train_dataset: TrainDataset, vectors: WordVectors
) -> PredictorModel:
    """
    Given the command line args and the dataset and vectors to train on,
    initializes a new model
    :param args: TrainArgs containing setting for invocation
    :param train_dataset: TrainDataset object to compute vocabulary on
    :param vectors: WordVectors object to use in the model
    :returns: A new PredictorModel
    """
    device = get_device(args.disable_cuda)
    predictor_config = DocQAConfig(
        gru=GRUConfig(
            args.rnn_hidden_size,
            args.rnn_num_layers,
            args.dropout,
            args.rnn_unidirectional,
        ),
        dropout_prob=args.dropout,
        attention_linear_hidden_size=args.attention_linear_hidden_size,
        use_self_attention=(not args.no_self_attention),
        batch_size=args.batch_size,
    )
    word_embedding_config = WordEmbeddorConfig(vectors=vectors, train_vecs=False)
    char_embedding_config: Optional[PoolingCharEmbeddorConfig]
    if args.char_embedding_size:
        char_embedding_config = PoolingCharEmbeddorConfig(
            embedding_dimension=args.char_embedding_size,
            char_vocab_size=train_dataset.corpus.stats.char_vocab_size,
        )
    else:
        char_embedding_config = None

    embeddor_config = EmbeddorConfig(
        word_embeddor=word_embedding_config,
        char_embeddor=char_embedding_config,
        highway_layers=args.highway_layers,
    )
    embeddor: Embeddor = make_embeddor(embeddor_config, device)
    if args.simple_bidaf:
        predictor: PredictorModel = BidafPredictor(embeddor, predictor_config.gru).to(
            device
        )
    else:
        predictor = DocQAPredictor(embeddor, predictor_config).to(device)
    return predictor


def get_datasets(args: TrainArgs) -> Tuple[TrainDataset, EvalDataset, WordVectors]:
    """
    Returns the Datasets to be used for training (and the word vectors used to embed them,
     parsed form the args
    :param args: CLI args to get the datasets
    :returns: a Tuple of the training dataset and the dev dataset and the word vectors
    """
    device = get_device(args.disable_cuda)

    tokenizer: Tokenizer = NltkTokenizer()
    processor: TextProcessor = TextProcessor({"lowercase": True})
    vectors: WordVectors = WordVectors.load_vectors(args.word_vector_file)

    train_dataset: TrainDataset = TrainDataset.load_dataset(
        args.train_file,
        vectors,
        tokenizer,
        processor,
        force_single_answer=not args.multi_answer,
    )
    dev_dataset: EvalDataset = EvalDataset.load_dataset(
        args.dev_file, vectors, train_dataset.char_mapping, tokenizer, processor
    )
    return train_dataset, dev_dataset, vectors


def get_training_config(args: TrainArgs) -> Trainer.TrainingConfig:
    """
    Parse the command line args builds a TrainingConfig object
    :param args: TrainArgs object containing invocation parameters
    :returns: A well formatted TrainingConfig object that can be used
        for training the model
    """
    return Trainer.TrainingConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_question_size=args.max_question_size,
        max_context_size=args.max_context_size,
        device=get_device(args.disable_cuda),
        loader_num_workers=args.loader_num_workers,
        model_checkpoint_path=args.run_name,
    )


def main() -> None:
    args = TrainArgs.get_args()
    train_dataset, dev_dataset, vectors = get_datasets(args)
    training_config = get_training_config(args)
    with open(f"{args.run_name}_config.json", "w") as config_file:
        json.dump(vars(args), config_file, indent=2)

    try:
        print(f"Attempting to load model to train from {args.run_name}.pth")
        model = t.load(f"{args.run_name}.pth").to(training_config.device)
    except IOError as e:
        print(f"Can't load model: {e}, initializing from scratch")
        model = initialize_model(args, train_dataset, vectors)

    Trainer.train_model(
        model, train_dataset, dev_dataset, training_config, debug=args.debug
    )
    dev_answers = Trainer.answer_dataset(dev_dataset, model, training_config)
    gold_answers = dev_dataset.get_gold_answers()
    qid_to_answers = {}
    for qid, model_answer in dev_answers.items():
        qid_to_answers[qid] = {
            "model_answer": model_answer,
            "gold_answer": gold_answers[qid],
        }
    with open("dev-pred.json", "w") as f:
        json.dump(dev_answers, f)
    with open("dev-pred-with-gold.json", "w") as f:
        json.dump(qid_to_answers, f)
    print("Final evaluation on dev")
    eval_results = Trainer.evaluate_on_squad_dataset(
        dev_dataset, model, training_config
    )
    print(eval_results)

    print(f"Saving model to {args.run_name}.pth")
    t.save(model, f"{args.run_name}.pth")


if __name__ == "__main__":
    main()
