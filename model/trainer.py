"""
Module that holds the training harness
"""

import json
from typing import Dict, NamedTuple

from tqdm import tqdm, trange
import torch as t
from torch.utils.data import DataLoader
import torch.optim as optim

from model.corpus import QADataset, TrainDataset, EvalDataset
from model.qa import QuestionId
from model.batcher import QABatch, get_collator
from model.predictor import PredictorModel, ModelPredictions
from model.profiler import memory_profiled, autograd_profiled

from model.evaluator import (
    Evaluator,
    MultiClassLossEvaluator,
    SingleClassLossEvaluator,
    get_answer_token_idxs,
)

import scripts.evaluate_v1_1 as evaluate_v1_1
import scripts.evaluate_v2_0 as evaluate_v2_0


class Trainer:
    """
    Class holding all utilities and functions related to training QA models
    """

    """
    Config for training runs:
        :learning_rate: LR for Adam optimizer
        :num_epochs: Number of epochs to train for
        :batch_size: Size of each training batch
        :max_question_size: Trims longer questions to this length
        :max_context_size: Trims longer contexts to this length
        :device: Torch device to use for training
        :loader_num_workers: Number of workers to use for DataLoader
        :model_checkpoint_path: Path to save serialized model parameters to
    """
    TrainingConfig = NamedTuple(
        "TrainingConfig",
        [
            ("learning_rate", float),
            ("num_epochs", int),
            ("batch_size", int),
            ("max_question_size", int),
            ("max_context_size", int),
            ("device", t.device),
            ("loader_num_workers", int),
            ("model_checkpoint_path", str),
        ],
    )

    @classmethod
    def one_train_iteration(
        cls,
        batch: QABatch,
        model: PredictorModel,
        evaluator: Evaluator,
        optimizer: t.optim.Optimizer,
    ):
        """
        Runs one train iteration of the given model on the given batch,
        evaluating using the given evaluator and updating parameters
        using the given optimizer.
        :param batch: Batch to train on
        :param model: Model to train
        :param evaluator: Evaluator to compute loss
        :param optimizer: Optimizer to step over model
        :returns: Total loss for batch
        """
        optimizer.zero_grad()
        predictions: ModelPredictions = model(batch)
        loss = evaluator(batch, predictions)
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        return batch_loss

    @classmethod
    def training_run(
        cls,
        loader: DataLoader,
        model: PredictorModel,
        evaluator: Evaluator,
        optimizer: t.optim.Optimizer,
        dev_dataset: EvalDataset,
        training_config: TrainingConfig,
    ) -> None:
        """
        Trains the given model over the entire data loader for as many epochs as specified, validating on dev
        after every epoch and saving the model to disk after every epoch
        :param loader: DataLoader to load the batches
        :param model: Model to train
        :param evaluator: Evaluator to compute loss
        :param optimizer: Optimizer to step over model
        :param dev_dataset: EvalDataset to validate on
        :param training_config: TrainingConfig object describing parameters for training
        """
        with trange(training_config.num_epochs) as epochs:
            for epoch in epochs:
                epochs.set_description("Epoch %d" % (epoch + 1))
                epoch_loss = 0.0
                with tqdm(loader) as batch_loop:
                    for batch_num, batch in enumerate(batch_loop):
                        batch.to(training_config.device)
                        batch_loop.set_description("Batch %d" % (batch_num + 1))
                        batch_loss = cls.one_train_iteration(
                            batch, model, evaluator, optimizer
                        )
                        epoch_loss += batch_loss
                        batch_loop.set_postfix(loss=batch_loss)
                epoch_loss = epoch_loss / len(loader)
                epochs.set_postfix(loss=epoch_loss)
                cls.validate(dev_dataset, model, evaluator, training_config, epoch)
                print(
                    "Saving model checkpoint to {}".format(
                        training_config.model_checkpoint_path
                    )
                )
                t.save(model, training_config.model_checkpoint_path)

    @classmethod
    def train_model(
        cls,
        model: PredictorModel,
        train_dataset: TrainDataset,
        dev_dataset: EvalDataset,
        training_config: TrainingConfig,
        debug: bool = False,
    ) -> None:
        """
        Trains a DocQAPredictor model on the given train set with given params and returns
        the trained model instance

        :param model: A PredictorModel to train the parameters of
        :param train_dataset: A TrainDataset object of training data
        :param dev_dataset: An EvalDataset object of dev data
        :param training_config: TrainingConfig object describing parameters of training run
        :param debug: If True train on a single batch and profile performance (default False)

        :returns: A Trained PredictorModel object
        """

        train_evaluator: Evaluator
        if train_dataset.corpus.stats.single_answer:
            train_evaluator = SingleClassLossEvaluator()
        else:
            train_evaluator = MultiClassLossEvaluator()
        trainable_parameters = filter(
            lambda p: p.requires_grad, set(model.parameters())
        )
        optimizer: optim.Optimizer = optim.Adadelta(
            trainable_parameters, lr=training_config.learning_rate
        )
        loader: DataLoader = DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=training_config.loader_num_workers,
            collate_fn=get_collator(
                training_config.max_question_size, training_config.max_context_size
            ),
        )
        if debug:
            # Wrap in profiler
            unwrapped_iteration = cls.one_train_iteration
            unwrapped_train_run = cls.training_run
            setattr(
                cls, "one_train_iteration", memory_profiled(cls.one_train_iteration)
            )
            setattr(cls, "training_run", autograd_profiled(cls.training_run))
        cls.training_run(
            loader, model, train_evaluator, optimizer, dev_dataset, training_config
        )
        if debug:
            setattr(cls, "one_train_iteration", unwrapped_iteration)
            setattr(cls, "training_run", unwrapped_train_run)

    @classmethod
    def validate(
        cls,
        dataset: QADataset,
        model: PredictorModel,
        evaluator: Evaluator,
        training_config: TrainingConfig,
        epoch: int = 0,
    ) -> None:
        """
        Validates the given model over the given dataset, both using the official
        SQuAD evaluation script to obtain F1 and EM scores and using the evaluator
        to get loss values
        :param dataset: QADataset object to validate on
        :param model: PredictorModel to validate
        :param evaluator: Evaluator to compute loss
        :param epoch: Current epoch number for logging
        :param training_config: Training config to pull parameters from
        """
        print(
            "\n=== EPOCH {}: Measuring QA performance on the dev set\n".format(
                epoch + 1
            )
        )
        try:
            dev_perf = cls.evaluate_on_squad_dataset(dataset, model, training_config)
            print("\n=== Dev set performance: {}\n".format(json.dumps(dev_perf)))
        except Exception as err:
            print("\nError when trying to get full evaluation: {}\n".format(err))
        print("\n=== EPOCH %d: Measuring loss on the dev set\n".format(epoch + 1))
        dev_loss = cls.get_dataset_loss(dataset, model, evaluator, training_config)
        print("\n=== Dev set loss: {}\n".format(dev_loss))

    @classmethod
    def get_dataset_loss(
        cls,
        dataset: QADataset,
        model: PredictorModel,
        evaluator: Evaluator,
        training_config: TrainingConfig,
    ) -> float:
        """
        Computes total loss of the model over the entire dataset
        :param dataset: QADataset object to validate on
        :param model: PredictorModel to validate
        :param evaluator: Evaluator to compute loss
        :param training_config: Training config to pull parameters from
        """
        loader: DataLoader = DataLoader(
            dataset,
            training_config.batch_size,
            collate_fn=get_collator(
                training_config.max_question_size, training_config.max_context_size
            ),
        )
        total_loss = 0.0
        batch: QABatch
        for batch in tqdm(loader, desc="Loss computation batch"):
            with t.no_grad():
                batch.to(training_config.device)
                predictions: ModelPredictions = model(batch)
                total_loss += evaluator(batch, predictions).item()
        return total_loss

    @classmethod
    def answer_dataset(
        cls, dataset: QADataset, model: PredictorModel, training_config: TrainingConfig
    ) -> Dict[QuestionId, str]:
        """
        Generates well-formatted answers for the given dataset using the
        given model.
        :param dataset: QADataset object to validate on
        :param model: PredictorModel to validate
        :param training_config: Training config to pull parameters from
        """
        loader: DataLoader = DataLoader(
            dataset,
            training_config.batch_size,
            collate_fn=get_collator(
                training_config.max_question_size, training_config.max_context_size
            ),
        )
        batch: QABatch
        qid_to_answer: Dict[QuestionId, str] = dict()
        for batch_num, batch in enumerate(tqdm(loader, desc="Answer generation batch")):
            with t.no_grad():
                batch.to(training_config.device)
                predictions: ModelPredictions = model(batch)
                qid_to_answer.update(get_answer_token_idxs(batch, predictions))
        return dataset.get_answer_texts(qid_to_answer)

    @classmethod
    def evaluate_on_squad_dataset(
        cls, dataset: QADataset, model: PredictorModel, training_config: TrainingConfig
    ) -> Dict[str, str]:
        """
        Generates well formatted answers for the given dataset using the given
        model, then runs the official SQuAD evaluation script on it to obtain
        f1 and em scores.
        :param dataset: QADataset object to validate on
        :param model: PredictorModel to validate
        :param training_config: Training config to pull parameters from
        """
        answer_dict = cls.answer_dataset(dataset, model, training_config)
        with open(dataset.source_file) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset_version = dataset_json["version"]
            if dataset_version == "1.1":
                eval_fn = evaluate_v1_1.evaluate
            elif dataset_version == "2.0":
                eval_fn = evaluate_v2_0.evaluate
            else:
                raise Exception("Dataset version malformed: {}".format(dataset_version))
            dataset_dict = dataset_json["data"]
        return eval_fn(dataset_dict, answer_dict)
