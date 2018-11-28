"""
Module that holds the training harness
"""

import json
from typing import (
    Any,
    Callable,
    List,
    Dict,
    Tuple,
    Iterable,
    NamedTuple,
    cast,
    Optional,
)

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

import scripts.evaluate_v1_1 as evaluate_v1_1  # type: ignore
import scripts.evaluate_v2_0 as evaluate_v2_0  # type: ignore


class Trainer:
    """
    Class holding all utilities and functions related to training QA models
    """

    """
    Config for training runs:
        :learning_rate: lr for adam optimizer
        :weight_decay: weight decay to use in the optimizer
        :max_grad_norm: Maximum gradient norm for gradient clipping
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
            ("weight_decay", float),
            ("max_grad_norm", float),
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
        parameters: Iterable[Any],
        evaluator: Evaluator,
        optimizer: t.optim.Optimizer,
        max_grad_norm: Optional[float] = None,
    ) -> float:
        """
        Runs one train iteration of the given model on the given batch,
        evaluating using the given evaluator and updating parameters
        using the given optimizer.
        :param batch: Batch to train on
        :param model: Model to train
        :param parameters: Parameters of model to train
        :param evaluator: Evaluator to compute loss
        :param optimizer: Optimizer to step over model
        :param max_grad_norm: If specified clip gradients at this norm
        :returns: Total loss for batch
        """
        predictions: ModelPredictions = model(batch)
        loss = evaluator(batch, predictions)
        model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm:
            t.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        optimizer.step()
        batch_loss = loss.item()
        return cast(float, batch_loss)

    @classmethod
    def training_run(
        cls,
        loader: DataLoader,
        model: PredictorModel,
        parameters: Iterable[Any],
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
        :param parameters: Parameters of model to train
        :param evaluator: Evaluator to compute loss
        :param optimizer: Optimizer to step over model
        :param dev_dataset: EvalDataset to validate on
        :param training_config: TrainingConfig object describing parameters for training
        """
        epoch_losses: List[float] = []
        dev_losses: List[float] = []
        dev_f1s: List[float] = []
        dev_ems: List[float] = []
        with trange(training_config.num_epochs) as epochs:
            for epoch in epochs:
                model.train()
                epochs.set_description("Epoch %d" % (epoch + 1))
                epoch_loss = 0.0
                with tqdm(loader) as batch_loop:
                    for batch_num, batch in enumerate(batch_loop):
                        batch.to(training_config.device)
                        batch_loop.set_description("Batch %d" % (batch_num + 1))
                        batch_loss = cls.one_train_iteration(
                            batch,
                            model,
                            parameters,
                            evaluator,
                            optimizer,
                            training_config.max_grad_norm,
                        ) / len(batch)
                        epoch_loss += batch_loss
                        batch_loop.set_postfix(loss=batch_loss)
                epoch_loss = epoch_loss / len(loader)
                dev_loss, dev_f1, dev_em = cls.validate(
                    dev_dataset, model, evaluator, training_config, epoch
                )
                epoch_losses.append(epoch_loss)
                dev_losses.append(dev_loss)
                dev_f1s.append(dev_f1)
                dev_ems.append(dev_em)
                epochs.set_postfix(loss=epoch_loss, f1=dev_f1, em=dev_em)
                print(
                    f"Saving model checkpoint to {training_config.model_checkpoint_path}"
                )
                save_path = (
                    training_config.model_checkpoint_path + ".pth"
                    if not training_config.model_checkpoint_path.endswith(".pth")
                    else training_config.model_checkpoint_path
                )
                t.save(model, save_path)
                run_stats_dict = {
                    "current_epoch": epoch,
                    "current_epoch_loss": epoch_loss,
                    "current_dev_loss": dev_loss,
                    "current_dev_f1": dev_f1,
                    "current_dev_em": dev_em,
                    "all_epoch_losses": epoch_losses,
                    "all_dev_losses": dev_losses,
                    "all_dev_f1s": dev_f1s,
                    "all_dev_ems": dev_ems,
                }
                with open("run-stats.json", "w") as stats_file:
                    json.dump(run_stats_dict, stats_file)

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
        :param debug: If True profile performance (default False)

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
        optimizer: optim.Optimizer = optim.Adam(
            trainable_parameters,
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
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
            setattr(
                cls,
                "training_run",
                autograd_profiled(
                    cls.training_run,
                    use_cuda=training_config.device == t.device("cuda"),
                ),
            )
        cls.training_run(
            loader,
            model,
            trainable_parameters,
            train_evaluator,
            optimizer,
            dev_dataset,
            training_config,
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
    ) -> Tuple[float, float, float]:
        """
        Validates the given model over the given dataset, both using the official
        SQuAD evaluation script to obtain F1 and EM scores and using the evaluator
        to get loss values
        :param dataset: QADataset object to validate on
        :param model: PredictorModel to validate
        :param evaluator: Evaluator to compute loss
        :param epoch: Current epoch number for logging
        :param training_config: Training config to pull parameters from
        :returns: A Tuple of average dev set loss, F1 and EM performance
        """
        model.eval()
        print(f"\n=== EPOCH {epoch + 1}: Validating on dev set\n\n")
        f1: float = 0.0
        em: float = 0.0
        try:
            dev_perf = cls.evaluate_on_squad_dataset(dataset, model, training_config)
            f1 = float(dev_perf.get("f1", 0.0))
            em = float(dev_perf.get("em", 0.0))
        except Exception as err:
            print(f"Error when trying to get full evaluation: {err}")
        dev_loss = cls.get_dataset_loss(dataset, model, evaluator, training_config)
        print(f"\nDev set loss: {dev_loss}, F1: {f1}, EM: {em}\n\n")
        return (dev_loss, f1, em)

    @classmethod
    def get_dataset_loss(
        cls,
        dataset: QADataset,
        model: PredictorModel,
        evaluator: Evaluator,
        training_config: TrainingConfig,
    ) -> float:
        """
        Computes the average loss of the model over the entire dataset
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
        return total_loss / len(dataset)

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
        qid_to_answer: Dict[QuestionId, Tuple[Any, ...]] = dict()
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
        try:
            answer_dict = cls.answer_dataset(dataset, model, training_config)
        except Exception as ex:
            raise Exception(f"Can't answer dataset: {ex}")
        with open(dataset.source_file) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset_version = dataset_json["version"]
            eval_fn: Callable[[Dict[Any, Any], Dict[QuestionId, str]], Dict[str, str]]
            if dataset_version == "1.1":
                eval_fn = evaluate_v1_1.evaluate
            elif dataset_version == "2.0":
                eval_fn = evaluate_v2_0.evaluate
            else:
                raise Exception("Dataset version malformed: {}".format(dataset_version))
            dataset_dict = dataset_json["data"]
        return eval_fn(dataset_dict, answer_dict)
