"""
Module that holds the training harness
"""

import json
from typing import Any, Dict, Optional

from tqdm import tqdm, trange
import torch as t
from torch.utils.data import DataLoader
import torch.optim as optim

from model.corpus import QADataset, TrainDataset, EvalDataset
from model.qa import QuestionId
from model.batcher import QABatch, get_collator
from model.predictor import PredictorModel, ModelPredictions

from model.util import get_device

import model.evaluator as evaluator
from model.evaluator import (Evaluator, MultiClassLossEvaluator,
                             SingleClassLossEvaluator)

import scripts.evaluate_v1_1 as evaluate_v1_1
import scripts.evaluate_v2_0 as evaluate_v2_0


def train_model(model: PredictorModel,
                train_dataset: TrainDataset,
                dev_dataset: EvalDataset,
                learning_rate: float,
                num_epochs: int,
                batch_size: int,
                use_cuda: bool = False,
                loader_num_workers: int=2,
                debug: bool = False,
                model_checkpoint_path: Optional[str] = None) -> None:
    """
    Trains a DocQAPredictor model on the given train set with given params and returns
    the trained model instance

    :param model: A PredictorModel to train the parameters of
    :param train_dataset: A Processed TrainDataset object of training data
    :param dev_dataset: A Processed EvalDataset object of dev data
    :param learning_rate: LR for Adam optimizer
    :param num_epochs: Number of epochs to train for
    :param batch_size: Size of each training batch
    :param use_cuda: If True use CUDA (default False)
    :param loader_num_workers: Number of workers to use for DataLoader (default 2)
    :param debug: If True train on a single batch and profile performance (default False)
    :param model_checkpoint_path: if specified path to save serialized model parameters to
        (default None-> no checkpoint serialization)

    :returns: A Trained PredictorModel object
    """

    device = get_device(use_cuda)
    train_evaluator: Evaluator
    if train_dataset.corpus.stats.single_answer:
        train_evaluator = SingleClassLossEvaluator()
    else:
        train_evaluator = MultiClassLossEvaluator()
    trainable_parameters = filter(lambda p: p.requires_grad,
                                  set(model.parameters()))
    optimizer: optim.Optimizer = optim.Adam(
        trainable_parameters, lr=learning_rate)
    loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=loader_num_workers,
        collate_fn=get_collator(device))
    if debug:
        debug_run(loader, model, optimizer, train_evaluator, use_cuda)
    else:
        training_run(loader, model, optimizer, train_evaluator, dev_dataset, num_epochs, use_cuda, model_checkpoint_path)


def one_train_iteration(optimizer, model, batch, evaluator):
    """
    Runs one train iteration of the given model on the given batch,
    evaluating using the given evaluator and updating parameters
    using the given optimizer.
    Returns the batch loss
    """
    optimizer.zero_grad()
    predictions: ModelPredictions = model(batch)
    loss = evaluator(batch, predictions)
    loss.backward()
    optimizer.step()
    batch_loss = loss.item()
    return batch_loss


def debug_run(loader, model, optimizer, evaluator, use_cuda: bool, num_epochs: int=1):
    """
    Runs the given model setup while profiling with the torch autograd profiler
    and tracking memory allocation of tensors across batches.
    """
    import gc

    def print_all_tensors(message: str):
        print(message)
        sizes = []
        for obj in gc.get_objects():
            try:
                if t.is_tensor(obj) or (hasattr(obj, 'data') and t.is_tensor(obj.data)):
                    print(type(obj), obj.size)
                    sizes.append(obj.size)
            except Exception:
                pass
            print("Total size of {} tensors allocated: {}".format(len(sizes), sum(sizes)))

    with t.autograd.profiler.profile(use_cuda=use_cuda) as prof:
        with trange(num_epochs) as epochs:
            for epoch in epochs:
                epochs.set_description('Epoch %d' % (epoch + 1))
                epoch_loss = 0.0
                with tqdm(loader) as batch_loop:
                    for batch_num, batch in enumerate(batch_loop):
                        batch_loop.set_description('Batch %d' % (batch_num + 1))
                        print_all_tensors("Before batch {}, allocated tensors:".format(batch_num + 1))
                        batch_loss = one_train_iteration(optimizer, model, batch, evaluator)
                        print_all_tensors("After batch {}, allocated tensors:".format(batch_num + 1))
                        epoch_loss += batch_loss
                        batch_loop.set_postfix(loss=batch_loss)
    print("Debug run complete, printing CPU profile")
    prof.table(sort_by='cpu_time_total')
    print("Debug run complete, printing CUDA profile")
    prof.table(sort_by='cuda_time_total')


def training_run(loader, model, optimizer, evaluator, dev_dataset, num_epochs, use_cuda, model_checkpoint_path):
    """
    Trains the given model over the entire data loader for as many epochs as specified, validating on dev
    after every epoch and saving the model to disk after every epoch
    """
    with trange(num_epochs) as epochs:
        for epoch in epochs:
            epochs.set_description('Epoch %d' % (epoch + 1))
            epoch_loss = 0.0
            with tqdm(loader) as batch_loop:
                for batch_num, batch in enumerate(batch_loop):
                    batch_loop.set_description('Batch %d' % (batch_num + 1))
                    batch_loss = one_train_iteration(optimizer, model, batch, evaluator)
                    epoch_loss += batch_loss
                    batch_loop.set_postfix(loss=batch_loss)
            epoch_loss = epoch_loss / len(loader)
            epochs.set_postfix(loss=epoch_loss)
            validate(dev_dataset, model, evaluator, use_cuda, epoch)
            print(
                'Saving model checkpoint to {}'.format(model_checkpoint_path))
            t.save(model, model_checkpoint_path)


def validate(dataset: QADataset,
             predictor: PredictorModel,
             evaluator: Any,
             use_cuda: bool,
             epoch: int = 0,
             batch_size: int = 64) -> None:
    """
    Validates the given model over the given dataset, both using the official
    SQuAD evaluation script to obtain F1 and EM scores and using the evaluator
    to get loss values
    """
    print('\n=== EPOCH {}: Measuring QA performance on the dev set\n'.format(
        epoch + 1))
    try:
        dev_perf = evaluate_on_squad_dataset(dataset, predictor, use_cuda,
                                             batch_size)
        print('\n=== Dev set performance: {}\n'.format(json.dumps(dev_perf)))
    except Exception as err:
        print('\nError when trying to get full evaluation: {}\n'.format(err))
    print('\n=== EPOCH %d: Measuring loss on the dev set\n'.format(epoch + 1))
    dev_loss = get_dataset_loss(dataset, predictor, evaluator, use_cuda,
                                batch_size)
    print('\n=== Dev set loss: {}\n'.format(dev_loss))


def get_dataset_loss(dataset: QADataset,
                     predictor: PredictorModel,
                     evaluator: Any,
                     use_cuda: bool,
                     batch_size: int = 64) -> float:
    """
    Computes total loss of the model over the entire dataset
    """
    device = get_device(use_cuda)
    loader: DataLoader = DataLoader(
        dataset, batch_size, collate_fn=get_collator(device))
    total_loss = 0.0
    batch: QABatch
    for batch in tqdm(loader, desc='Loss computation batch'):
        with t.no_grad():
            batch.to(device)
            predictions: ModelPredictions = predictor(batch)
            total_loss += evaluator(batch, predictions).item()
    return total_loss


def answer_dataset(dataset: QADataset,
                   predictor: PredictorModel,
                   use_cuda: bool,
                   batch_size: int = 64) -> Dict[QuestionId, str]:
    """
    Generates well-formatted answers for the given dataset using the
    given model.
    """
    device = get_device(use_cuda)
    loader: DataLoader = DataLoader(
        dataset, batch_size, collate_fn=get_collator(device))
    batch: QABatch
    qid_to_answer: Dict[QuestionId, str] = dict()
    for batch_num, batch in enumerate(
            tqdm(loader, desc='Answer generation batch')):
        with t.no_grad():
            batch.to(device)
            predictions: ModelPredictions = predictor(batch)
            qid_to_answer.update(
                evaluator.get_answer_token_idxs(batch, predictions))
    return dataset.get_answer_texts(qid_to_answer)


def evaluate_on_squad_dataset(dataset: QADataset,
                              predictor: PredictorModel,
                              use_cuda: bool,
                              batch_size: int = 64) -> Dict[str, str]:
    """
    Generates well formatted answers for the given dataset using the given
    model, then runs the official SQuAD evaluation script on it to obtain
    f1 and em scores.
    """
    answer_dict = answer_dataset(dataset, predictor, use_cuda, batch_size)
    with open(dataset.source_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset_version = dataset_json['version']
        if dataset_version == '1.1':
            eval_fn = evaluate_v1_1.evaluate
        elif dataset_version == '2.0':
            eval_fn = evaluate_v2_0.evaluate
        else:
            raise Exception(
                'Dataset version malformed: {}'.format(dataset_version))
        dataset_dict = dataset_json['data']
    return eval_fn(dataset_dict, answer_dict)
