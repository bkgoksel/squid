"""
Module that deals with preparing QA corpora
"""

import json
import pickle
from typing import Any, Optional, List, Dict, Set, Tuple, NamedTuple, cast
from collections import defaultdict

from torch.utils.data import Dataset

from model.text_processor import TextProcessor
from model.tokenizer import Tokenizer
from model.qa import (
    Answer,
    QuestionAnswer,
    ContextQuestionAnswer,
    EncodedContextQuestionAnswer,
    EncodedSample,
    QuestionId,
)

from model.wv import WordVectors

CorpusStats = NamedTuple(
    "CorpusStats",
    [
        ("n_contexts", int),
        ("n_questions", int),
        ("n_answerable", int),
        ("n_unanswerable", int),
        ("max_context_len", int),
        ("max_q_len", int),
        ("max_word_len", int),
        ("single_answer", bool),
        ("word_vocab_size", int),
        ("char_vocab_size", int),
    ],
)


class Corpus:
    """
    Class that contains a corpus
    """

    source_file: Optional[str]
    context_qas: List[ContextQuestionAnswer]
    quids_to_context_qas: Dict[QuestionId, ContextQuestionAnswer]
    token_mapping: Dict[str, int]
    char_mapping: Dict[str, int]
    stats: CorpusStats

    def __init__(
        self,
        context_qas: List[ContextQuestionAnswer],
        token_mapping: Dict[str, int],
        char_mapping: Dict[str, int],
        stats: CorpusStats,
        source_file: Optional[str] = None,
    ) -> None:
        self.source_file = source_file
        self.context_qas = context_qas
        self.quids_to_context_qas = {
            qa.question_id: cqa for cqa in context_qas for qa in cqa.qas
        }
        self.token_mapping = token_mapping
        self.char_mapping = char_mapping
        self.stats = stats

    @classmethod
    def from_disk(cls, serialized_file: str) -> "Corpus":
        """
        Loads a pickle serialized corpus from disk
        :param serialized_file: Name of the pickle file to load
        :returns: A Corpus object
        """
        with open(serialized_file, "rb") as f:
            return cast(Corpus, pickle.load(f))

    @classmethod
    def from_raw(
        cls,
        data_file: str,
        tokenizer: Tokenizer,
        processor: TextProcessor,
        word_vectors: WordVectors,
        force_single_answer: bool = False,
        char_mapping: Optional[Dict[str, int]] = None,
    ) -> "Corpus":
        """
        Reads a Corpus of QA questions from a file
        :param data_file: File to read from
        :param tokenizer: Tokenizer to tokenize all text read
        :param processor: TextProcessor object that contains all textual processing
            to be applied to the text before tokenization
        :param force_single_answer: if True only include first answer span as true
            (default False)
        :param word_vectors: WordVectors to build encoding indices from
        :param char_mapping: Optional mapping from chars to ints, will be computed
            from scratch if not specified
        """
        context_qas = cls.read_context_qas(
            data_file, tokenizer, processor, force_single_answer
        )

        def return_one() -> int:
            return 1

        token_mapping = defaultdict(return_one, word_vectors.word_to_idx)
        if char_mapping is None:
            char_mapping = cls.compute_char_indices(context_qas)
        stats = cls.compute_stats(context_qas, token_mapping, char_mapping)
        return cls(context_qas, token_mapping, char_mapping, stats, data_file)

    @staticmethod
    def read_context_qas(
        data_file: str,
        tokenizer: Tokenizer,
        processor: TextProcessor,
        force_single_answer: bool,
    ) -> List[ContextQuestionAnswer]:
        """
        Reads a SQUAD formattted JSON file into ContextQuestionAnswer objects
        :param data_file: filename of the JSON questions file
        :param tokenizer: Tokenizer object to use to tokenize the text
        :param processor: TextProcessor object to process text before tokenization
        :param force_single_answer: Bool if True only pick first answer span
        :returns: List[ContextQuestionAnswer], list of all the contexts and questions
        """
        contexts: List[ContextQuestionAnswer] = []
        with open(data_file, "r") as f:
            json_dict = json.load(f)
            for doc in json_dict["data"]:
                for paragraph in doc["paragraphs"]:
                    context: str = paragraph["context"]
                    qas: List[QuestionAnswer] = []
                    for qa in paragraph["qas"]:
                        q_text: str = qa["question"]
                        q_id: QuestionId = cast(QuestionId, qa["id"])
                        answers: Set[Answer] = set()
                        for answer in qa["answers"]:
                            text: str = answer["text"]
                            span_start: int = answer["answer_start"]
                            tokenized_answer = Answer(
                                text, span_start, tokenizer, processor
                            )
                            answers.add(tokenized_answer)
                            if force_single_answer:
                                break
                        tokenized_question = QuestionAnswer(
                            q_id, q_text, answers, tokenizer, processor
                        )
                        qas.append(tokenized_question)
                    tokenized_context = ContextQuestionAnswer(
                        context, qas, tokenizer, processor
                    )
                    contexts.append(tokenized_context)
        return contexts

    @staticmethod
    def compute_char_indices(
        context_qas: List[ContextQuestionAnswer]
    ) -> Dict[str, int]:
        """
        Takes in a list of contexts and qas and returns a mapping from each char seen to an index
        :param context_qas: List[ContextQuestionAnswer] all the context qa's
        :returns: Dict[str, int] mapping from each character seen to an index
        """
        chars: Set[str] = set()
        for ctx in context_qas:
            for tok in ctx.tokens:
                chars.update(set(char for char in tok.word))
            for qa in ctx.qas:
                for tok in qa.tokens:
                    chars.update(set(char for char in tok.word))
        char_mapping: Dict[str, int] = {
            char: idx for idx, char in enumerate(chars, 2)
        }  # idx 1 reserved for UNK
        return char_mapping

    @staticmethod
    def compute_stats(
        context_qas: List[ContextQuestionAnswer],
        token_mapping: Dict[str, int],
        char_mapping: Dict[str, int],
    ) -> CorpusStats:
        """
        Method that computes statistics given list of context qas and vocab
        :param context_qas: List of contextQA objects
        :param vocab: set of strings that contains all tokens in vocab
        :returns: A CorpusStats object with stats of the corpus
        """
        n_contexts: int = len(context_qas)
        n_questions: int = sum(len(ctx.qas) for ctx in context_qas)
        n_answerable: int = sum(
            len([qa for qa in ctx.qas if qa.answers]) for ctx in context_qas
        )
        n_unanswerable: int = sum(
            len([qa for qa in ctx.qas if not qa.answers]) for ctx in context_qas
        )
        single_answer: bool = all(
            all(len(qa.answers) <= 1 for qa in ctx.qas) for ctx in context_qas
        )
        max_context_len: int = max(len(ctx.tokens) for ctx in context_qas)
        max_q_len: int = max(len(qa.tokens) for ctx in context_qas for qa in ctx.qas)

        max_num_answer_spans = 0
        max_word_len = 0
        for ctx in context_qas:
            max_curr_len = max(len(tok.word) for tok in ctx.tokens)
            max_word_len = max(max_curr_len, max_word_len)
            for qa in ctx.qas:
                max_curr_len = max(len(tok.word) for tok in qa.tokens)
                max_word_len = max(max_curr_len, max_word_len)
                max_num_answer_spans = max(max_num_answer_spans, len(qa.answers))

        return CorpusStats(
            n_contexts=n_contexts,
            n_questions=n_questions,
            n_answerable=n_answerable,
            n_unanswerable=n_unanswerable,
            max_context_len=max_context_len,
            max_q_len=max_q_len,
            max_word_len=max_word_len,
            single_answer=single_answer,
            word_vocab_size=max(token_mapping.values()) + 1,
            char_vocab_size=max(char_mapping.values()) + 1,
        )

    def get_single_answer_text(
        self, qid: QuestionId, span_start: int, span_end: int
    ) -> str:
        """
        Turns a single qid, span_start, span_end triplet into a string
        of concatenated tokens
        :param qid: The QuestionId of the question
        :param span_start: Model output index for answer start
        :param span_end: Model output index for answer end
        :returns: All tokens (inclusive) from the tokenized context,
            space separated
        """
        context = self.quids_to_context_qas[qid]
        try:
            assert span_start <= span_end
            start_idx = context.tokens[span_start].span[0]
            end_idx = context.tokens[min(span_end, len(context.tokens))].span[1]
            return context.original_text[
                start_idx : min(end_idx, len(context.original_text))
            ]
        except Exception as ex:
            print(
                f"Error while reconstructing answer. num tokens: {len(context.tokens)}, first token: {span_start}, last token: {span_end}"
            )
            print(
                f"text len: {len(context.text)}, first char: {context.tokens[span_start].span[0]}, last char: {context.tokens[span_end].span[1]}"
            )
            return ""

    def get_answer_texts(
        self, answer_token_idxs: Dict[QuestionId, Tuple[Any, ...]]
    ) -> Dict[QuestionId, str]:
        """
        Given a mapping from questions id's to answer token indices, returns
        a SQuAD eval script readable version of the answer
        :param answer_token_idxs: a Mapping from QuestionId's to tuples
            of integers where the first elem is the span start
            prediction of the model and the second elem is the span
            end prediction of the model
        :returns: A Mapping from QuestionId's to strings
        """
        return {
            qid: self.get_single_answer_text(qid, span_start, span_end)
            for qid, (span_start, span_end) in answer_token_idxs.items()
        }

    def get_gold_answers(self) -> Dict[QuestionId, str]:
        qid_to_gold_answer = {}
        for cqa in self.context_qas:
            for qa in cqa.qas:
                qid_to_gold_answer[qa.question_id] = list(qa.answers)[0].original_text
        return qid_to_gold_answer

    def save(self, file_name: str) -> None:
        """
        Serializes this corpus to file with file_name
        :param file_name: File name to save corpus to
        :returns: None
        """
        with open(file_name, "wb") as f:
            pickle.dump(self, f)


class EncodedCorpus(Corpus):
    """
    Class that holds a Corpus alongside token and char mappings
    """

    vocab: Set[str]
    token_mapping: Dict[str, int]
    char_mapping: Dict[str, int]
    stats: CorpusStats
    encoded_context_qas: List[EncodedContextQuestionAnswer]

    def __init__(self, corpus: Corpus) -> None:
        super().__init__(
            corpus.context_qas,
            corpus.token_mapping,
            corpus.char_mapping,
            corpus.stats,
            corpus.source_file,
        )
        self.encoded_context_qas = EncodedCorpus.encode(
            self.context_qas, self.token_mapping, self.char_mapping
        )

    @staticmethod
    def encode(
        context_qas: List[ContextQuestionAnswer],
        token_mapping: Dict[str, int],
        char_mapping: Dict[str, int],
    ) -> List[EncodedContextQuestionAnswer]:
        """
        Method that encodes all the given contextQA's
        :param context_qas: List of ContextQA objects
        :param token_mapping: Dictionary from tokens to indices
        :param char_mapping: Dictionary from characters to indices
        :returns: List of EncodedContextQuestionAnswer objects
        """
        return [
            EncodedContextQuestionAnswer(cqa, token_mapping, char_mapping)
            for cqa in context_qas
        ]


class SampleCorpus(EncodedCorpus):
    """
    Class that stores a corpus of <context, question, answers> samples
    """

    context_qas: List[ContextQuestionAnswer]
    stats: CorpusStats
    encoded_context_qas: List[EncodedContextQuestionAnswer]
    samples: List[EncodedSample]
    n_samples: int

    def __init__(self, corpus: Corpus) -> None:
        super().__init__(corpus)
        self.samples = SampleCorpus.make_samples(self.encoded_context_qas)
        self.n_samples = len(self.samples)

    @staticmethod
    def make_samples(
        context_qas: List[EncodedContextQuestionAnswer]
    ) -> List[EncodedSample]:
        """
        Method that converts a list of EncodedContextQA objects that store
        a given context with all its methods into EncodedSample objects
        that store each question with its context and answers
        :param context_qas: List of EncodedContextQuestionAnswer objects
        :returns: List of EncodedSample objects
        """
        return [
            EncodedSample(ctx.word_encoding, ctx.char_encoding, qa)
            for ctx in context_qas
            for qa in ctx.qas
        ]


class QADataset(Dataset):
    """
    Class that turns a SampleCorpus into a PyTorch Dataset
    Main difference is that __getitem__ returns a dict instead of an EncodedSample
    """

    corpus: SampleCorpus
    _source_file: Optional[str]

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = SampleCorpus(corpus)
        self._source_file = self.corpus.source_file

    def __len__(self) -> int:
        return self.corpus.n_samples

    def __getitem__(self, idx: int) -> EncodedSample:
        return self.corpus.samples[idx]

    def get_gold_answers(self) -> Dict[QuestionId, str]:
        return self.corpus.get_gold_answers()

    def get_answer_texts(
        self, answer_token_idxs: Dict[QuestionId, Tuple[Any, ...]]
    ) -> Dict[QuestionId, str]:
        """
        Given a mapping from questions id's to answer token indices, returns
        a SQuAD eval script readable version of the answer
        :param answer_token_idxs: a Mapping from QuestionId's to tuples
            of integers where the first elem is the span start
            prediction of the model and the second elem is the span
            end prediction of the model
        :returns: A Mapping from QuestionId's to strings
        """
        return self.corpus.get_answer_texts(answer_token_idxs)

    @property
    def stats(self) -> CorpusStats:
        return self.corpus.stats

    @property
    def source_file(self) -> str:
        if self._source_file is not None:
            return self._source_file
        else:
            raise Exception("Dataset file provenance not available for this dataset")


class TrainDataset(QADataset):
    """
    Class that holds a dataset used for training a model
    (and therefore is the ground truth for character -> id mappinggs)
    """

    token_mapping: Dict[str, int]
    char_mapping: Dict[str, int]

    def __init__(self, corpus: Corpus) -> None:
        super().__init__(corpus)
        self.token_mapping = corpus.token_mapping
        self.char_mapping = corpus.char_mapping

    @classmethod
    def load_dataset(
        cls,
        filename: str,
        vectors: WordVectors,
        tokenizer: Tokenizer,
        processor: TextProcessor,
        force_single_answer: bool = True,
    ) -> QADataset:
        """
        Reads the given qa data file and processes it into a TrainDataset using
        the provided word vectors' vocab, tokenizer and text processor
        :param filename: File that contains the QA data
        :param vectors: WordVectors object whose vocab is used to construct the token encoding
        :param tokenizer: Tokenizer object used to tokenize the text
        :param processor: TextProcessor object to apply to the text before tokenization
        :param force_single_answer: if True only include the first answer span
            (default True)
        :returns: A TrainDataset object
        """
        corpus: Corpus
        try:
            corpus = Corpus.from_disk(filename)
        except (IOError, pickle.UnpicklingError) as e:
            corpus = Corpus.from_raw(
                filename,
                tokenizer,
                processor,
                vectors,
                force_single_answer=force_single_answer,
            )
        return cls(corpus)


class EvalDataset(QADataset):
    """
    Class that holds a dataset to be used for evaluating a model trained on a different dataset
    (and therefore needs the training dataset's character -> id mappings to be instantiated)
    """

    def __init__(self, corpus: Corpus) -> None:
        super().__init__(corpus)

    @classmethod
    def load_dataset(
        cls,
        filename: str,
        vectors: WordVectors,
        char_mapping: Dict[str, int],
        tokenizer: Tokenizer,
        processor: TextProcessor,
        force_single_answer: bool = True,
    ) -> QADataset:
        """
        Reads the given qa data file and processes it into a TrainDataset using
        the provided word vectors' vocab, tokenizer and text processor
        :param filename: File that contains the QA data
        :param token_mapping: The token -> id mapping to use from the ground truth dataset
        :param char_mapping: The char -> id mapping to use from the ground truth dataset
        :param tokenizer: Tokenizer object used to tokenize the text
        :param processor: TextProcessor object to apply to the text before tokenization
        :param force_single_answer: if True only include first answer span as true
            (default True)
        :returns: An EvalDataset object
        """
        corpus: Corpus
        try:
            corpus = Corpus.from_disk(filename)
        except (IOError, pickle.UnpicklingError) as e:
            corpus = Corpus.from_raw(
                filename,
                tokenizer,
                processor,
                vectors,
                force_single_answer=force_single_answer,
                char_mapping=char_mapping,
            )
        return cls(corpus)
