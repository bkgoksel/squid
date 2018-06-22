"""
Module that deals with preparing QA corpora
"""

import json
import pickle
from typing import (Any,
                    List,
                    Dict,
                    Set,
                    Tuple,
                    NamedTuple,
                    cast)

from torch.utils.data import Dataset

from model.text_processor import TextProcessor
from model.tokenizer import Tokenizer
from model.qa import (Answer,
                      QuestionAnswer,
                      ContextQuestionAnswer,
                      EncodedContextQuestionAnswer,
                      EncodedSample,
                      QuestionId)

from model.wv import WordVectors

CorpusStats = NamedTuple('CorpusStats', [
    ('n_contexts', int),
    ('n_questions', int),
    ('n_answerable', int),
    ('n_unanswerable', int),
    ('max_context_len', int),
    ('max_q_len', int),
    ('vocab_size', int)
])


class Corpus():
    """
    Class that contains a corpus
    This probably needs to store:
        - all the tokenized context-qas objects
        - the word vocab
    """
    context_qas: List[ContextQuestionAnswer]
    quids_to_context_qas: Dict[QuestionId, ContextQuestionAnswer]
    vocab: Set[str]
    stats: CorpusStats

    def __init__(self, context_qas: List[ContextQuestionAnswer],
                 vocab: Set[str], stats: CorpusStats) -> None:
        self.context_qas: List[ContextQuestionAnswer] = context_qas
        self.quids_to_context_qas: Dict[QuestionId, ContextQuestionAnswer] = dict()
        for cqa in context_qas:
            self.quids_to_context_qas.update({qa.question_id: cqa for qa in cqa.qas})
        self.vocab = vocab
        self.stats = stats

    @classmethod
    def from_disk(cls, serialized_file: str):
        """
        Loads a pickle serialized corpus from disk
        :param serialized_file: Name of the pickle file to load
        :returns: A Corpus object
        """
        with open(serialized_file, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def from_raw(cls, data_file: str, tokenizer: Tokenizer, processor: TextProcessor):
        context_qas = cls.read_context_qas(data_file, tokenizer, processor)
        vocab = cls.compute_vocab(context_qas)
        stats = cls.compute_stats(context_qas, vocab)
        return cls(context_qas, vocab, stats)

    @staticmethod
    def read_context_qas(data_file: str, tokenizer: Tokenizer, processor: TextProcessor) -> List[ContextQuestionAnswer]:
        """
        Reads a SQUAD formattted JSON file into ContextQuestionAnswer objects
        :param data_file: filename of the JSON questions file
        :param tokenizer: Tokenizer object to use to tokenize the text
        :returns: List[ContextQuestionAnswer], list of all the contexts and questions
        """
        contexts: List[ContextQuestionAnswer] = []
        with open(data_file, 'r') as f:
            json_dict = json.load(f)
            for doc in json_dict['data']:
                for paragraph in doc['paragraphs']:
                    context: str = paragraph['context']
                    qas: List[QuestionAnswer] = []
                    for qa in paragraph['qas']:
                        q_text: str = qa['question']
                        q_id: QuestionId = cast(QuestionId, qa['id'])
                        answers: Set[Answer] = set()
                        for answer in qa['answers']:
                            text: str = answer['text']
                            span_start: int = answer['answer_start']
                            tokenized_answer = Answer(text, span_start, tokenizer, processor)
                            answers.add(tokenized_answer)
                        tokenized_question = QuestionAnswer(q_id, q_text, answers, tokenizer, processor)
                        qas.append(tokenized_question)
                    tokenized_context = ContextQuestionAnswer(context, qas, tokenizer, processor)
                    contexts.append(tokenized_context)
        return contexts

    @staticmethod
    def compute_vocab(context_qas: List[ContextQuestionAnswer]) -> Set[str]:
        """
        Takes in a list of contexts and qas and returns the set of all words in them
        :param context_qas: List[ContextQuestionAnswer] all the context qa's
        :returns: Set[str] set of all strings in all the contexts and qas
        """
        vocab: Set[str] = set()
        for ctx in context_qas:
            vocab.update(set(tok.word for tok in ctx.tokens))
            for qa in ctx.qas:
                vocab.update(set(tok.word for tok in qa.tokens))
        return vocab

    @staticmethod
    def compute_stats(context_qas: List[ContextQuestionAnswer],
                      vocab: Set[str]) -> CorpusStats:
        """
        Method that computes statistics given list of context qas and vocab
        :param context_qas: List of contextQA objects
        :param vocab: set of strings that contains all tokens in vocab
        :returns: A CorpusStats object with stats of the corpus
        """
        n_contexts: int = len(context_qas)
        n_questions: int = sum(len(ctx.qas) for ctx in context_qas)
        n_answerable: int = sum(len([qa for qa in ctx.qas if qa.answers]) for ctx in context_qas)
        n_unanswerable: int = sum(len([qa for qa in ctx.qas if not qa.answers]) for ctx in context_qas)
        max_context_len: int = max(len(ctx.tokens) for ctx in context_qas)
        max_q_len: int = max(len(qa.tokens) for ctx in context_qas for qa in ctx.qas)
        return CorpusStats(n_contexts=n_contexts,
                           n_questions=n_questions,
                           n_answerable=n_answerable,
                           n_unanswerable=n_unanswerable,
                           max_context_len=max_context_len,
                           max_q_len=max_q_len,
                           vocab_size=len(vocab))

    def get_single_answer_text(self, qid: QuestionId, span_start: int, span_end: int) -> str:
        """
        Turns a single qid, span_start, span_end triplet into a string
        of concatenated tokens
        :param qid: The QuestionId of the question
        :param span_start: Model output index for answer start
        :param span_end: Model output index for answer end
        :returns: All tokens (inclusive) from the tokenized context,
            space separated
        """
        tokens = self.quids_to_context_qas[qid].tokens[span_start: span_end + 1]
        return ' '.join([tok.word for tok in tokens])

    def get_answer_texts(self, answer_token_idxs: Dict[QuestionId, Tuple[Any, ...]]) -> Dict[QuestionId, str]:
        """
        Given a mapping from questions id's to answer token indices, returns
        a SQuAD eval script readable version of the answer
        :param answer_token_idxs: a Mapping from QuestionId's to tuples
            of integers where the first elem is the span start
            prediction of the model and the second elem is the span
            end prediction of the model
        :returns: A Mapping from QuestionId's to strings
        """
        return {qid: self.get_single_answer_text(qid, span_start, span_end) for qid, (span_start, span_end) in answer_token_idxs.items()}

    def save(self, file_name: str) -> None:
        """
        Serializes this corpus to file with file_name
        :param file_name: File name to save corpus to
        :returns: None
        """
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)


class EncodedCorpus(Corpus):
    """
    Class that holds a Corpus alongside
    word vectors and token mappings
    """

    vocab: Set[str]
    stats: CorpusStats
    word_vectors: WordVectors
    encoded_context_qas: List[EncodedContextQuestionAnswer]

    def __init__(self, corpus: Corpus, word_vectors: WordVectors) -> None:
        super().__init__(corpus.context_qas, corpus.vocab, corpus.stats)
        self.word_vectors = word_vectors
        self.encoded_context_qas = EncodedCorpus.encode(self.context_qas, self.word_vectors)

    @staticmethod
    def encode(context_qas: List[ContextQuestionAnswer], word_vectors: WordVectors) -> List[EncodedContextQuestionAnswer]:
        """
        Method that encodes all the given contextQA's using given word vectors
        :param context_qas: List of ContextQA objects
        :param word_vectors: a WordVectors objects used to encode
        :returns: List of EncodedContextQuestionAnswer objects
        """
        return [EncodedContextQuestionAnswer(cqa, word_vectors) for cqa in context_qas]


class SampleCorpus(EncodedCorpus):
    """
    Class that stores a corpus of <context, question, answers> samples
    """

    context_qas: List[ContextQuestionAnswer]
    vocab: Set[str]
    stats: CorpusStats
    word_vectors: WordVectors
    encoded_context_qas: List[EncodedContextQuestionAnswer]
    samples: List[EncodedSample]
    n_samples: int

    def __init__(self, corpus: Corpus, word_vectors: WordVectors) -> None:
        super().__init__(corpus, word_vectors)
        self.samples = SampleCorpus.make_samples(self.encoded_context_qas)
        self.n_samples = len(self.samples)

    @staticmethod
    def make_samples(context_qas: List[EncodedContextQuestionAnswer]) -> List[EncodedSample]:
        """
        Method that converts a list of EncodedContextQA objects that store
        a given context with all its methods into EncodedSample objects
        that store each question with its context and answers
        :param context_qas: List of EncodedContextQuestionAnswer objects
        :returns: List of EncodedSample objects
        """
        return [EncodedSample(ctx.encoding, qa) for ctx in context_qas for qa in ctx.qas]


class QADataset(Dataset):
    """
    Class that turns a SampleCorpus into a PyTorch Dataset
    Main difference is that __getitem__ returns a dict instead of an EncodedSample
    """

    corpus: SampleCorpus

    def __init__(self, corpus: Corpus, word_vectors: WordVectors) -> None:
        self.corpus = SampleCorpus(corpus, word_vectors)

    def __len__(self):
        return self.corpus.n_samples

    def __getitem__(self, idx):
        return self.corpus.samples[idx]

    def get_answer_texts(self, answer_token_idxs: Dict[QuestionId, Tuple[Any, ...]]) -> Dict[QuestionId, str]:
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
    def stats(self):
        return self.corpus.stats
