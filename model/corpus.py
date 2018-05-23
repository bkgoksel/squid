"""
Module that deals with preparing QA corpora
"""

import json
import pickle
from typing import List, Set
from tokenizer import Tokenizer
from qa import Answer, QuestionAnswer, ContextQuestionAnswer


class Corpus():
    """
    Class that contains a corpus
    This probably needs to store:
        - all the tokenized context-qas objects
        - the word vocab
    """
    context_qas: List[ContextQuestionAnswer]
    vocab: Set[str]
    tokenizer: Tokenizer

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
    def read_context_qas(cls, data_file: str, tokenizer: Tokenizer) -> List[ContextQuestionAnswer]:
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
                        answers: List[Answer] = []
                        for answer in qa['answers']:
                            text: str = answer['text']
                            span_start: int = answer['answer_start']
                            tokenized_answer = Answer(text, span_start, tokenizer)
                            answers.append(tokenized_answer)
                        tokenized_question = QuestionAnswer(q_text, answers, tokenizer)
                        qas.append(tokenized_question)
                    tokenized_context = ContextQuestionAnswer(context, qas, tokenizer)
                    contexts.append(tokenized_context)
        return contexts

    @classmethod
    def compute_vocab(cls, context_qas: List[ContextQuestionAnswer]) -> Set[str]:
        """
        Takes in a list of contexts and qas and returns the set of all words in them
        :param context_qas: List[ContextQuestionAnswer] all the context qa's
        :returns: Set[str] set of all strings in all the contexts and qas
        """
        vocab: Set[str] = set()
        for ctx in context_qas:
            vocab.update(set(ctx.tokens))
            for qa in ctx.qas:
                vocab.update(set(qa.tokens))
                for answer in qa.answers:
                    vocab.update(set(answer.tokens))
        return vocab

    def __init__(self, data_file: str, tokenizer: Tokenizer) -> None:
        self.context_qas = Corpus.read_context_qas(data_file, tokenizer)
        self.vocab = Corpus.compute_vocab(self.context_qas)
        self.tokenizer = tokenizer

    def save(self, file_name: str) -> None:
        """
        Serializes this corpus to file with file_name
        :param file_name: File name to save corpus to
        :returns: None
        """
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)
