"""
Module to encompass logic around word vectors
"""

import pickle
import numpy as np
from typing import Tuple, List, Dict, Any


class WordVectors():
    """
    Class that reads, stores and saves pre-trained word vectors from disk
    """
    file_name: str
    idx_to_word: Dict[int, str]
    word_to_idx: Dict[str, int]
    vectors: Any  # numpy array
    dim: int

    def __init__(self,
                 vectors: Any,
                 idx_to_word: Dict[int, str],
                 word_to_idx: Dict[str, int]) -> None:
        self.vectors = vectors
        self.dim = vectors.shape[1]
        self.idx_to_word = idx_to_word
        self.word_to_idx = word_to_idx

    @classmethod
    def from_serialized(cls, file_name: str):
        """
        Class method that reads pickle files(under same name) from disk
        and restores a WordVectors object
        :param file_name: Prefix of pickle files for these vectors
        :returns: a WordVectors object
        """
        with open(file_name + '-idx-to-word.pkl', 'rb') as f:
            idx_to_word: Dict[int, str] = pickle.load(f)
        with open(file_name + '-word-to-idx.pkl', 'rb') as f:
            word_to_idx: Dict[str, int] = pickle.load(f)
        with open(file_name + '-vectors.npy', 'rb') as f:
            vectors: Any = np.load(f)
        return cls(vectors, idx_to_word, word_to_idx)

    @classmethod
    def from_text_vectors(cls, vector_file: str, consume_first_line: bool=False):
        """
        Class method that creates a WordVectors object from a text formatted
        word vectors file on disk
        :param vector_file: Name of the text-formatted pretrained word vector file
        :param consume_first_line: If True skip first line (to be used when first line is metadata
        :returns: A WordVectors object initialized from the given file
        """
        vectors, word_to_idx, idx_to_word = cls.read_vectors(vector_file, consume_first_line)
        return cls(vectors, idx_to_word, word_to_idx)

    @staticmethod
    def read_vectors(vector_file: str, consume_first_line: bool) -> Tuple[Any, Dict[str, int], Dict[int, str]]:
        """
        Class method that reads word vectors into a word->numpy array dict
        :param vector_file: Name of the word vector file to read from disk
        :param consume_first_line: if True skip first line as it is metadata and not a word vector
        :returns:a Tuple of:
            - vectors: a 2D numpy array of shape [num_words, vector_dim]
            - word_to_idx: A mapping from each word to its idx in vectors
            - idx_to_word: Reverse of the above mapping
        """
        vectors_list: List[Any] = []
        vocab: List[str] = []
        with open(vector_file, 'r') as f:
            if consume_first_line:
                next(f)
            for line in f:
                word, vec_data = line.split(' ', 1)
                vector = np.array(vec_data.split(' '))
                vectors_list.append(vector)
                vocab.append(word)
        vectors = np.array(vectors_list)
        idx_to_word: Dict[int, str] = dict(enumerate(vocab))
        word_to_idx: Dict[str, int] = {w: i for i, w in idx_to_word.items()}
        return vectors, word_to_idx, idx_to_word

    def save(self, file_name: str) -> None:
        """
        Serializes the vectors to a collection of files.
        Creates the following files:
            - <file_name>-idx-to-word.pkl
            - <file_name>-word-to-idx.pkl
            - <file_name>-vectors.pkl
        :param file_name: Prefix for files to be saved
        """
        with open(file_name + '-idx-to-word.pkl', 'wb') as f:
            pickle.dump(self.idx_to_word, f)
        with open(file_name + '-word-to-idx.pkl', 'wb') as f:
            pickle.dump(self.word_to_idx, f)
        with open(file_name + '-vectors.pkl', 'wb') as f:
            pickle.dump(self.vectors, f)
