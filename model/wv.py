"""
Module to encompass logic around word vectors
"""

import pickle
import numpy as np
import torch as t
from typing import Tuple, List, Dict, Any, ClassVar


class WordVectors():
    """
    Class that reads, stores and saves pre-trained word vectors from disk
    """
    UNK_TOKEN: ClassVar[str] = '<UNK_TOKEN>'
    PAD_TOKEN: ClassVar[str] = '<PAD_TOKEN>'

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

    def contains(self, token: str) -> bool:
        """
        Checks whether these word vectors have an embedding for the given token
        :param: token, case sensitive
        :returns: True if embedding for token exists false otherwise
        """
        return token in self.word_to_idx

    def build_embeddings_matrix_for(self, token_mapping: Dict[str, int]) -> t.Tensor:
        """
        Builds a compact embeddings matrix for the given token -> idx mapping.
        The given token mapping should not contain any out-of-vocab words
        :raises: Exception if not all tokens in token_mapping are in vocab for the vectors object
        :param token_mapping: Mapping from tokens to would-be indices in the embedding matrix
            - All tokens should be in vocabulary for the vectors object
            - Indices 0 and 1 should be reserved for the padding and unk tokens
        :returns: Embedding matrix where each index <i> has the pretrained embeddings for the token
            that maps to that index in token_mapping.
            - index 0 is an all-0 padding vector
            - index 1 is a randomly initialized UNK vector
        """
        if not all(token in self.word_to_idx for token, idx in token_mapping.items() if idx != 1):
            raise Exception("Token not in word vector vocab")
        idx_sorted_tokens = [WordVectors.PAD_TOKEN, WordVectors.UNK_TOKEN] + [tok for tok, idx in sorted(token_mapping.items(), key=lambda x: x[1])]
        vector_indices = [self.word_to_idx[tok] for tok in idx_sorted_tokens]
        return np.take(self.vectors, vector_indices, axis=0)

    @classmethod
    def load_vectors(cls, file_name: str):
        """
        Class method that loads vectors from an arbitrary filename
        If the file is a pickle prefix for the processed vectors loads that
        if that fails tries to read them as text-based word vectors
        """
        try:
            vectors = WordVectors.from_disk(file_name)
        except (IOError, pickle.UnpicklingError) as e:
            vectors = WordVectors.from_text_vectors(file_name)
        return vectors

    @classmethod
    def from_disk(cls, file_name: str):
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
    def from_text_vectors(cls,
                          vector_file: str,
                          consume_first_line: bool=False):
        """
        Class method that creates a WordVectors object from a text formatted
        word vectors file on disk
        :param vector_file: Name of the text-formatted pretrained word vector file
        :param consume_first_line: If True skip first line (to be used when first line is metadata)
        :returns: A WordVectors object initialized from the given file
        """
        vectors, word_to_idx, idx_to_word = cls.read_vectors(vector_file, consume_first_line)
        return cls(vectors, idx_to_word, word_to_idx)

    @staticmethod
    def read_vectors(vector_file: str,
                     consume_first_line: bool) -> Tuple[Any, Dict[str, int], Dict[int, str]]:
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
                word, vec_data = line.rstrip('\n').split(' ', 1)
                vector = np.array([float(num) for num in vec_data.split(' ')], dtype=np.float32)
                vectors_list.append(vector)
                vocab.append(word)
        pad_vector = np.zeros_like(vectors_list[0])
        vectors_list.insert(0, pad_vector)
        vocab.insert(0, WordVectors.PAD_TOKEN)
        unk_vector = np.random.randn((vectors_list[0].shape[0]))
        vectors_list.insert(1, unk_vector)
        vocab.insert(1, WordVectors.UNK_TOKEN)
        vectors = np.stack(vectors_list)
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
        with open(file_name + '-vectors.npy', 'wb') as f:
            np.save(f, self.vectors)
