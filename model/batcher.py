"""
Module that handles batching logic
"""

from typing import List
import numpy as np
from corpus import SampleCorpus
from qa import SampleBatch


class RandomBatcher():
    """
    Class that takes a Corpus and provides random batches of
    context-question-answer triplets
    """

    corpus: SampleCorpus
    batch_size: int
    num_samples: int
    batches_per_epoch: int
    sample_free: List[bool]

    def __init__(self, corpus: SampleCorpus, batch_size: int) -> None:
        self.corpus = corpus
        self.batch_size = batch_size
        self.num_samples = len(self.corpus.samples)
        self.batches_per_epoch = self.num_samples // self.batch_size
        self.sample_free = [True for _ in range(self.num_samples)]

    def __iter__(self):
        """
        Iterator method for RandomBatcher
        """
        return self

    def __next__(self):
        """
        Iterator method for RandomBatcher
        :returns: Next batch of encoded ctx-q-a triplets

        """
        if sum(self.sample_free) < self.batch_size:
            # We don't have enough samples left for a batch, restart with all
            self.sample_free = [True for _ in self.corpus.samples]
        samples = list(np.random.choice(self.corpus.samples,
                                        self.batch_size,
                                        replace=False,
                                        p=self.sample_free))
        return SampleBatch(samples)
