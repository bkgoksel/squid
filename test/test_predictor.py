"""
Module for testing predictor model utilities
"""

import unittest
from unittest.mock import Mock

import torch as t
import torch.nn as nn
from torch.nn.utils.rnn import (PackedSequence,
                                pack_padded_sequence)

from model.predictor import (BasicPredictor,
                             BasicPredictorConfig)


class PredictorTestCase(unittest.TestCase):
    def setUp(self):
        self.batch_size = 3
        self.seq_len = 3
        self.input_dim = 2
        self.hidden_size = 10
        self.config = Mock(BasicPredictorConfig)

    def get_input(self) -> PackedSequence:
        seqs = t.Tensor([
            [[1, 1], [1, 1], [-1, -1]],
            [[1, 1], [-1, -1], [0, 0]],
            [[-1, -1], [0, 0], [0, 0]]
        ])
        lens = [3, 2, 1]
        return pack_padded_sequence(seqs, lens, batch_first=True)

    def get_rnn(self, num_layers: int=1, bidirectional: bool=False):
        self.config.n_directions = 1 + int(bidirectional)
        self.config.total_hidden_size = self.config.n_directions * self.hidden_size
        return nn.RNN(self.input_dim,
                      self.hidden_size,
                      num_layers,
                      batch_first=True,
                      bidirectional=bidirectional)

    def test_get_last_hidden_states_simple(self):
        """
        Checks that get_last_hidden_states correctly gets the last
        hidden state of a unidirectional single layer RNN
        """
        rnn = self.get_rnn()
        inpt = self.get_input()
        _, states = rnn(inpt)
        last_hidden_state = BasicPredictor.get_last_hidden_states(states, self.config)

    def test_get_last_hidden_states_two_layers(self):
        """
        Checks that get_last_hidden_states correctly gets the last
        hidden state of a unidirectional two layer RNN
        """
        rnn = self.get_rnn(num_layers=2)
        inpt = self.get_input()
        _, states = rnn(inpt)
        last_hidden_state = BasicPredictor.get_last_hidden_states(states, self.config)

    def test_get_last_hidden_states_bidirectional(self):
        """
        Checks that get_last_hidden_states correctly gets the last
        hidden state of a bidirectional single layer RNN
        """
        rnn = self.get_rnn(bidirectional=True)
        inpt = self.get_input()
        _, states = rnn(inpt)
        last_hidden_state = BasicPredictor.get_last_hidden_states(states, self.config)

    def test_get_last_hidden_states_bidirectional_two_layer(self):
        """
        Checks that get_last_hidden_states correctly gets the last
        hidden state of a bidirectional two layer RNN
        """
        rnn = self.get_rnn(num_layers=2, bidirectional=True)
        inpt = self.get_input()
        _, states = rnn(inpt)
        last_hidden_state = BasicPredictor.get_last_hidden_states(states, self.config)
