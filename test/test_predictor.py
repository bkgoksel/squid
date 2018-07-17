"""
Module for testing predictor model utilities
"""

import unittest
from unittest.mock import Mock

import numpy as np
import torch as t
import torch.nn as nn
from torch.nn.utils.rnn import (PackedSequence,
                                pack_padded_sequence,
                                pad_packed_sequence,
                                pad_sequence)

from model.predictor import (BidafPredictor,
                             PredictorConfig)
from model.util import get_last_hidden_states


class PredictorTestCase(unittest.TestCase):
    def setUp(self):
        self.batch_size = 3
        self.seq_len = 4
        self.input_dim = 2
        self.hidden_size = 10
        self.config = Mock(PredictorConfig)

    def get_input(self) -> PackedSequence:
        seqs = []
        lens = list(range(self.seq_len, self.seq_len - self.batch_size, -1))
        for seq_len in lens:
            seqs.append(t.randn((seq_len, self.input_dim)))
        seqs = pad_sequence(seqs, batch_first=True)
        return pack_padded_sequence(seqs, lens, batch_first=True)

    def get_rnn(self, num_layers: int=1, bidirectional: bool=False):
        """
        Returns RNN with weights all set to 1
        """
        self.config.n_directions = 1 + int(bidirectional)
        self.config.total_hidden_size = self.config.n_directions * self.hidden_size
        rnn = nn.RNN(self.input_dim,
                     self.hidden_size,
                     num_layers,
                     batch_first=True,
                     bidirectional=bidirectional)
        return rnn

    def check_match(self, all_states, last_hidden_state, seq_lens):
        """
        all_states[sample][len][:hidden_size] (forward last hidden state)
        all_states[sample][0][hidden_size:] (backward last hidden state)
        """
        for sample in range(self.batch_size):
            hidden = t.cat([all_states[sample][seq_lens[sample] - 1][:self.hidden_size].detach(),
                            all_states[sample][0][self.hidden_size:].detach()])
            self.assertTrue(np.allclose(hidden,
                                        last_hidden_state[sample].detach()),
                            'Calculated last hidden state doesn\'t match expected. \n Calculated: %s \n Expected: %s' %
                            (last_hidden_state[sample], all_states[sample][seq_lens[sample] - 1]))

    def test_get_last_hidden_states_simple(self):
        """
        Checks that get_last_hidden_states correctly gets the last
        hidden state of a unidirectional single layer RNN
        """
        rnn = self.get_rnn()
        inpt = self.get_input()
        all_states, states = rnn(inpt)
        all_states, lens = pad_packed_sequence(all_states)
        all_states.transpose_(0, 1)
        last_hidden_state = get_last_hidden_states(states, self.config)
        self.check_match(all_states, last_hidden_state, lens)

    def test_get_last_hidden_states_two_layers(self):
        """
        Checks that get_last_hidden_states correctly gets the last
        hidden state of a unidirectional two layer RNN
        """
        rnn = self.get_rnn(num_layers=2)
        inpt = self.get_input()
        all_states, states = rnn(inpt)
        all_states, lens = pad_packed_sequence(all_states)
        all_states.transpose_(0, 1)
        last_hidden_state = get_last_hidden_states(states, self.config)
        self.check_match(all_states, last_hidden_state, lens)

    def test_get_last_hidden_states_bidirectional(self):
        """
        Checks that get_last_hidden_states correctly gets the last
        hidden state of a bidirectional single layer RNN
        """
        rnn = self.get_rnn(bidirectional=True)
        inpt = self.get_input()
        all_states, states = rnn(inpt)
        all_states, lens = pad_packed_sequence(all_states)
        all_states.transpose_(0, 1)
        last_hidden_state = get_last_hidden_states(states, self.config)
        self.check_match(all_states, last_hidden_state, lens)

    def test_get_last_hidden_states_bidirectional_two_layer(self):
        """
        Checks that get_last_hidden_states correctly gets the last
        hidden state of a bidirectional two layer RNN
        """
        rnn = self.get_rnn(num_layers=2, bidirectional=True)
        inpt = self.get_input()
        all_states, states = rnn(inpt)
        all_states, lens = pad_packed_sequence(all_states)
        all_states.transpose_(0, 1)
        last_hidden_state = get_last_hidden_states(states, self.config)
        self.check_match(all_states, last_hidden_state, lens)
