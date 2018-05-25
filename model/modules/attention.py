"""
Attention mechanism module
Hold modules for various Attention mechanisms
"""
from typing import NamedTuple

import torch as t
import torch.nn as nn
import torch.nn.functional as F


AttentionConfig = NamedTuple('AttentionConfig', [
    ('input_size', int),
    ('hidden_size', int)
])


class SimpleAttention(nn.Module):
    """
    Simple QA Attention layer as described in
    "Teaching Machines to Read and Comprehend"
    """

    config: AttentionConfig
    tanh: nn.Tanh
    Wym: nn.Parameter  # (input_size, hidden_size)
    Wum: nn.Parameter  # (input_size, hidden_size)
    Wms: nn.Parameter  # (hidden_size, 1)

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.tanh = nn.Tanh()
        self.Wym = nn.Parameter(t.Tensor(config.input_size, config.hidden_size))
        self.Wum = nn.Parameter(t.Tensor(config.input_size, config.hidden_size))
        self.Wms = nn.Parameter(t.Tensor(config.hidden_size, 1))

    def forward(self, question_out, context_enc):
        """
        Computes the final embedding for the context attended with the
        query encoding

        1. Compute u: concatenation of final forward and backward outputs of question_enc
            - (batch, 1, din)
        2. Compute m:
            - c@Wym -> (batch, seq_len, din) @ (din, dhid) -> (batch, seq_len, dhid)
            - u@Wum -> (batch, 1, din) @ (din, dhid) -> (batch, 1, dhid)
            - add: (batch, seq_len, dhid)
            - tanh: (batch, seq_len, dhid)
        3. Compute s:
            - m@Wms -> (batch, seq_len, dhid) @ (dhid, 1) -> (batch, seq_len, 1)
            - exp -> (batch, seq_len, 1)
        4. Compute r:
            - c*s -> (batch, seq_len, din) * (seq_len, 1) -> (batch, seq_len, din)
            - sum(cs, 1) -> (batch, din)

        :param question_u: Output of question encoder:
            Tensor of shape [batch_size, 1, input_size]
        :param context_enc: All hidden states of context encoder:
            Tensor of shape [batch_size, max_context_len, input_size]
        :returns: Question-attended Context encoding
            Tensor of shape [batch_size, input_size]
        """
        __import__('pdb').set_trace()
        context_enc = context_enc @ self.Wym
        q_enc = question_out @ self.Wum
        m = self.tanh(context_enc + q_enc.unsqueeze(1))
        s = t.exp(m @ self.Wms)
        r = t.sum(context_enc * s, 1)
