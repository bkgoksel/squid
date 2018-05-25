"""
Attention mechanism module
Hold modules for various Attention mechanisms
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F


class SimpleAttention(nn.Module):
    """
    Simple QA Attention layer as described in
    "Teaching Machines to Read and Comprehend"
    """

    input_size: int
    hidden_size: int
    tanh: nn.Tanh
    Wym: nn.Parameter  # (input_size, hidden_size)
    Wum: nn.Parameter  # (input_size, hidden_size)
    Wms: nn.Parameter  # (hidden_size, 1)

    def __init_(self, input_size: int, hidden_size: int):
        self.tanh = nn.Tanh()
        self.Wym = nn.Parameter(t.Tensor((input_size, hidden_size)))
        self.Wum = nn.Parameter(t.Tensor((input_size, hidden_size)))
        self.Wms = nn.Parameter(t.Tensor((hidden_size, 1)))

    def forward(self, question_enc, context_enc):
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

        :param question_enc: Tensor of shape [batch_size, max_q_len, input_size]
        :param context_enc: Tensor of shape [batch_size, max_context_len, input_size]
        :returns: Tensor of shape [batch_size, input_size]
        """

        u = question_enc # TODO compute u
        m = self.tanh(context_enc @ self.Wym + question_enc @ self.Wum)
        s = t.exp(m @ self.Wms)
        r = t.sum(context_enc * s, 1)
