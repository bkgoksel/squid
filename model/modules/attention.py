"""
Attention mechanism module
Hold modules for various Attention mechanisms
"""
from typing import NamedTuple

import torch as t
import torch.nn as nn
from torch import Tensor as Tensor

from model.modules.masked import MaskedLinear


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
    Wcm: nn.Parameter  # (input_size, hidden_size)
    Wqm: nn.Parameter  # (input_size, hidden_size)
    attention: MaskedLinear

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.attention = MaskedLinear(config.hidden_size, config.input_size)
        self.Wcm = nn.Parameter(t.empty(config.input_size, config.hidden_size))
        self.Wqm = nn.Parameter(t.empty(config.input_size, config.hidden_size))

        nn.init.xavier_uniform_(self.Wcm)
        nn.init.xavier_uniform_(self.Wqm)

    def forward(self, question_u: Tensor, context_enc: Tensor, context_mask: t.LongTensor):
        """
        Computes the final embedding for the context attended with the
        query encoding

        1. Compute m:
            - c@Wcm -> (batch, seq_len, din) @ (din, dhid) -> (batch, seq_len, dhid)
            - q@Wqm -> (batch, 1, din) @ (din, dhid) -> (batch, 1, dhid)
            - elem-wise: (batch, seq_len, dhid)
        2. Compute attention weights:
            - Linear(m) -> (batch_seq_len, dhid) -> (batch, seq_len, din)

        :param question_u: Output of question encoder:
            Tensor of shape [batch_size, 1, input_size]
        :param context_enc: All hidden states of context encoder:
            Tensor of shape [batch_size, max_context_len, input_size]
        :param context_mask: t.LongTensor that is 0 for padding indices of contexts:
            t.LongTensor of shape: [batch_size, max_context_len]
        :returns: Attended context
            Tensor of shape [batch_size, max_context_len, din]
        """
        context_enc = context_enc @ self.Wcm
        q_enc = question_u @ self.Wqm
        attended = self.attention(context_enc * q_enc.unsqueeze(1), context_mask)

        return attended


class BidirectionalAttention(nn.Module):
    """
    Bidirectional Attention computations as described in Bidirectional
    Attention Flow
    """

    def __init__(self, config):
        self.ctx_softmax = nn.Softmax(dim=2)
        self.config = config

    def forward(self, context, question):
        """
        Computes Context2Query and Query2Context attention given context
        and query embeddings
        :param context: Context embeddings (batch_len, max_context_len, embedding_size)
        :param question: Query embeddings (batch_len, max_question_len, embedding_size)
        :returns: Tuple[Context2QueryAttention, Query2ContextAttention]
            Context2QueryAttention: (batch_len, max_context_len, embedding_size)
            Query2ContextAttention: (batch_len, embedding_size)
        """
        # TODO: Build S (batch_len, max_context_len, max_question_len)
        # S[:,t,j] = ws @ concat(context[:,t,:], question[:,j,:], ctx[:,t,:]*q[:,j,:])
        S_tj = t.zeros((self.config.batch_len, self.config.max_context_len, self.config.max_question_len))
        a_ctx = self.ctx_softmax(S_tj)  # (batch_len, max_context_len, max_question_len)
        # TODO: Build c2q (batch_len, max_context_len, embedding_size)
        # c2q[:,t,:] = sum_j(a_ctx[:,t,j]*question[:,j])
        pass
