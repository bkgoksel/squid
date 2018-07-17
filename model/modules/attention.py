"""
Attention mechanism module
Hold modules for various Attention mechanisms
"""
from typing import NamedTuple

import torch as t
import torch.nn as nn
from torch import Tensor as Tensor

from model.modules.masked import MaskedLinear


class BidirectionalAttention(nn.Module):
    """
    Bidirectional Attention computations as described in Bidirectional
    Attention Flow
    """

    def __init__(self, config):
        self.ws = nn.Parameter(t.empty(3 * config.input_size))
        self.ctx_softmax = nn.Softmax(dim=2)
        self.q_softmax = nn.Softmax(dim=1)
        self.config = config
        nn.init.xavier_uniform_(self.ws)

    def forward(self, context, question):
        """
        Computes Context2Query and Query2Context attention given context
        and query embeddings
        :param context: Context embeddings (batch_len, max_context_len, embedding_size)
        :param question: Query embeddings (batch_len, max_question_len, embedding_size)
        :returns: Tuple[Context2QueryAttention, Query2ContextAttention]
            Context2QueryAttention: (batch_len, max_context_len, embedding_size)
            Query2ContextAttention: (batch_len, max_context_len, embedding_size)
        """
        q_unsqueeze = question.unsqueeze(1)
        ctx_unsqueeze = context.unsqueeze(2)
        similarity = self.ws @ t.cat([q_unsqueeze, ctx_unsqueeze, q_unsqueeze * ctx_unsqueeze], dim=3)
        c2q_att = t.bmm(self.ctx_softmax(similarity), question)
        q2c_att = t.bmm(self.q_softmax(similarity.max(2)[0]).unsqueeze(1), context)
        return c2q_att, q2c_att
