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

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.ws = nn.Parameter(t.empty(3 * input_size))
        self.ctx_softmax = nn.Softmax(dim=2)
        self.q_softmax = nn.Softmax(dim=1)
        nn.init.normal_(self.ws)

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
        batch_len, max_context_len, embedding_size = context.size()
        _, max_question_len, _ = question.size()
        q_unsqueeze = question.unsqueeze(1).expand((batch_len, max_context_len, max_question_len, embedding_size))
        ctx_unsqueeze = context.unsqueeze(2).expand((batch_len, max_context_len, max_question_len, embedding_size))
        similarity = t.cat([q_unsqueeze, ctx_unsqueeze, q_unsqueeze * ctx_unsqueeze], dim=3) @ self.ws
        c2q_att = t.bmm(self.ctx_softmax(similarity), question)
        q2c_att = t.bmm(self.q_softmax(similarity.max(2)[0]).unsqueeze(1), context)
        return c2q_att, q2c_att
