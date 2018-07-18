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
        self.w_question = nn.Parameter(t.empty(input_size))
        self.w_context = nn.Parameter(t.empty(input_size))
        self.w_multiple = nn.Parameter(t.empty(input_size))

        nn.init.normal_(self.w_question)
        nn.init.normal_(self.w_context)
        nn.init.normal_(self.w_multiple)

        self.ctx_softmax = nn.Softmax(dim=2)
        self.q_softmax = nn.Softmax(dim=1)

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

        q_weighted = question @ self.w_question  # (batch_len, max_question_len)
        ctx_weighted = context @ self.w_context  # (batch_len, max_context_len)
        multiple_weighted = (question.unsqueeze(1) * context.unsqueeze(2)) @ self.w_multiple  # (batch_len, max_context_len, max_question_len)

        similarity = t.sum(t.cat([q_weighted.unsqueeze(1).unsqueeze(3).expand((batch_len, max_context_len, max_question_len, 1)),
                                  ctx_weighted.unsqueeze(2).unsqueeze(3).expand((batch_len, max_context_len, max_question_len, 1)),
                                  multiple_weighted.unsqueeze(3)], dim=3), dim=3)

        c2q_att = t.bmm(self.ctx_softmax(similarity), question)
        q2c_att = t.bmm(self.q_softmax(similarity.max(2)[0]).unsqueeze(1), context)
        return c2q_att, q2c_att
