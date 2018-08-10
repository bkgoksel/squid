"""
Attention mechanism module
Hold modules for various Attention mechanisms
"""

from typing import ClassVar
import torch as t
import torch.nn as nn

from model.modules.masked import MaskedLinear


class BaseBidirectionalAttention(nn.Module):
    """
    Base Bidirectional Attention computations as described in Bidirectional
    Attention Flow, with added ability to compute self-attention as described
    in DocumentQA
    """

    NEGATIVE_COEFF: ClassVar[float] = -1e30

    self_attention: bool
    final_encoding_size: int
    dropout: nn.Dropout
    w_question: nn.Parameter
    w_context: nn.Parameter
    w_multiple: nn.Parameter
    ctx_softmax: nn.Softmax
    q_softmax: nn.Softmax
    final_linear: nn.Sequential

    def __init__(
        self,
        input_size: int,
        linear_hidden_size: int,
        dropout_prob: float,
        self_attention: bool = False,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.final_encoding_size = input_size if self.self_attention else 4 * input_size

        self.dropout = nn.Dropout(dropout_prob)
        self.w_question = nn.Parameter(t.empty(input_size))
        self.w_context = nn.Parameter(t.empty(input_size))
        self.w_multiple = nn.Parameter(t.empty(input_size))

        nn.init.normal_(self.w_question)
        nn.init.normal_(self.w_context)
        nn.init.normal_(self.w_multiple)

        self.ctx_softmax = nn.Softmax(dim=2)
        self.q_softmax = nn.Softmax(dim=1)

        self.final_linear = nn.Sequential(
            MaskedLinear(self.final_encoding_size, linear_hidden_size),
            MaskedLinear(linear_hidden_size, self.final_encoding_size),
            nn.ReLU(),
        )

    def forward(
        self, context: t.Tensor, question: t.Tensor, context_mask: t.Tensor
    ) -> t.Tensor:
        """
        Computes Context2Query and Query2Context attention given context
        and query embeddings
        :param context: Context embeddings (batch_len, max_context_len, embedding_size)
        :param question: Query embeddings (batch_len, max_question_len, embedding_size)
        :param context_mask: Context mask (batch_len, max_context_len)
        :returns: Attended context (batch_len, max_context_len, embedding_size)
        """
        batch_len, max_context_len, embedding_size = context.size()
        _, max_question_len, _ = question.size()

        similarity_tensors = []

        q_dropped = self.dropout(question)
        del question
        q_weighted = q_dropped @ self.w_question  # (batch_len, max_question_len)
        similarity_tensors.append(
            q_weighted.unsqueeze(1)
            .unsqueeze(3)
            .expand((batch_len, max_context_len, max_question_len, 1))
        )
        del q_weighted

        ctx_dropped = self.dropout(context)
        del context
        ctx_weighted = ctx_dropped @ self.w_context  # (batch_len, max_context_len)
        similarity_tensors.append(
            ctx_weighted.unsqueeze(2)
            .unsqueeze(3)
            .expand((batch_len, max_context_len, max_question_len, 1))
        )
        del ctx_weighted

        # TODO: Make this not build a (B x H x H x E) tensor
        multiple_weighted = (
            q_dropped.unsqueeze(1) * ctx_dropped.unsqueeze(2)
        ) @ self.w_multiple  # (batch_len, max_context_len, max_question_len)
        similarity_tensors.append(multiple_weighted.unsqueeze(3))
        del multiple_weighted

        similarity = t.sum(t.cat(similarity_tensors, dim=3), dim=3)
        del similarity_tensors

        if self.self_attention:
            similarity = (
                similarity
                + t.eye(similarity.size(1), device=similarity.device).unsqueeze(0)
                * BaseBidirectionalAttention.NEGATIVE_COEFF
            )

        c2q_att = t.bmm(self.ctx_softmax(similarity), q_dropped)
        if self.self_attention:
            attended_tensors = c2q_att
        else:
            q2c_att = t.bmm(
                self.q_softmax(similarity.max(2)[0]).unsqueeze(1), ctx_dropped
            )
            attended_tensors = t.cat(
                [ctx_dropped, c2q_att, ctx_dropped * c2q_att, ctx_dropped * q2c_att],
                dim=2,
            )
        del similarity

        attended_context = self.final_linear(attended_tensors, mask=context_mask)
        return attended_context


class BidirectionalAttention(BaseBidirectionalAttention):
    """
    Bidirectional Attention computations as described in Bidirectional
    Attention Flow
    """

    def __init__(
        self, input_size: int, linear_hidden_size: int, dropout_prob: float
    ) -> None:
        super().__init__(
            input_size, linear_hidden_size, dropout_prob, self_attention=False
        )


class SelfAttention(BaseBidirectionalAttention):
    """
    Self Attention computations as described in DocumentQA
    """

    def __init__(
        self, input_size: int, linear_hidden_size: int, dropout_prob: float
    ) -> None:
        super().__init__(
            input_size, linear_hidden_size, dropout_prob, self_attention=True
        )
