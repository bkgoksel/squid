"""
Attention mechanism module
Hold modules for various Attention mechanisms
"""

from typing import ClassVar, Optional
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
    use_linear_layer: bool
    final_encoding_size: int
    w_question: nn.Parameter
    w_context: nn.Parameter
    w_multiple: nn.Parameter
    ctx_softmax: nn.Softmax
    q_softmax: nn.Softmax
    final_linear_layer_1: Optional[MaskedLinear]
    final_linear_layer_2: Optional[MaskedLinear]
    final_linear_activation: Optional[nn.ReLU]

    def __init__(
        self,
        input_size: int,
        linear_layer: bool = False,
        linear_hidden_size: int = 0,
        self_attention: bool = False,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.use_linear_layer = linear_layer

        self.w_question = nn.Parameter(t.empty(input_size))
        self.w_context = nn.Parameter(t.empty(input_size))
        self.w_multiple = nn.Parameter(t.empty(input_size))

        nn.init.normal_(self.w_question)
        nn.init.normal_(self.w_context)
        nn.init.normal_(self.w_multiple)

        self.ctx_softmax = nn.Softmax(dim=2)
        self.q_softmax = nn.Softmax(dim=1)
        self.final_encoding_size = input_size if self.self_attention else 4 * input_size

        if self.use_linear_layer and linear_hidden_size:
            self.final_linear_layer_1 = MaskedLinear(
                self.final_encoding_size, linear_hidden_size
            )
            self.final_linear_layer_2 = MaskedLinear(
                linear_hidden_size, self.final_encoding_size
            )
            self.final_linear_activation = nn.ReLU()

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

        q_weighted = question @ self.w_question  # (batch_len, max_question_len)
        similarity_tensors.append(
            q_weighted.unsqueeze(1)
            .unsqueeze(3)
            .expand((batch_len, max_context_len, max_question_len, 1))
        )
        del q_weighted

        ctx_weighted = context @ self.w_context  # (batch_len, max_context_len)
        similarity_tensors.append(
            ctx_weighted.unsqueeze(2)
            .unsqueeze(3)
            .expand((batch_len, max_context_len, max_question_len, 1))
        )
        del ctx_weighted

        multiple_weighted = t.einsum(
            "e,bqe,bce->bcq", [self.w_multiple, question, context]
        )
        similarity_tensors.append(multiple_weighted.unsqueeze(3))
        del multiple_weighted

        similarity = t.sum(t.cat(similarity_tensors, dim=3), dim=3)
        del similarity_tensors

        if self.self_attention:
            # Mask out the diagonal from the similarity matrix
            similarity = (
                similarity
                + t.eye(similarity.size(1), device=similarity.device).unsqueeze(0)
                * BaseBidirectionalAttention.NEGATIVE_COEFF
            )

        c2q_att = t.bmm(self.ctx_softmax(similarity), question)
        if self.self_attention:
            attended_ctx = c2q_att
        else:
            q2c_att = t.bmm(self.q_softmax(similarity.max(2)[0]).unsqueeze(1), context)
            attended_ctx = t.cat(
                [context, c2q_att, context * c2q_att, context * q2c_att], dim=2
            )
        del similarity

        if (
            self.linear_layer
            and self.final_linear_layer_1
            and self.final_linear_layer_2
            and self.final_linear_activation
        ):
            final_linear = self.final_linear_layer_1(attended_ctx, mask=context_mask)
            final_linear = self.final_linear_layer_2(final_linear, mask=context_mask)
            attended_ctx = self.final_linear_activation(final_linear)
        return attended_ctx


class BidafBidirectionalAttention(BaseBidirectionalAttention):
    """
    Basic Bidirectional Attention Flow layer as described in the
    original paper
    """

    def __init__(self, input_size: int) -> None:
        super().__init__(input_size)


class DocQABidirectionalAttention(BaseBidirectionalAttention):
    """
    Bidirectional Attention computations as described in DocQA
    (with linear layer at the end)
    """

    def __init__(self, input_size: int, linear_hidden_size: int) -> None:
        super().__init__(
            input_size, linear_layer=True, linear_hidden_size=linear_hidden_size
        )


class SelfAttention(BaseBidirectionalAttention):
    """
    Self Attention computations as described in DocumentQA
    """

    def __init__(self, input_size: int, linear_hidden_size: int) -> None:
        super().__init__(
            input_size,
            linear_layer=True,
            linear_hidden_size=linear_hidden_size,
            self_attention=True,
        )
