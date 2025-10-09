from typing import Tuple

import numpy as np
from numba import njit, prange

from ..autodiff import Context
from ..tensor.tensor import Tensor
from ..tensor.data import (
    MAX_DIMS,
    Index,
    Shape,
    Strides,
    broadcast_index,
    index_to_position,
    to_index,
)
from ..tensor.functions import Function

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


def _tensor_conv1d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
    stride: int,
) -> None:
    """
    1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, output_width`

    where output_width = (width - k_width) // stride + 1

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right
        stride (int): stride for convolution
    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides

    s1_b, s1_c, s1_w = s1[0], s1[1], s1[2]
    s2_oc, s2_ic, s2_k = s2[0], s2[1], s2[2]
    for i in prange(out_size):
        b = i // (out_channels * out_width)
        oc = (i // out_width) % out_channels
        ow = i % out_width

        total = 0.0
        for ic in range(in_channels):
            for k in range(kw):
                if reverse:
                    iw = ow * stride - k
                else:
                    iw = ow * stride + k
                if iw >= 0 and iw < width:
                    in_pos = b * s1_b + ic * s1_c + iw * s1_w
                    w_pos = oc * s2_oc + ic * s2_ic + k * s2_k
                    total += input[in_pos] * weight[w_pos]
        out[i] = total


tensor_conv1d = njit(parallel=True)(_tensor_conv1d)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor, stride: int = 1) -> Tensor:
        """
        Compute a 1D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x w
            weight : out_channel x in_channel x kw
            stride : stride for convolution

        Returns:
            batch x out_channel x output_w
        """
        ctx.save_for_backward(input, weight)
        ctx.stride = stride
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Calculate output width with stride
        out_w = (w - kw) // stride + 1

        # Run convolution
        output = input.zeros((batch, out_channels, out_w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False, stride
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        stride = ctx.stride
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        _, _, out_w = grad_output.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
            stride,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
            stride,
        )
        return grad_input, grad_weight


def conv1d(input: Tensor, weight: Tensor, stride: int = 1) -> Tensor:
    """Wrapper for Conv1dFun that handles non-Tensor stride parameter"""
    # Check if gradients are needed
    need_grad = input.requires_grad() or weight.requires_grad()

    # Detach inputs
    raw_input = input.detach()
    raw_weight = weight.detach()

    # Create context
    from ..autodiff import Context
    ctx = Context(not need_grad)

    # Call forward
    result = Conv1dFun.forward(ctx, raw_input, raw_weight, stride)

    # Create history if needed
    if need_grad:
        import minitorch
        back = minitorch.History(Conv1dFun, ctx, (input, weight))
        return minitorch.Tensor(result._tensor, back, backend=result.backend)
    return result


def _tensor_conv2d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
    stride_h: int,
    stride_w: int,
) -> None:
    """
    2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes output of

       `batch, out_channels, output_height, output_width`

    where output_height = (height - k_height) // stride_h + 1
          output_width = (width - k_width) // stride_w + 1

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right
        stride_h (int): stride for height dimension
        stride_w (int): stride for width dimension
    """
    batch_, out_channels, out_height, out_width = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    # inners
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    s1_b, s1_c, s1_h, s1_w = s1[0], s1[1], s1[2], s1[3]
    s2_oc, s2_ic, s2_kh, s2_kw = s2[0], s2[1], s2[2], s2[3]
    for i in prange(out_size):
        b = i // (out_channels * out_height * out_width)
        oc = (i // (out_height * out_width)) % out_channels
        oh = (i // out_width) % out_height
        ow = i % out_width

        total = 0.0
        for ic in range(in_channels):
            for kh_i in range(kh):
                for kw_i in range(kw):
                    if reverse:
                        ih = oh * stride_h - kh_i
                        iw = ow * stride_w - kw_i
                    else:
                        ih = oh * stride_h + kh_i
                        iw = ow * stride_w + kw_i

                    if ih >= 0 and ih < height and iw >= 0 and iw < width:
                        in_pos = b * s1_b + ic * s1_c + ih * s1_h + iw * s1_w
                        w_pos = oc * s2_oc + ic * s2_ic + kh_i * s2_kh + kw_i * s2_kw
                        total += input[in_pos] * weight[w_pos]
        out[i] = total


tensor_conv2d = njit(parallel=True, fastmath=True)(_tensor_conv2d)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor, stride: Tuple[int, int] = (1, 1)) -> Tensor:
        """
        Compute a 2D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw
            stride : tuple of (stride_h, stride_w) for convolution

        Returns:
            (:class:`Tensor`) : batch x out_channel x output_h x output_w
        """
        ctx.save_for_backward(input, weight)
        ctx.stride = stride
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2

        # Calculate output dimensions with stride
        stride_h, stride_w = stride
        out_h = (h - kh) // stride_h + 1
        out_w = (w - kw) // stride_w + 1

        output = input.zeros((batch, out_channels, out_h, out_w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False, stride_h, stride_w
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        stride = ctx.stride
        stride_h, stride_w = stride
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
            stride_h,
            stride_w,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
            stride_h,
            stride_w,
        )
        return grad_input, grad_weight


def conv2d(input: Tensor, weight: Tensor, stride: Tuple[int, int] = (1, 1)) -> Tensor:
    """Wrapper for Conv2dFun that handles non-Tensor stride parameter"""
    # Check if gradients are needed
    need_grad = input.requires_grad() or weight.requires_grad()

    # Detach inputs
    raw_input = input.detach()
    raw_weight = weight.detach()

    # Create context
    from ..autodiff import Context
    ctx = Context(not need_grad)

    # Call forward
    result = Conv2dFun.forward(ctx, raw_input, raw_weight, stride)

    # Create history if needed
    if need_grad:
        import minitorch
        back = minitorch.History(Conv2dFun, ctx, (input, weight))
        return minitorch.Tensor(result._tensor, back, backend=result.backend)
    return result
