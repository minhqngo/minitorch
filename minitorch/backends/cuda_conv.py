from typing import Tuple

import numba
from numba import cuda

from ..autodiff import Context
from ..tensor.tensor import Tensor
from ..tensor.data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    index_to_position,
    to_index,
)
from ..tensor.functions import Function
from .cuda_ops import CudaOps, THREADS_PER_BLOCK

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)


def _tensor_conv1d_kernel(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= out_size:
        return

    # Deconstruct i into batch, out_channel, out_width
    out_index = cuda.local.array(MAX_DIMS, numba.int32)
    to_index(i, out_shape, out_index)

    b = out_index[0]
    oc = out_index[1]
    ow = out_index[2]

    _, in_channels, width = input_shape
    _, _, kw = weight_shape

    total = 0.0
    for ic in range(in_channels):
        for k in range(kw):
            iw = ow - k if reverse else ow + k
            if iw >= 0 and iw < width:
                input_index = cuda.local.array(MAX_DIMS, numba.int32)
                input_index[0] = b
                input_index[1] = ic
                input_index[2] = iw
                in_pos = index_to_position(input_index, input_strides)

                weight_index = cuda.local.array(MAX_DIMS, numba.int32)
                weight_index[0] = oc
                weight_index[1] = ic
                weight_index[2] = k
                w_pos = index_to_position(weight_index, weight_strides)

                total += input[in_pos] * weight[w_pos]

    out_pos = index_to_position(out_index, out_strides)
    out[out_pos] = total


tensor_conv1d = cuda.jit(_tensor_conv1d_kernel)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (output.size + (threadsperblock - 1)) // threadsperblock
        tensor_conv1d[
            blockspergrid,
            threadsperblock,
        ](*output.tuple(), output.size, *input.tuple(), *weight.tuple(), False)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, _, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (grad_weight.size + (threadsperblock - 1)) // threadsperblock
        tensor_conv1d[
            blockspergrid,
            threadsperblock,
        ](
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (grad_input.size + (threadsperblock - 1)) // threadsperblock
        tensor_conv1d[
            blockspergrid,
            threadsperblock,
        ](
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d_kernel(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= out_size:
        return

    out_index = cuda.local.array(MAX_DIMS, numba.int32)
    to_index(i, out_shape, out_index)

    b = out_index[0]
    oc = out_index[1]
    oh = out_index[2]
    ow = out_index[3]

    _, in_channels, height, width = input_shape
    _, _, kh, kw = weight_shape

    total = 0.0
    for ic in range(in_channels):
        for kh_i in range(kh):
            for kw_i in range(kw):
                ih = oh - kh_i if reverse else oh + kh_i
                iw = ow - kw_i if reverse else ow + kw_i

                if ih >= 0 and ih < height and iw >= 0 and iw < width:
                    input_index = cuda.local.array(MAX_DIMS, numba.int32)
                    input_index[0] = b
                    input_index[1] = ic
                    input_index[2] = ih
                    input_index[3] = iw
                    in_pos = index_to_position(input_index, input_strides)

                    weight_index = cuda.local.array(MAX_DIMS, numba.int32)
                    weight_index[0] = oc
                    weight_index[1] = ic
                    weight_index[2] = kh_i
                    weight_index[3] = kw_i
                    w_pos = index_to_position(weight_index, weight_strides)

                    total += input[in_pos] * weight[w_pos]

    out_pos = index_to_position(out_index, out_strides)
    out[out_pos] = total


tensor_conv2d = cuda.jit(_tensor_conv2d_kernel)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (output.size + (threadsperblock - 1)) // threadsperblock
        tensor_conv2d[
            blockspergrid,
            threadsperblock,
        ](*output.tuple(), output.size, *input.tuple(), *weight.tuple(), False)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, _, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (grad_weight.size + (threadsperblock - 1)) // threadsperblock
        tensor_conv2d[
            blockspergrid,
            threadsperblock,
        ](
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (grad_input.size + (threadsperblock - 1)) // threadsperblock
        tensor_conv2d[
            blockspergrid,
            threadsperblock,
        ](
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
