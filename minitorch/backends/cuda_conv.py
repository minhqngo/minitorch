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
    stride: int,
) -> None:
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= out_size:
        return

    out_index = cuda.local.array(MAX_DIMS, numba.int32)
    to_index(i, out_shape, out_index)

    b = out_index[0]
    oc = out_index[1]
    ow = out_index[2]

    _, in_channels, width = input_shape
    _, _, kw = weight_shape

    # Reuse these arrays instead of creating new ones in the loop
    input_index = cuda.local.array(MAX_DIMS, numba.int32)
    weight_index = cuda.local.array(MAX_DIMS, numba.int32)

    total = 0.0
    for ic in range(in_channels):
        for k in range(kw):
            if reverse:
                iw = ow * stride - k
            else:
                iw = ow * stride + k
            if iw >= 0 and iw < width:
                input_index[0] = b
                input_index[1] = ic
                input_index[2] = iw
                in_pos = index_to_position(input_index, input_strides)

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
    def forward(ctx: Context, input: Tensor, weight: Tensor, stride: int = 1) -> Tensor:
        ctx.save_for_backward(input, weight)
        ctx.stride = stride
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        out_w = (w - kw) // stride + 1

        output = input.zeros((batch, out_channels, out_w))
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (output.size + (threadsperblock - 1)) // threadsperblock
        tensor_conv1d[
            blockspergrid,
            threadsperblock,
        ](*output.tuple(), output.size, *input.tuple(), *weight.tuple(), False, stride)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        stride = ctx.stride
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
            stride,
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
    stride_h: int,
    stride_w: int,
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

    # Reuse these arrays instead of creating new ones in the loop
    input_index = cuda.local.array(MAX_DIMS, numba.int32)
    weight_index = cuda.local.array(MAX_DIMS, numba.int32)

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
                    input_index[0] = b
                    input_index[1] = ic
                    input_index[2] = ih
                    input_index[3] = iw
                    in_pos = index_to_position(input_index, input_strides)

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
    def forward(ctx: Context, input: Tensor, weight: Tensor, stride: Tuple[int, int] = (1, 1)) -> Tensor:
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
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (output.size + (threadsperblock - 1)) // threadsperblock
        tensor_conv2d[
            blockspergrid,
            threadsperblock,
        ](*output.tuple(), output.size, *input.tuple(), *weight.tuple(), False, stride_h, stride_w)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        stride = ctx.stride
        stride_h, stride_w = stride
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
            stride_h,
            stride_w,
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


def _tensor_avgpool2d_kernel(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
) -> None:
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= out_size:
        return

    out_index = cuda.local.array(MAX_DIMS, numba.int32)
    to_index(i, out_shape, out_index)

    b = out_index[0]
    c = out_index[1]
    oh = out_index[2]
    ow = out_index[3]

    _, _, height, width = input_shape
    kernel_size = kernel_h * kernel_w

    # Reuse this array instead of creating new ones in the loop
    input_index = cuda.local.array(MAX_DIMS, numba.int32)

    total = 0.0
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            ih = oh * stride_h + kh
            iw = ow * stride_w + kw
            if ih < height and iw < width:
                input_index[0] = b
                input_index[1] = c
                input_index[2] = ih
                input_index[3] = iw
                in_pos = index_to_position(input_index, input_strides)
                total += input[in_pos]

    out_pos = index_to_position(out_index, out_strides)
    out[out_pos] = total / kernel_size


tensor_avgpool2d = cuda.jit(_tensor_avgpool2d_kernel)


def _tensor_avgpool2d_backward_kernel(
    grad_input: Storage,
    grad_input_shape: Shape,
    grad_input_strides: Strides,
    grad_input_size: int,
    grad_output: Storage,
    grad_output_shape: Shape,
    grad_output_strides: Strides,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
) -> None:
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= grad_input_size:
        return

    grad_input_index = cuda.local.array(MAX_DIMS, numba.int32)
    to_index(i, grad_input_shape, grad_input_index)

    b = grad_input_index[0]
    c = grad_input_index[1]
    ih = grad_input_index[2]
    iw = grad_input_index[3]

    _, _, out_height, out_width = grad_output_shape
    kernel_size = kernel_h * kernel_w

    grad = 0.0
    grad_output_index = cuda.local.array(MAX_DIMS, numba.int32)

    # Find all output positions that this input contributes to
    for oh in range(out_height):
        for ow in range(out_width):
            # Check if this input position is in the pooling window for this output
            h_start = oh * stride_h
            h_end = h_start + kernel_h
            w_start = ow * stride_w
            w_end = w_start + kernel_w

            if ih >= h_start and ih < h_end and iw >= w_start and iw < w_end:
                grad_output_index[0] = b
                grad_output_index[1] = c
                grad_output_index[2] = oh
                grad_output_index[3] = ow
                out_pos = index_to_position(grad_output_index, grad_output_strides)
                grad += grad_output[out_pos] / kernel_size

    in_pos = index_to_position(grad_input_index, grad_input_strides)
    grad_input[in_pos] = grad


tensor_avgpool2d_backward = cuda.jit(_tensor_avgpool2d_backward_kernel)


def _tensor_maxpool2d_kernel(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
) -> None:
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= out_size:
        return

    out_index = cuda.local.array(MAX_DIMS, numba.int32)
    to_index(i, out_shape, out_index)

    b = out_index[0]
    c = out_index[1]
    oh = out_index[2]
    ow = out_index[3]

    _, _, height, width = input_shape

    # Reuse this array instead of creating new ones in the loop
    input_index = cuda.local.array(MAX_DIMS, numba.int32)

    # Initialize with first element
    ih_start = oh * stride_h
    iw_start = ow * stride_w
    input_index[0] = b
    input_index[1] = c
    input_index[2] = ih_start
    input_index[3] = iw_start
    in_pos = index_to_position(input_index, input_strides)
    max_val = input[in_pos]

    for kh in range(kernel_h):
        for kw in range(kernel_w):
            ih = oh * stride_h + kh
            iw = ow * stride_w + kw
            if ih < height and iw < width:
                input_index[0] = b
                input_index[1] = c
                input_index[2] = ih
                input_index[3] = iw
                in_pos = index_to_position(input_index, input_strides)
                val = input[in_pos]
                if val > max_val:
                    max_val = val

    out_pos = index_to_position(out_index, out_strides)
    out[out_pos] = max_val


tensor_maxpool2d = cuda.jit(_tensor_maxpool2d_kernel)


def _tensor_maxpool2d_backward_kernel(
    grad_input: Storage,
    grad_input_shape: Shape,
    grad_input_strides: Strides,
    grad_input_size: int,
    grad_output: Storage,
    grad_output_shape: Shape,
    grad_output_strides: Strides,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
) -> None:
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= grad_input_size:
        return

    grad_input_index = cuda.local.array(MAX_DIMS, numba.int32)
    to_index(i, grad_input_shape, grad_input_index)

    b = grad_input_index[0]
    c = grad_input_index[1]
    ih = grad_input_index[2]
    iw = grad_input_index[3]

    _, _, height, width = input_shape
    _, _, out_height, out_width = grad_output_shape

    grad = 0.0
    grad_output_index = cuda.local.array(MAX_DIMS, numba.int32)
    input_index = cuda.local.array(MAX_DIMS, numba.int32)

    # Find all output positions that this input might have contributed to
    for oh in range(out_height):
        for ow in range(out_width):
            # Check if this input position is in the pooling window for this output
            h_start = oh * stride_h
            h_end = min(h_start + kernel_h, height)
            w_start = ow * stride_w
            w_end = min(w_start + kernel_w, width)

            if ih >= h_start and ih < h_end and iw >= w_start and iw < w_end:
                # Find the max value in this window
                max_val = -1e9
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        test_h = h_start + kh
                        test_w = w_start + kw
                        if test_h < height and test_w < width:
                            input_index[0] = b
                            input_index[1] = c
                            input_index[2] = test_h
                            input_index[3] = test_w
                            in_pos = index_to_position(input_index, input_strides)
                            val = input[in_pos]
                            if val > max_val:
                                max_val = val

                # Check if current position was the max
                input_index[0] = b
                input_index[1] = c
                input_index[2] = ih
                input_index[3] = iw
                in_pos = index_to_position(input_index, input_strides)
                if abs(input[in_pos] - max_val) < 1e-8:  # This position was the max
                    grad_output_index[0] = b
                    grad_output_index[1] = c
                    grad_output_index[2] = oh
                    grad_output_index[3] = ow
                    out_pos = index_to_position(grad_output_index, grad_output_strides)
                    grad += grad_output[out_pos]

    in_pos = index_to_position(grad_input_index, grad_input_strides)
    grad_input[in_pos] = grad


tensor_maxpool2d_backward = cuda.jit(_tensor_maxpool2d_backward_kernel)


class AvgPool2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, kernel: Tuple[int, int], stride: Tuple[int, int]) -> Tensor:
        ctx.save_for_backward(input)
        ctx.kernel = kernel
        ctx.stride = stride

        batch, channel, height, width = input.shape
        kh, kw = kernel
        sh, sw = stride

        out_h = (height - kh) // sh + 1
        out_w = (width - kw) // sw + 1

        output = input.zeros((batch, channel, out_h, out_w))
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (output.size + (threadsperblock - 1)) // threadsperblock
        tensor_avgpool2d[blockspergrid, threadsperblock](
            *output.tuple(), output.size, *input.tuple(), kh, kw, sh, sw
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        input, = ctx.saved_values
        kernel = ctx.kernel
        stride = ctx.stride
        kh, kw = kernel
        sh, sw = stride

        grad_input = input.zeros(input.shape)
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (grad_input.size + (threadsperblock - 1)) // threadsperblock
        tensor_avgpool2d_backward[blockspergrid, threadsperblock](
            *grad_input.tuple(), grad_input.size, *grad_output.tuple(), kh, kw, sh, sw
        )
        return grad_input


class MaxPool2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, kernel: Tuple[int, int], stride: Tuple[int, int]) -> Tensor:
        ctx.save_for_backward(input)
        ctx.kernel = kernel
        ctx.stride = stride

        batch, channel, height, width = input.shape
        kh, kw = kernel
        sh, sw = stride

        out_h = (height - kh) // sh + 1
        out_w = (width - kw) // sw + 1

        output = input.zeros((batch, channel, out_h, out_w))
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (output.size + (threadsperblock - 1)) // threadsperblock
        tensor_maxpool2d[blockspergrid, threadsperblock](
            *output.tuple(), output.size, *input.tuple(), kh, kw, sh, sw
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        input, = ctx.saved_values
        kernel = ctx.kernel
        stride = ctx.stride
        kh, kw = kernel
        sh, sw = stride

        grad_input = input.zeros(input.shape)
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (grad_input.size + (threadsperblock - 1)) // threadsperblock
        tensor_maxpool2d_backward[blockspergrid, threadsperblock](
            *grad_input.tuple(), grad_input.size, *grad_output.tuple(),
            *input.tuple(), kh, kw, sh, sw
        )
        return grad_input


def avgpool2d(input: Tensor, kernel: Tuple[int, int], stride: Tuple[int, int] = None) -> Tensor:
    """
    CUDA-accelerated average pooling 2D with autograd support

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling
        stride: height x width stride of pooling (defaults to kernel size)

    Returns:
        Pooled tensor
    """
    if stride is None:
        stride = kernel

    # Check if gradients are needed
    need_grad = input.requires_grad()

    # Create context
    ctx = Context(not need_grad)

    # Call forward
    result = AvgPool2dFun.forward(ctx, input, kernel, stride)

    # Create history if needed
    if need_grad:
        import minitorch
        back = minitorch.History(AvgPool2dFun, ctx, (input,))
        return minitorch.Tensor(result._tensor, back, backend=result.backend)
    return result


def maxpool2d(input: Tensor, kernel: Tuple[int, int], stride: Tuple[int, int] = None) -> Tensor:
    """
    CUDA-accelerated max pooling 2D with autograd support

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling
        stride: height x width stride of pooling (defaults to kernel size)

    Returns:
        Pooled tensor
    """
    if stride is None:
        stride = kernel

    # Check if gradients are needed
    need_grad = input.requires_grad()

    # Create context
    ctx = Context(not need_grad)

    # Call forward
    result = MaxPool2dFun.forward(ctx, input, kernel, stride)

    # Create history if needed
    if need_grad:
        import minitorch
        back = minitorch.History(MaxPool2dFun, ctx, (input,))
        return minitorch.Tensor(result._tensor, back, backend=result.backend)
    return result
