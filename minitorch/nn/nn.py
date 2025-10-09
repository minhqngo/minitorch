from typing import Tuple
from .. import common_operators
from ..autodiff import Context
from ..backends.fast_ops import FastOps
from ..tensor.tensor import Tensor
from ..tensor.functions import Function, rand, tensor


def tile(input: Tensor, kernel: Tuple[int, int], stride: Tuple[int, int] = None) -> Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling
        stride: height x width stride of pooling (defaults to kernel size)

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """
    batch, channel, height, width = input.shape
    kh, kw = kernel

    if stride is None:
        stride = kernel
    sh, sw = stride

    if (sh, sw) == (kh, kw) and height % kh == 0 and width % kw == 0:
        input = input.contiguous()
        new_height, new_width = height // kh, width // kw
        input = input.view(batch, channel, new_height, kh, new_width, kw)
        input = input.permute(0, 1, 2, 4, 3, 5)
        input = input.contiguous()
        input = input.view(batch, channel, new_height, new_width, kh * kw)
        return input, new_height, new_width

    new_height = (height - kh) // sh + 1
    new_width = (width - kw) // sw + 1

    output = input.zeros((batch, channel, new_height, new_width, kh * kw))

    for b in range(batch):
        for c in range(channel):
            for oh in range(new_height):
                for ow in range(new_width):
                    start_h = oh * sh
                    start_w = ow * sw
                    idx = 0
                    for kh_i in range(kh):
                        for kw_i in range(kw):
                            ih = start_h + kh_i
                            iw = start_w + kw_i
                            if ih < height and iw < width:
                                output[b, c, oh, ow, idx] = input[b, c, ih, iw]
                            idx += 1

    return output, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int], stride: Tuple[int, int] = None) -> Tensor:
    """
    Tiled average pooling 2D

    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling
        stride : height x width stride of pooling (defaults to kernel size)

    Returns:
        Pooled tensor
    """
    if input.backend.cuda:
        from ..backends.cuda_conv import avgpool2d as cuda_avgpool2d
        return cuda_avgpool2d(input, kernel, stride)

    batch, channel, _, _ = input.shape
    tiled, new_height, new_width = tile(input, kernel, stride)
    out = tiled.sum(4) / (kernel[0] * kernel[1])
    return out.view(batch, channel, new_height, new_width)


max_reduce = FastOps.reduce(common_operators.max, -1e9)


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        "Forward of max should be max reduction"
        dim_val = int(dim.item())
        ctx.save_for_backward(input, dim_val)
        return input.backend.max_reduce(input, dim_val)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        "Backward of max should be argmax (see above)"
        input, dim = ctx.saved_values
        mask = argmax_onehot(input, dim)
        return mask * grad_output, 0.0


def argmax_onehot(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = input.backend.max_reduce(input, dim)
    mask = out == input
    num_max = mask.sum(dim)
    return mask * (1.0 / num_max)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as indices.

    Args:
        input : input tensor
        dim : dimension to apply argmax

    Returns:
        tensor with indices of highest values along dim
    """
    onehot = argmax_onehot(input, dim)
    shape = list(input.shape)
    indices_shape = [1] * len(shape)
    indices_shape[dim] = shape[dim]
    indices = tensor(
        list(range(shape[dim])), backend=input.backend
    ).view(*indices_shape)

    # Multiply one-hot by indices and sum along dim to get argmax
    return (onehot * indices).sum(dim)


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the softmax as a tensor.
    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
        input : input tensor
        dim : dimension to apply softmax

    Returns:
        softmax tensor
    """
    exp_input = input.exp()
    sum_exp = exp_input.sum(dim)
    return exp_input / sum_exp


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
         log of softmax tensor
    """
    m = max(input, dim)
    exp_input = (input - m).exp()
    sum_exp = exp_input.sum(dim)
    return input - m - sum_exp.log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int], stride: Tuple[int, int] = None) -> Tensor:
    """
    Tiled max pooling 2D

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling
        stride: height x width stride of pooling (defaults to kernel size)

    Returns:
        Tensor : pooled tensor
    """
    if input.backend.cuda:
        from ..backends.cuda_conv import maxpool2d as cuda_maxpool2d
        return cuda_maxpool2d(input, kernel, stride)

    batch, channel, height, width = input.shape
    tiled, new_height, new_width = tile(input, kernel, stride)
    out = max(tiled, 4)
    return out.view(batch, channel, new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
        tensor with random positions dropped out
    """
    if ignore:
        return input
    if rate == 1.0:
        return input * 0
    p_keep = 1.0 - rate
    mask = tensor([1.0]) - (rand(input.shape) < rate)
    return input * mask * (1.0 / p_keep)
