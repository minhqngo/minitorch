import pytest
from hypothesis import given, settings

import minitorch
from minitorch import Tensor
from .tensor_strategies import tensors

try:
    from minitorch.cuda_conv import conv1d, conv2d
    from minitorch.cuda_ops import CudaOps
    from minitorch.tensor_ops import TensorBackend
    import numba.cuda

    HAS_CUDA = numba.cuda.is_available()
    if HAS_CUDA:
        cuda_backend = TensorBackend(CudaOps)
except ImportError:
    HAS_CUDA = False

requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="requires CUDA integration")


@requires_cuda
@pytest.mark.cuda_conv
def test_conv1d_simple() -> None:
    t = minitorch.tensor([0, 1, 2, 3]).view(1, 1, 4)
    t2 = minitorch.tensor([[1, 2, 3]]).view(1, 1, 3)
    t_gpu = minitorch.tensor(t.to_numpy(), backend=cuda_backend)
    t2_gpu = minitorch.tensor(t2.to_numpy(), backend=cuda_backend)
    t_gpu.requires_grad_(True)
    t2_gpu.requires_grad_(True)
    minitorch.grad_check(conv1d, t_gpu, t2_gpu)


@requires_cuda
@pytest.mark.cuda_conv
@given(tensors(shape=(1, 1, 6)), tensors(shape=(1, 1, 4)))
def test_conv1d(input: Tensor, weight: Tensor) -> None:
    input_gpu = minitorch.tensor(input.to_numpy(), backend=cuda_backend)
    weight_gpu = minitorch.tensor(weight.to_numpy(), backend=cuda_backend)
    input_gpu.requires_grad_(True)
    weight_gpu.requires_grad_(True)
    minitorch.grad_check(conv1d, input_gpu, weight_gpu)


@requires_cuda
@pytest.mark.cuda_conv
@given(tensors(shape=(2, 2, 6)), tensors(shape=(3, 2, 2)))
@settings(max_examples=50)
def test_conv1d_channel(input: Tensor, weight: Tensor) -> None:
    input_gpu = minitorch.tensor(input.to_numpy(), backend=cuda_backend)
    weight_gpu = minitorch.tensor(weight.to_numpy(), backend=cuda_backend)
    input_gpu.requires_grad_(True)
    weight_gpu.requires_grad_(True)
    minitorch.grad_check(conv1d, input_gpu, weight_gpu)


@requires_cuda
@pytest.mark.cuda_conv
@given(tensors(shape=(1, 1, 6, 6)), tensors(shape=(1, 1, 2, 4)))
def test_conv(input: Tensor, weight: Tensor) -> None:
    input_gpu = minitorch.tensor(input.to_numpy(), backend=cuda_backend)
    weight_gpu = minitorch.tensor(weight.to_numpy(), backend=cuda_backend)
    input_gpu.requires_grad_(True)
    weight_gpu.requires_grad_(True)
    minitorch.grad_check(conv2d, input_gpu, weight_gpu)


@requires_cuda
@pytest.mark.cuda_conv
@given(tensors(shape=(2, 1, 6, 6)), tensors(shape=(1, 1, 2, 4)))
@settings(max_examples=10)
def test_conv_batch(input: Tensor, weight: Tensor) -> None:
    input_gpu = minitorch.tensor(input.to_numpy(), backend=cuda_backend)
    weight_gpu = minitorch.tensor(weight.to_numpy(), backend=cuda_backend)
    input_gpu.requires_grad_(True)
    weight_gpu.requires_grad_(True)
    minitorch.grad_check(conv2d, input_gpu, weight_gpu)


@requires_cuda
@pytest.mark.cuda_conv
@given(tensors(shape=(2, 2, 6, 6)), tensors(shape=(3, 2, 2, 4)))
@settings(max_examples=10)
def test_conv_channel(input: Tensor, weight: Tensor) -> None:
    input_gpu = minitorch.tensor(input.to_numpy(), backend=cuda_backend)
    weight_gpu = minitorch.tensor(weight.to_numpy(), backend=cuda_backend)
    input_gpu.requires_grad_(True)
    weight_gpu.requires_grad_(True)
    minitorch.grad_check(conv2d, input_gpu, weight_gpu)


@requires_cuda
@pytest.mark.cuda_conv
def test_conv2() -> None:
    t = minitorch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]).view(
        1, 1, 4, 4
    )
    t2 = minitorch.tensor([[1, 1], [1, 1]]).view(1, 1, 2, 2)
    t_gpu = minitorch.tensor(t.to_numpy(), backend=cuda_backend)
    t2_gpu = minitorch.tensor(t2.to_numpy(), backend=cuda_backend)
    t_gpu.requires_grad_(True)
    t2_gpu.requires_grad_(True)
    minitorch.grad_check(conv2d, t_gpu, t2_gpu)
