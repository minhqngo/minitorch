from ..tensor.functions import rand, zeros
from .module import Module, Parameter
from ..backends import fast_conv
from . import init

try:
    from ..backends import cuda_conv
except ImportError:
    cuda_conv = None


class Linear(Module):
    def __init__(self, in_size, out_size, backend, initializer=init.kaiming_uniform):
        super().__init__()
        self.weights = Parameter(rand((in_size, out_size), backend=backend))
        initializer(self.weights.value, in_size)
        self.bias = Parameter(zeros((out_size,), backend=backend))
        self.out_size = out_size

    def forward(self, x):
        batch, in_size = x.shape
        return (
            x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size) + self.bias.value


class Conv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_width, backend, stride=1, initializer=init.kaiming_uniform):
        super().__init__()
        self.weights = Parameter(rand((out_channels, in_channels, kernel_width), backend=backend))
        fan_in = in_channels * kernel_width
        initializer(self.weights.value, fan_in)
        self.bias = Parameter(zeros((1, out_channels, 1), backend=backend))
        self.backend = backend
        self.stride = stride
        self.kernel_width = kernel_width

    def forward(self, input):
        if self.backend.cuda and cuda_conv is not None:
            out = cuda_conv.conv1d(input, self.weights.value, self.stride) + self.bias.value
        else:
            out = fast_conv.conv1d(input, self.weights.value, self.stride) + self.bias.value
        return out


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel, backend, stride=1, initializer=init.kaiming_uniform):
        super().__init__()
        
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        kernel = kernel if isinstance(kernel, tuple) else (kernel, kernel)
        kh, kw = kernel
        
        self.weights = Parameter(rand((out_channels, in_channels, kh, kw), backend=backend))
        fan_in = in_channels * kh * kw
        initializer(self.weights.value, fan_in)
        
        self.bias = Parameter(zeros((out_channels, 1, 1), backend=backend))
        
        self.backend = backend

    def forward(self, input):
        if self.backend.cuda and cuda_conv is not None:
            out = cuda_conv.conv2d(input, self.weights.value, self.stride) + self.bias.value
        else:
            out = fast_conv.conv2d(input, self.weights.value, self.stride) + self.bias.value
        return out
