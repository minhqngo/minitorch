from ..tensor.functions import rand, zeros
from .module import Module, Parameter
from ..backends import fast_conv, fast_ops
from . import init


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
    def __init__(self, in_channels, out_channels, kernel_width, backend, initializer=init.kaiming_uniform):
        super().__init__()
        self.weights = Parameter(rand((out_channels, in_channels, kernel_width), backend=backend))
        fan_in = in_channels * kernel_width
        initializer(self.weights.value, fan_in)
        self.bias = Parameter(zeros((1, out_channels, 1), backend=backend))

    def forward(self, input):
        out = fast_conv.conv1d(input, self.weights.value) + self.bias.value
        return out


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kh, kw, backend, initializer=init.kaiming_uniform):
        super().__init__()
        self.weights = Parameter(rand((out_channels, in_channels, kh, kw), backend=backend))
        fan_in = in_channels * kh * kw
        initializer(self.weights.value, fan_in)
        self.bias = Parameter(zeros((out_channels, 1, 1), backend=backend))

    def forward(self, input):
        out = fast_conv.conv2d(input, self.weights.value) + self.bias.value
        return out
