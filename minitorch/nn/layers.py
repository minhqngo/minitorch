from ..tensor.operators import TensorBackend
from ..tensor.functions import rand, zeros
from .module import Module, Parameter
from ..backends import fast_conv, fast_ops

BACKEND = TensorBackend(fast_ops.FastOps)


class Linear(Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        
        # He initialization
        scale = (2.0 / in_size) ** 0.5
        self.weights = Parameter(scale * rand((in_size, out_size), backend=BACKEND))
        self.bias = Parameter(zeros((out_size,), backend=BACKEND))
        self.out_size = out_size

    def forward(self, x):
        batch, in_size = x.shape
        return (
            x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size) + self.bias.value


class Conv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_width):
        super().__init__()
        
        # He initialization
        fan_in = in_channels * kernel_width
        scale = (2.0 / fan_in) ** 0.5
        self.weights = Parameter(
            scale * rand((out_channels, in_channels, kernel_width), backend=BACKEND)
        )
        self.bias = Parameter(zeros((1, out_channels, 1), backend=BACKEND))

    def forward(self, input):
        out = fast_conv.conv1d(input, self.weights.value) + self.bias.value
        return out


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kh, kw):
        super().__init__()
        
        # He initialization
        fan_in = in_channels * kh * kw
        scale = (2.0 / fan_in) ** 0.5
        self.weights = Parameter(
            scale * rand((out_channels, in_channels, kh, kw), backend=BACKEND)
        )
        self.bias = Parameter(zeros((out_channels, 1, 1), backend=BACKEND))

    def forward(self, input):
        out = fast_conv.conv2d(input, self.weights.value) + self.bias.value
        return out
