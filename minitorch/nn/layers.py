from ..tensor.functions import rand, zeros
from .module import Module, Parameter
from ..backends import fast_conv
from . import init

try:
    from ..backends import cuda_conv
except ImportError:
    cuda_conv = None


__all__ = ['Linear', 'Conv1d', 'Conv2d', 'RNN', 'tanh']


def tanh(x):
    """
    Hyperbolic tangent activation function.

    tanh(x) = (exp(2x) - 1) / (exp(2x) + 1) = 2*sigmoid(2x) - 1

    Args:
        x: Input tensor

    Returns:
        Tensor with tanh applied element-wise
    """
    # Using the sigmoid-based formula for numerical stability
    return 2.0 * (2.0 * x).sigmoid() - 1.0


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


class RNN(Module):
    def __init__(self, input_size, hidden_size, backend, initializer=init.glorot_uniform):
        super().__init__()

        # Input-to-hidden weights
        self.W_ih = Parameter(rand((input_size, hidden_size), backend=backend))
        initializer(self.W_ih.value, input_size, hidden_size)

        # Hidden-to-hidden weights
        self.W_hh = Parameter(rand((hidden_size, hidden_size), backend=backend))
        initializer(self.W_hh.value, hidden_size, hidden_size)

        # Bias
        self.bias = Parameter(zeros((hidden_size,), backend=backend))

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.backend = backend

    def forward(self, x, h=None):
        batch_size, seq_len, input_size = x.shape
        assert input_size == self.input_size, f"Expected input size {self.input_size}, got {input_size}"

        if h is None:
            h = zeros((batch_size, self.hidden_size), backend=self.backend)

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            h = tanh(
                x_t.view(batch_size, self.input_size) @ self.W_ih.value.view(self.input_size, self.hidden_size)
                + h.view(batch_size, self.hidden_size) @ self.W_hh.value.view(self.hidden_size, self.hidden_size)
                + self.bias.value.view(1, self.hidden_size)
            )
            outputs.append(h)
        
        output_tensors = []
        for i, out in enumerate(outputs):
            output_tensors.append(out.view(batch_size, 1, self.hidden_size))

        if seq_len == 1:
            output = output_tensors[0]
        else:
            output_list = []
            for b in range(batch_size):
                for t in range(seq_len):
                    for h_idx in range(self.hidden_size):
                        output_list.append(outputs[t][b, h_idx])

            from ..tensor.functions import tensor
            output = tensor(output_list, backend=self.backend).view(batch_size, seq_len, self.hidden_size)

        return output, h
