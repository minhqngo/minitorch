import pytest
from hypothesis import given
import numpy as np

import minitorch
from minitorch import Tensor

from .strategies import assert_close
from .tensor_strategies import tensors

# Get backends for testing
FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)


@pytest.mark.nn_layers
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t: Tensor) -> None:
    out = minitorch.avgpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = minitorch.avgpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = minitorch.avgpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.nn_layers
@given(tensors(shape=(2, 3, 4)))
def test_max(t: Tensor) -> None:
    out = minitorch.max(t, 0)
    assert_close(out[0, 0, 0], max(t[i, 0, 0] for i in range(2)))
    out = minitorch.max(t, 1)
    assert_close(out[0, 0, 0], max(t[0, i, 0] for i in range(3)))
    out = minitorch.max(t, 2)
    assert_close(out[0, 0, 0], max(t[0, 0, i] for i in range(4)))


@pytest.mark.nn_layers
@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t: Tensor) -> None:
    out = minitorch.maxpool2d(t, (2, 2))
    print(out)
    print(t)
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = minitorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = minitorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.nn_layers
@given(tensors())
def test_drop(t: Tensor) -> None:
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0
    q = minitorch.dropout(t, 1.0, ignore=True)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]


@pytest.mark.nn_layers
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)


@pytest.mark.nn_layers
@given(tensors(shape=(1, 1, 4, 4)))
def test_log_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    q2 = minitorch.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])

    minitorch.grad_check(lambda a: minitorch.logsoftmax(a, dim=2), t)


@pytest.mark.nn_layers
def test_rnn_init() -> None:
    """Test RNN layer initialization"""
    input_size = 5
    hidden_size = 10

    rnn = minitorch.RNN(input_size, hidden_size, FastTensorBackend)

    assert rnn.W_ih.value.shape == (input_size, hidden_size)
    assert rnn.W_hh.value.shape == (hidden_size, hidden_size)
    assert rnn.bias.value.shape == (hidden_size,)

    assert rnn.input_size == input_size
    assert rnn.hidden_size == hidden_size


@pytest.mark.nn_layers
def test_rnn_forward_single_step() -> None:
    batch_size = 2
    seq_len = 1
    input_size = 3
    hidden_size = 4

    rnn = minitorch.RNN(input_size, hidden_size, FastTensorBackend)

    x = minitorch.rand((batch_size, seq_len, input_size), backend=FastTensorBackend)

    output, h_final = rnn(x)

    assert output.shape == (batch_size, seq_len, hidden_size)
    assert h_final.shape == (batch_size, hidden_size)


@pytest.mark.nn_layers
def test_rnn_forward_multiple_steps() -> None:
    batch_size = 2
    seq_len = 5
    input_size = 3
    hidden_size = 4

    rnn = minitorch.RNN(input_size, hidden_size, FastTensorBackend)

    x = minitorch.rand((batch_size, seq_len, input_size), backend=FastTensorBackend)

    output, h_final = rnn(x)

    # Check output shapes
    assert output.shape == (batch_size, seq_len, hidden_size)
    assert h_final.shape == (batch_size, hidden_size)


@pytest.mark.nn_layers
def test_rnn_forward_with_initial_hidden() -> None:
    batch_size = 2
    seq_len = 3
    input_size = 3
    hidden_size = 4

    rnn = minitorch.RNN(input_size, hidden_size, FastTensorBackend)

    x = minitorch.rand((batch_size, seq_len, input_size), backend=FastTensorBackend)
    h0 = minitorch.rand((batch_size, hidden_size), backend=FastTensorBackend)
    
    output, h_final = rnn(x, h0)

    assert output.shape == (batch_size, seq_len, hidden_size)
    assert h_final.shape == (batch_size, hidden_size)


@pytest.mark.nn_layers
def test_rnn_output_values() -> None:
    batch_size = 1
    seq_len = 3
    input_size = 2
    hidden_size = 3

    rnn = minitorch.RNN(input_size, hidden_size, FastTensorBackend)

    x_data = [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]
    x = minitorch.tensor(x_data, backend=FastTensorBackend)

    output, h_final = rnn(x)

    for b in range(batch_size):
        for t in range(seq_len):
            for h in range(hidden_size):
                val = output[b, t, h]
                assert -1.5 <= val <= 1.5, f"Output value {val} out of expected range"


@pytest.mark.nn_layers
def test_rnn_hidden_state_evolution() -> None:
    batch_size = 1
    seq_len = 3
    input_size = 2
    hidden_size = 3

    rnn = minitorch.RNN(input_size, hidden_size, FastTensorBackend)

    x = minitorch.rand((batch_size, seq_len, input_size), backend=FastTensorBackend)

    output, h_final = rnn(x)

    for h in range(hidden_size):
        assert_close(h_final[0, h], output[0, seq_len - 1, h])


@pytest.mark.nn_layers
def test_rnn_parameters() -> None:
    input_size = 3
    hidden_size = 4

    rnn = minitorch.RNN(input_size, hidden_size, FastTensorBackend)

    params = rnn.parameters()

    assert len(params) == 3

    named_params = dict(rnn.named_parameters())
    assert 'W_ih' in named_params
    assert 'W_hh' in named_params
    assert 'bias' in named_params


@pytest.mark.nn_layers
def test_rnn_batching() -> None:
    seq_len = 3
    input_size = 2
    hidden_size = 3

    rnn = minitorch.RNN(input_size, hidden_size, FastTensorBackend)

    x1 = minitorch.rand((1, seq_len, input_size), backend=FastTensorBackend)
    output1, h1 = rnn(x1)
    assert output1.shape == (1, seq_len, hidden_size)

    x4 = minitorch.rand((4, seq_len, input_size), backend=FastTensorBackend)
    output4, h4 = rnn(x4)
    assert output4.shape == (4, seq_len, hidden_size)
