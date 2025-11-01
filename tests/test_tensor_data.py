import pytest
from hypothesis import given
from hypothesis.strategies import DataObject, data

import minitorch
from minitorch import TensorData

from .tensor_strategies import indices, tensor_data


@pytest.mark.tensor_data
def test_layout() -> None:
    "Test basis properties of layout and strides"
    data = [0] * 3 * 5
    tensor_data = minitorch.TensorData(data, (3, 5), (5, 1))

    assert tensor_data.is_contiguous()
    assert tensor_data.shape == (3, 5)
    assert tensor_data.index((1, 0)) == 5
    assert tensor_data.index((1, 2)) == 7

    tensor_data = minitorch.TensorData(data, (5, 3), (1, 5))
    assert tensor_data.shape == (5, 3)
    assert not tensor_data.is_contiguous()

    data = [0] * 4 * 2 * 2
    tensor_data = minitorch.TensorData(data, (4, 2, 2))
    assert tensor_data.strides == (4, 2, 1)


@pytest.mark.xfail
def test_layout_bad() -> None:
    "Test basis properties of layout and strides"
    data = [0] * 3 * 5
    minitorch.TensorData(data, (3, 5), (6,))


@pytest.mark.tensor_data
@given(tensor_data())
def test_enumeration(tensor_data: TensorData) -> None:
    "Test enumeration of tensor_datas."
    indices = list(tensor_data.indices())

    # Check that enough positions are enumerated.
    assert len(indices) == tensor_data.size

    # Check that enough positions are enumerated only once.
    assert len(set(tensor_data.indices())) == len(indices)

    # Check that all indices are within the shape.
    for ind in tensor_data.indices():
        for i, p in enumerate(ind):
            assert p >= 0 and p < tensor_data.shape[i]


@pytest.mark.tensor_data
@given(tensor_data())
def test_index(tensor_data: TensorData) -> None:
    "Test enumeration of tensor_data."
    # Check that all indices are within the size.
    for ind in tensor_data.indices():
        pos = tensor_data.index(ind)
        assert pos >= 0 and pos < tensor_data.size

    base = [0] * tensor_data.dims
    with pytest.raises(minitorch.IndexingError):
        base[0] = -1
        tensor_data.index(tuple(base))

    if tensor_data.dims > 1:
        with pytest.raises(minitorch.IndexingError):
            base = [0] * (tensor_data.dims - 1)
            tensor_data.index(tuple(base))


@pytest.mark.tensor_data
@given(data())
def test_permute(data: DataObject) -> None:
    td = data.draw(tensor_data())
    ind = data.draw(indices(td))
    td_rev = td.permute(*list(reversed(range(td.dims))))
    assert td.index(ind) == td_rev.index(tuple(reversed(ind)))

    td2 = td_rev.permute(*list(reversed(range(td_rev.dims))))
    assert td.index(ind) == td2.index(ind)


@pytest.mark.tensor_data
def test_shape_broadcast() -> None:
    c = minitorch.shape_broadcast((1,), (5, 5))
    assert c == (5, 5)

    c = minitorch.shape_broadcast((5, 5), (1,))
    assert c == (5, 5)

    c = minitorch.shape_broadcast((1, 5, 5), (5, 5))
    assert c == (1, 5, 5)

    c = minitorch.shape_broadcast((5, 1, 5, 1), (1, 5, 1, 5))
    assert c == (5, 5, 5, 5)

    with pytest.raises(minitorch.IndexingError):
        c = minitorch.shape_broadcast((5, 7, 5, 1), (1, 5, 1, 5))
        print(c)

    with pytest.raises(minitorch.IndexingError):
        c = minitorch.shape_broadcast((5, 2), (5,))
        print(c)

    c = minitorch.shape_broadcast((2, 5), (5,))
    assert c == (2, 5)


@given(tensor_data())
def test_string(tensor_data: TensorData) -> None:
    tensor_data.to_string()


@pytest.mark.tensor_data
def test_slice_basic_1d() -> None:
    data = list(range(10))
    tensor_data = minitorch.TensorData(data, (10,))

    sliced = tensor_data.slice(slice(2, 5))
    assert sliced.shape == (3,), f"Expected shape (3,), got {sliced.shape}"
    assert sliced.get((0,)) == 2.0
    assert sliced.get((1,)) == 3.0
    assert sliced.get((2,)) == 4.0


@pytest.mark.tensor_data
def test_slice_basic_2d() -> None:
    "Test basic slicing on 2D tensor"
    data = list(range(12))
    tensor_data = minitorch.TensorData(data, (3, 4))

    sliced = tensor_data.slice((slice(1, 3), slice(None)))
    assert sliced.shape == (2, 4), f"Expected shape (2, 4), got {sliced.shape}"

    sliced = tensor_data.slice((slice(None), slice(1, 3)))
    assert sliced.shape == (3, 2), f"Expected shape (3, 2), got {sliced.shape}"


@pytest.mark.tensor_data
def test_slice_with_step() -> None:
    data = list(range(10))
    tensor_data = minitorch.TensorData(data, (10,))

    sliced = tensor_data.slice(slice(0, 10, 2))
    assert sliced.shape == (5,), f"Expected shape (5,), got {sliced.shape}"
    assert sliced.get((0,)) == 0.0
    assert sliced.get((1,)) == 2.0

    sliced = tensor_data.slice(slice(0, 10, 3))
    assert sliced.shape == (4,), f"Expected shape (4,), got {sliced.shape}"


@pytest.mark.tensor_data
def test_slice_negative_indices() -> None:
    "Test slicing with negative indices"
    data = list(range(10))
    tensor_data = minitorch.TensorData(data, (10,))

    sliced = tensor_data.slice(slice(-3, None))
    assert sliced.shape == (3,), f"Expected shape (3,), got {sliced.shape}"

    sliced = tensor_data.slice(slice(None, -2))
    assert sliced.shape == (8,), f"Expected shape (8,), got {sliced.shape}"

    sliced = tensor_data.slice(slice(-5, -2))
    assert sliced.shape == (3,), f"Expected shape (3,), got {sliced.shape}"


@pytest.mark.tensor_data
def test_slice_mixed_indexing() -> None:
    data = list(range(24))
    tensor_data = minitorch.TensorData(data, (4, 3, 2))

    sliced = tensor_data.slice((1, slice(None), slice(None)))
    assert sliced.shape == (3, 2), f"Expected shape (3, 2), got {sliced.shape}"

    sliced = tensor_data.slice((slice(None), 1, slice(None)))
    assert sliced.shape == (4, 2), f"Expected shape (4, 2), got {sliced.shape}"

    sliced = tensor_data.slice((slice(None), slice(None), 1))
    assert sliced.shape == (4, 3), f"Expected shape (4, 3), got {sliced.shape}"


@pytest.mark.tensor_data
def test_slice_full_slice() -> None:
    data = list(range(12))
    tensor_data = minitorch.TensorData(data, (3, 4))

    sliced = tensor_data.slice((slice(None), slice(None)))
    assert sliced.shape == (3, 4), f"Expected shape (3, 4), got {sliced.shape}"


@pytest.mark.tensor_data
def test_slice_empty() -> None:
    data = list(range(10))
    tensor_data = minitorch.TensorData(data, (10,))

    sliced = tensor_data.slice(slice(5, 5))
    assert sliced.shape == (0,), f"Expected shape (0,), got {sliced.shape}"

    sliced = tensor_data.slice(slice(7, 5))
    assert sliced.shape == (0,), f"Expected shape (0,), got {sliced.shape}"
