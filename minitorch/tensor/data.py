from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from ..common_operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index : index tuple of ints
        strides : tensor strides

    Returns:
        Position in storage
    """
    position = 0
    for i in range(len(strides)):
        position += index[i] * strides[i]
    return int(position)


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """
    Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    local_ordinal = ordinal
    for i in range(len(shape) - 1, -1, -1):
        if shape[i] == 0:
            out_index[i] = 0
            continue
        out_index[i] = local_ordinal % shape[i]
        local_ordinal = local_ordinal // shape[i]


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """
    Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
        None
    """
    len_diff = len(big_shape) - len(shape)
    for i in range(len(shape)):
        if shape[i] == big_shape[i + len_diff]:
            out_index[i] = big_index[i + len_diff]
        elif shape[i] == 1:
            out_index[i] = 0
        else:
            raise IndexingError("Cannot broadcast")


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """
    Broadcast two shapes to create a new union shape.

    Args:
        shape1 : first shape
        shape2 : second shape

    Returns:
        broadcasted shape

    Raises:
        IndexingError : if cannot broadcast
    """
    len1, len2 = len(shape1), len(shape2)
    max_len = max(len1, len2)

    s1 = ([1] * (max_len - len1)) + list(shape1)
    s2 = ([1] * (max_len - len2)) + list(shape2)

    new_shape = []
    for i in range(max_len):
        d1, d2 = s1[i], s2[i]
        if d1 == d2:
            new_shape.append(d1)
        elif d1 == 1:
            new_shape.append(d2)
        elif d2 == 1:
            new_shape.append(d1)
        else:
            raise IndexingError(f"Cannot broadcast shapes {shape1} and {shape2}")

    return tuple(new_shape)


def strides_from_shape(shape: UserShape) -> UserStrides:
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


def normalize_slice(s: slice, dim_size: int) -> Tuple[int, int, int]:
    """
    Normalize a slice object to (start, stop, step) with proper bounds.

    Args:
        s: slice object
        dim_size: size of the dimension being sliced

    Returns:
        (start, stop, step) tuple with normalized values
    """
    step = s.step if s.step is not None else 1
    if step == 0:
        raise IndexingError("slice step cannot be zero")

    if step < 0:
        start = s.start if s.start is not None else dim_size - 1
        stop = s.stop if s.stop is not None else -dim_size - 1
    else:
        start = s.start if s.start is not None else 0
        stop = s.stop if s.stop is not None else dim_size

    if start < 0:
        start = max(0, dim_size + start)
    else:
        start = min(start, dim_size)

    if stop < 0:
        stop = max(-1 if step < 0 else 0, dim_size + stop)
    else:
        stop = min(stop, dim_size)

    return start, stop, step


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        # Note: Storage can be larger than size for non-contiguous views
        # assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        if isinstance(index, tuple):
            aindex = array(index)

        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """
        Permute the dimensions of the tensor.

        Args:
            *order: a permutation of the dimensions

        Returns:
            New `TensorData` with the same storage and a new dimension order.
        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        new_shape = tuple(self.shape[i] for i in order)
        new_strides = tuple(self.strides[i] for i in order)
        return TensorData(self._storage, new_shape, new_strides)

    def slice(self, key: Union[int, slice, Sequence[Union[int, slice]]]) -> TensorData:
        """
        Create a sliced view of the tensor.

        Args:
            key: int, slice, or tuple of ints/slices for indexing

        Returns:
            New TensorData representing the sliced view
        """
        if isinstance(key, (int, slice)):
            key = (key,)

        if len(key) > len(self.shape):
            raise IndexingError(f"Too many indices {len(key)} for tensor of dimension {len(self.shape)}")

        key = tuple(key) + (slice(None),) * (len(self.shape) - len(key))

        new_shape = []
        new_strides = []
        offset = 0

        for dim, (k, dim_size, stride) in enumerate(zip(key, self.shape, self.strides)):
            if isinstance(k, int):
                idx = k
                if idx < 0:
                    idx = dim_size + idx
                if idx < 0 or idx >= dim_size:
                    raise IndexingError(f"Index {k} out of range for dimension {dim} with size {dim_size}")
                offset += idx * stride
            elif isinstance(k, slice):
                start, stop, step = normalize_slice(k, dim_size)
                if step > 0:
                    size = max(0, (stop - start + step - 1) // step)
                else:
                    size = max(0, (stop - start + step + 1) // step)

                new_shape.append(size)
                new_strides.append(stride * step)
                offset += start * stride
            else:
                raise IndexingError(f"Unsupported index type: {type(k)}")

        if len(new_shape) == 0:
            scalar_val = self._storage[offset]
            return TensorData([scalar_val], (1,), (1,))

        return _make_tensor_data_view(self._storage, tuple(new_shape), tuple(new_strides), offset)

    def to_string(self) -> str:
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s


def _make_tensor_data_view(
    storage: Storage, shape: UserShape, strides: UserStrides, offset: int
) -> TensorData:
    """
    Create a TensorData view with an offset into the storage.

    Args:
        storage: The underlying storage array
        shape: Shape of the view
        strides: Strides for the view
        offset: Offset into the storage where the view starts

    Returns:
        TensorData representing the view
    """
    if len(shape) == 0 or prod(shape) == 0:
        # Empty tensor
        return TensorData([], shape, strides)

    if offset > 0:
        view_storage = storage[offset:]
    else:
        view_storage = storage

    return TensorData(view_storage, shape, strides)
