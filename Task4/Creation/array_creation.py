from core.ndarray import Array
from exceptions.errors import DimensionMismatchError
from validation.schemas import ArrayInput, ShapeInput

def array(data):
    ArrayInput(data=data)
    
    if not isinstance(data, list):
        return

    first_len = None
    for x in data:
        if isinstance(x, list):
            first_len = len(x)
            break

    for x in data:
        if isinstance(x, list):
            if first_len is not None and len(x) != first_len:
                raise DimensionMismatchError()
            array(x)

    return Array(data)


def zeros(shape):
    ShapeInput(shape=list(shape))
    if not shape:
        return 0
    lst = []
    for i in range(shape[0]):
        sub = zeros(shape[1:])
        if isinstance(sub, Array):
            sub = sub.data
        lst.append(sub)
    return Array(lst)


def ones(shape):
    ShapeInput(shape=list(shape))
    if not shape:
        return 1
    lst = []
    for i in range(shape[0]):
        sub = ones(shape[1:])
        if isinstance(sub, Array):
            sub = sub.data
        lst.append(sub)
    return Array(lst)


def eye(m, n=None, k=0):
    if n is None:
        n = m
    matrix = []
    for i in range(m):
        row = []
        for j in range(n):
            if j - i == k:
                row.append(1)
            else:
                row.append(0)
        matrix.append(row)
    return Array(matrix)
