from exceptions.errors import DimensionMismatchError, InvalidOperationError, NonNumericDataError

class Array:
    def __init__(self, data):
        self._check_numeric(data)
        self.data = data
        self.shape = self._get_shape(data)
        self.ndim = len(self.shape)

    def _check_numeric(self, item):
        if isinstance(item, list):
            for x in item:
                self._check_numeric(x)
        elif not isinstance(item, (int, float)):
            raise NonNumericDataError(item)

    def _get_shape(self, data):
        shape = []
        current = data
        while isinstance(current, list):
            shape.append(len(current))
            current = current[0]
        return tuple(shape)

    def flatten(self):
        result = []
        def collect(item):
            if isinstance(item, list):
                for x in item:
                    collect(x)
            else:
                result.append(item)
        collect(self.data)
        return Array(result)

    def __repr__(self):
        return f"Array(shape={self.shape})\n{self.data}"

 
    def _apply(self, other, op):
        # Other is also an Array
        if isinstance(other, Array):
            if self.shape != other.shape:
                raise DimensionMismatchError()
            def recurse(a, b):
                if isinstance(a, list):
                    return [recurse(x, y) for x, y in zip(a, b)]
                return op(a, b)           
            return Array(recurse(self.data, other.data))

        # Other is a number
        elif isinstance(other, (int, float)):
            def recurse(a):
                if isinstance(a, list):
                    return [recurse(x) for x in a]
                return op(a, other)
            return Array(recurse(self.data))

        else:
            raise InvalidOperationError()

    # Magic methods all use _apply 
    def __add__(self, other):
        return self._apply(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self._apply(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self._apply(other, lambda a, b: a * b)

    def __pow__(self, other):
        return self._apply(other, lambda a, b: a ** b)
