class InvalidShapeError(Exception):
    def __init__(self, message="Shape is invalid"):
        super().__init__(message)


class DimensionMismatchError(Exception):
    def __init__(self, message="Dimensions do not match"):
        super().__init__(message)


class NonNumericDataError(Exception):
    def __init__(self, item, message=None):
        if message is None:
            message = f"Expected numeric data, got: {item!r}"
        super().__init__(message)


class InvalidOperationError(Exception):
    def __init__(self, message="Operation not allowed on these arrays"):
        super().__init__(message)
