from pydantic import BaseModel, field_validator
from typing import Any, List


class ArrayInput(BaseModel):
    data: Any

    @field_validator("data")
    @classmethod
    def must_be_list(cls, v):
        if not isinstance(v, list):
            raise ValueError("data must be a list")
        return v


class ShapeInput(BaseModel):
    shape: List[int]

    @field_validator("shape")
    @classmethod
    def must_be_positive(cls, v):
        for dim in v:
            if dim <= 0:
                raise ValueError(f"Each dimension must be positive, got {dim}")
        return v
