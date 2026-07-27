"""Shared primitives for immutable V3 contracts."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class FrozenContract(BaseModel):
    """Base contract that rejects unknown fields and top-level mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)
