"""Model abstraction owned by mem0 and implemented by Summora."""

from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, Field

from mem0.v3.contracts.base import FrozenContract, NonEmptyStr


class ModelMessage(FrozenContract):
    role: Literal["system", "user", "assistant", "tool"]
    content: NonEmptyStr


class StructuredModelRequest(FrozenContract):
    operation: NonEmptyStr
    messages: tuple[ModelMessage, ...]
    timeout_ms: int = Field(gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


ModelOutput = TypeVar("ModelOutput", bound=BaseModel)


class ModelPort(Protocol):
    def generate_structured(
        self,
        *,
        request: StructuredModelRequest,
        response_model: type[ModelOutput],
    ) -> ModelOutput:
        """Return a validated structured response without exposing provider configuration."""
