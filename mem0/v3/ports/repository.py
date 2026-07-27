"""Read-only state and search ports implemented by Summora adapters."""

from typing import Any, Mapping, Optional, Protocol, Sequence

from pydantic import Field

from mem0.v3.contracts.base import FrozenContract, NonEmptyStr
from mem0.v3.contracts.snapshot import MemoryReadSnapshot


class SearchCandidate(FrozenContract):
    object_id: Optional[NonEmptyStr] = None
    evidence_id: Optional[NonEmptyStr] = None
    score: float = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryReadPort(Protocol):
    def get_object(
        self,
        *,
        snapshot: MemoryReadSnapshot,
        object_id: str,
    ) -> Optional[Mapping[str, Any]]:
        """Read one object at the supplied source-of-truth watermark."""

    def get_objects(
        self,
        *,
        snapshot: MemoryReadSnapshot,
        object_ids: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        """Read multiple objects at one consistent watermark."""


class MemorySearchPort(Protocol):
    def search(
        self,
        *,
        snapshot: MemoryReadSnapshot,
        query: str,
        limit: int,
    ) -> Sequence[SearchCandidate]:
        """Recall candidates only; callers must verify current truth through MemoryReadPort."""
