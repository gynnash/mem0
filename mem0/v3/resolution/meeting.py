"""Deterministic Meeting identity and optimistic-update selection."""

from typing import Optional, Tuple

from mem0.v3.domain import LifecycleOperation
from mem0.v3.resolution.models import MeetingObjectState


class MeetingResolver:
    @staticmethod
    def canonical_key(memory_id: str) -> str:
        return f"meeting:{memory_id}"

    def resolve(
        self,
        *,
        memory_id: str,
        transcript_version: int,
        transcript_content_hash: Optional[str],
        existing: Optional[MeetingObjectState],
    ) -> Tuple[LifecycleOperation, Optional[str], Optional[int]]:
        expected_key = self.canonical_key(memory_id)
        if existing is None:
            return LifecycleOperation.CREATE, None, None
        if existing.canonical_key != expected_key:
            raise ValueError("existing meeting binding violates deterministic identity")
        if transcript_version < existing.transcript_version:
            raise ValueError("older transcript version cannot replace current meeting state")
        if (
            transcript_version == existing.transcript_version
            and transcript_content_hash
            and existing.transcript_content_hash
            and transcript_content_hash != existing.transcript_content_hash
        ):
            raise ValueError("transcript content changed without a new version")
        return LifecycleOperation.UPDATE, existing.object_id, existing.lock_version
