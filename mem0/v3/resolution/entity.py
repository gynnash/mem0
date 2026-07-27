"""Conservative entity resolution; no model-guessed cross-meeting identity."""

from mem0.v3.resolution.models import (
    EntityCandidate,
    EntityResolutionDecision,
    EntityResolutionStatus,
)


class EntityResolver:
    def __init__(self, resolver_version: str = "entity-resolver/v1") -> None:
        self._resolver_version = resolver_version

    def resolve(
        self,
        *,
        participant_ref: str,
        candidates: tuple[EntityCandidate, ...],
    ) -> EntityResolutionDecision:
        ordered = tuple(
            sorted(candidates, key=lambda item: (-item.score, item.entity_object_id))
        )
        verified = tuple(
            item
            for item in ordered
            if item.same_external_binding or item.user_confirmed_alias
        )
        if verified and (
            len(verified) == 1
            or verified[0].score - verified[1].score >= 0.15
        ):
            return EntityResolutionDecision(
                participant_ref=participant_ref,
                decision=EntityResolutionStatus.LINKED,
                entity_object_id=verified[0].entity_object_id,
                confidence=verified[0].score,
                resolver_version=self._resolver_version,
            )
        if participant_ref.strip():
            return EntityResolutionDecision(
                participant_ref=participant_ref,
                decision=EntityResolutionStatus.NEW_SESSION_ENTITY,
                confidence=1,
                resolver_version=self._resolver_version,
            )
        return EntityResolutionDecision(
            participant_ref="unresolved",
            decision=EntityResolutionStatus.UNRESOLVED,
            confidence=0,
            resolver_version=self._resolver_version,
        )
