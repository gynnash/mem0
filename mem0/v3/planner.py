"""Pure semantic planning entrypoint; contains no persistence behavior."""

from typing import Optional, Protocol, Sequence

from pydantic import Field

from mem0.v3.contracts.base import FrozenContract, NonEmptyStr
from mem0.v3.contracts.changeset import (
    AssertionMutation,
    DomainEvent,
    EvidenceCreate,
    ObjectMutation,
    RelationMutation,
    RetractionMutation,
    SourceRef,
    ValidatedMemoryChangeSet,
)


class MemoryChangeDraft(FrozenContract):
    changeset_id: NonEmptyStr
    user_id: NonEmptyStr
    workspace_id: NonEmptyStr
    source_ref: SourceRef
    base_state_version: int = Field(ge=0)
    expected_object_versions: dict[NonEmptyStr, int] = Field(default_factory=dict)
    evidence_creates: tuple[EvidenceCreate, ...] = ()
    object_mutations: tuple[ObjectMutation, ...] = ()
    assertion_mutations: tuple[AssertionMutation, ...] = ()
    relation_mutations: tuple[RelationMutation, ...] = ()
    retractions: tuple[RetractionMutation, ...] = ()
    domain_events: tuple[DomainEvent, ...] = ()
    warnings: tuple[NonEmptyStr, ...] = ()


class ChangeSetRule(Protocol):
    def validate(self, changeset: ValidatedMemoryChangeSet) -> None:
        """Raise ValueError when a domain invariant is violated."""


class MemoryPlanner:
    """Build a validated semantic changeset without touching external state."""

    def __init__(self, rules: Optional[Sequence[ChangeSetRule]] = None) -> None:
        if rules is None:
            from mem0.v3.lifecycle import DEFAULT_CHANGESET_RULES

            rules = DEFAULT_CHANGESET_RULES
        self._rules = tuple(rules)

    def plan(self, draft: MemoryChangeDraft) -> ValidatedMemoryChangeSet:
        # Preserve nested ``fields_set`` so Summora can distinguish a patch
        # field that was explicitly cleared from a schema default that was not
        # supplied by the semantic planner.
        changeset = ValidatedMemoryChangeSet.model_validate(
            draft.model_dump(exclude_unset=True)
        )
        for rule in self._rules:
            rule.validate(changeset)
        return changeset
