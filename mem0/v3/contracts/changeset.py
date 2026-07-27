"""Immutable semantic change contract emitted by the V3 memory planner."""

from typing import Any, Optional

from pydantic import Field, model_validator

from mem0.v3.contracts.base import FrozenContract, NonEmptyStr
from mem0.v3.contracts.versions import CHANGESET_SCHEMA_VERSION, KERNEL_API_VERSION
from mem0.v3.contracts.payloads import (
    AssertionCreatePayload,
    ObjectMutationPayload,
    RelationCreatePayload,
)
from mem0.v3.domain.enums import (
    LifecycleOperation,
    MemoryObjectType,
    RelationType,
    RetractionTargetType,
)
from mem0.v3.domain.models import (
    CommitmentPayload,
    DecisionPayload,
    Evidence,
    IssuePayload,
    MeetingPayload,
    TopicPayload,
)


_PHYSICAL_INSTRUCTION_KEYS = {"sql", "opensearch_dsl", "outbox_event", "table_name"}

_TYPED_PAYLOAD_FIELDS = {
    MemoryObjectType.MEETING: {
        "external_memory_id",
        "started_at",
        "ended_at",
        "transcript_version",
        "participant_refs",
        "project_links",
        "processing_status",
    },
    MemoryObjectType.TOPIC: {
        "canonical_label",
        "aliases",
        "first_seen_at",
        "last_seen_at",
        "resolution_status",
        "scope_project_ids",
    },
    MemoryObjectType.COMMITMENT: {
        "committed_by",
        "committed_to",
        "action",
        "committed_at",
        "due_at",
        "fulfillment_status",
        "completion_evidence_ids",
    },
    MemoryObjectType.DECISION: {
        "decision_owner",
        "decision",
        "rationale",
        "alternatives",
        "effective_status",
        "effective_from",
    },
    MemoryObjectType.ISSUE: {
        "subtype",
        "severity",
        "affected_object_ids",
        "owner",
        "resolution_status",
        "resolution_evidence_ids",
    },
}
_TYPED_PAYLOAD_MODELS = {
    MemoryObjectType.MEETING: MeetingPayload,
    MemoryObjectType.TOPIC: TopicPayload,
    MemoryObjectType.COMMITMENT: CommitmentPayload,
    MemoryObjectType.DECISION: DecisionPayload,
    MemoryObjectType.ISSUE: IssuePayload,
}
_ALL_TYPED_PAYLOAD_FIELDS = frozenset().union(*_TYPED_PAYLOAD_FIELDS.values())


def _reject_physical_instructions(value: Any, path: str = "payload") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in _PHYSICAL_INSTRUCTION_KEYS:
                raise ValueError(f"{path} contains forbidden physical instruction key: {key}")
            _reject_physical_instructions(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_physical_instructions(nested, f"{path}[{index}]")


class SourceRef(FrozenContract):
    source_type: NonEmptyStr
    source_id: NonEmptyStr
    memory_id: Optional[NonEmptyStr] = None
    transcript_version: Optional[int] = Field(default=None, ge=1)


class EvidenceCreate(FrozenContract):
    logical_ref: NonEmptyStr
    evidence: Evidence


class ObjectMutation(FrozenContract):
    logical_ref: NonEmptyStr
    operation: LifecycleOperation
    object_type: MemoryObjectType
    object_id: Optional[NonEmptyStr] = None
    expected_version: Optional[int] = Field(default=None, ge=0)
    evidence_ids: tuple[NonEmptyStr, ...] = ()
    payload: ObjectMutationPayload

    @model_validator(mode="after")
    def reject_physical_instructions(self) -> "ObjectMutation":
        _reject_physical_instructions(self.payload.model_dump(mode="python"))
        supplied = {
            field_name
            for field_name in _ALL_TYPED_PAYLOAD_FIELDS
            if getattr(self.payload, field_name) not in (None, (), [])
        }
        allowed = _TYPED_PAYLOAD_FIELDS.get(self.object_type, set())
        incompatible = sorted((supplied & _ALL_TYPED_PAYLOAD_FIELDS) - allowed)
        if incompatible:
            raise ValueError(
                f"{self.object_type.value} payload contains fields owned by another "
                f"object type: {', '.join(incompatible)}"
            )
        if self.operation is LifecycleOperation.CREATE:
            missing = tuple(
                field_name
                for field_name in ("canonical_key", "title", "valid_from")
                if getattr(self.payload, field_name) is None
            )
            if missing:
                raise ValueError(
                    "create object mutation requires " + ", ".join(missing)
                )
            payload_model = _TYPED_PAYLOAD_MODELS.get(self.object_type)
            if payload_model is not None:
                payload_model.model_validate(
                    self.payload.model_dump(
                        mode="python",
                        include=allowed,
                        exclude_none=True,
                    )
                )
        return self


class AssertionMutation(FrozenContract):
    logical_ref: NonEmptyStr
    operation: LifecycleOperation
    assertion_id: Optional[NonEmptyStr] = None
    evidence_ids: tuple[NonEmptyStr, ...] = ()
    payload: AssertionCreatePayload

    @model_validator(mode="after")
    def reject_physical_instructions(self) -> "AssertionMutation":
        _reject_physical_instructions(self.payload.model_dump(mode="python"))
        return self


class RelationMutation(FrozenContract):
    logical_ref: NonEmptyStr
    operation: LifecycleOperation
    relation_id: Optional[NonEmptyStr] = None
    source_object_ref: NonEmptyStr
    target_object_ref: NonEmptyStr
    relation_type: RelationType
    evidence_ids: tuple[NonEmptyStr, ...] = ()
    payload: RelationCreatePayload

    @model_validator(mode="after")
    def validate_relation(self) -> "RelationMutation":
        if self.source_object_ref == self.target_object_ref:
            raise ValueError("relation source and target must differ")
        _reject_physical_instructions(self.payload.model_dump(mode="python"))
        return self


class RetractionMutation(FrozenContract):
    logical_ref: NonEmptyStr
    target_type: RetractionTargetType
    target_id: NonEmptyStr
    reason: NonEmptyStr
    evidence_ids: tuple[NonEmptyStr, ...] = ()


class DomainEvent(FrozenContract):
    event_type: NonEmptyStr
    aggregate_ref: NonEmptyStr
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_physical_instructions(self) -> "DomainEvent":
        _reject_physical_instructions(self.payload)
        return self


class ValidatedMemoryChangeSet(FrozenContract):
    changeset_id: NonEmptyStr
    schema_version: int = Field(default=CHANGESET_SCHEMA_VERSION, ge=1)
    kernel_version: int = Field(default=KERNEL_API_VERSION, ge=1)
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

    @model_validator(mode="after")
    def validate_changeset(self) -> "ValidatedMemoryChangeSet":
        if not any(
            (
                self.evidence_creates,
                self.object_mutations,
                self.assertion_mutations,
                self.relation_mutations,
                self.retractions,
            )
        ):
            raise ValueError("a changeset must contain at least one semantic mutation")

        logical_refs = [
            mutation.logical_ref
            for group in (
                self.evidence_creates,
                self.object_mutations,
                self.assertion_mutations,
                self.relation_mutations,
                self.retractions,
            )
            for mutation in group
        ]
        if len(logical_refs) != len(set(logical_refs)):
            raise ValueError("mutation logical_ref values must be unique within a changeset")

        for create in self.evidence_creates:
            if create.evidence.user_id != self.user_id:
                raise ValueError("evidence user_id must match the changeset scope")
            if create.evidence.workspace_id != self.workspace_id:
                raise ValueError("evidence workspace_id must match the changeset scope")

        invalid_versions = [version for version in self.expected_object_versions.values() if version < 0]
        if invalid_versions:
            raise ValueError("expected object versions must be non-negative")
        return self
