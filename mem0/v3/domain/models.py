"""Immutable domain models for the source-backed V3 memory graph."""

from datetime import datetime
import re
from typing import Any, Optional

from pydantic import Field, field_validator, model_validator

from mem0.v3.contracts.base import FrozenContract, NonEmptyStr
from mem0.v3.domain.enums import (
    AssertionValidity,
    EpistemicType,
    EvidenceValidity,
    FulfillmentStatus,
    MemoryObjectType,
    Polarity,
    RetentionStatus,
    RelationType,
    WorkflowStatus,
)


def _require_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value


class Evidence(FrozenContract):
    evidence_id: NonEmptyStr
    user_id: NonEmptyStr
    workspace_id: NonEmptyStr
    source_type: NonEmptyStr
    source_id: NonEmptyStr
    memory_id: Optional[NonEmptyStr] = None
    transcript_version: Optional[int] = Field(default=None, ge=1)
    speaker_id: Optional[NonEmptyStr] = None
    start_ms: Optional[int] = Field(default=None, ge=0)
    end_ms: Optional[int] = Field(default=None, ge=0)
    content: NonEmptyStr
    content_hash: NonEmptyStr
    recorded_at: datetime
    created_at: datetime
    validity: EvidenceValidity = EvidenceValidity.ACTIVE

    _validate_recorded_at = field_validator("recorded_at")(_require_timezone)
    _validate_created_at = field_validator("created_at")(_require_timezone)

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("content_hash must be a lowercase SHA-256 hex digest")
        return value

    @model_validator(mode="after")
    def validate_span(self) -> "Evidence":
        if (self.start_ms is None) != (self.end_ms is None):
            raise ValueError("start_ms and end_ms must be supplied together")
        if self.start_ms is not None and self.end_ms is not None and self.end_ms < self.start_ms:
            raise ValueError("end_ms cannot be earlier than start_ms")
        return self


class Assertion(FrozenContract):
    assertion_id: NonEmptyStr
    user_id: NonEmptyStr
    workspace_id: NonEmptyStr
    subject_object_id: NonEmptyStr
    predicate: NonEmptyStr
    value: Any
    asserted_by_entity_id: Optional[NonEmptyStr] = None
    epistemic_type: EpistemicType
    modality: NonEmptyStr
    polarity: Polarity
    confidence: float = Field(ge=0, le=1)
    evidence_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    asserted_at: datetime
    validity: AssertionValidity = AssertionValidity.ACTIVE

    _validate_asserted_at = field_validator("asserted_at")(_require_timezone)


class FieldProvenance(FrozenContract):
    field_name: NonEmptyStr
    evidence_ids: tuple[NonEmptyStr, ...] = ()
    model_version: Optional[NonEmptyStr] = None
    user_confirmed: bool = False
    user_locked: bool = False
    corrected_at: Optional[datetime] = None

    @field_validator("corrected_at")
    @classmethod
    def validate_corrected_at(cls, value: Optional[datetime]) -> Optional[datetime]:
        return _require_timezone(value) if value is not None else value


class CanonicalObject(FrozenContract):
    object_id: NonEmptyStr
    user_id: NonEmptyStr
    workspace_id: NonEmptyStr
    object_type: MemoryObjectType
    canonical_key: NonEmptyStr
    title: NonEmptyStr
    description: str = ""
    primary_project_id: Optional[NonEmptyStr] = None
    validity: AssertionValidity = AssertionValidity.ACTIVE
    workflow_status: WorkflowStatus = WorkflowStatus.PROPOSED
    retention_status: RetentionStatus = RetentionStatus.WORKING
    importance: float = Field(default=0, ge=0, le=1)
    confidence: float = Field(default=0, ge=0, le=1)
    salience: float = Field(default=0, ge=0, le=1)
    current_version_id: NonEmptyStr
    attributes: dict[str, Any] = Field(default_factory=dict)
    field_provenance: tuple[FieldProvenance, ...] = ()
    schema_version: int = Field(default=1, ge=1)
    created_at: datetime
    updated_at: datetime
    valid_from: datetime
    valid_to: Optional[datetime] = None
    lock_version: int = Field(default=0, ge=0)

    _validate_created_at = field_validator("created_at")(_require_timezone)
    _validate_updated_at = field_validator("updated_at")(_require_timezone)
    _validate_valid_from = field_validator("valid_from")(_require_timezone)

    @field_validator("valid_to")
    @classmethod
    def validate_valid_to_timezone(cls, value: Optional[datetime]) -> Optional[datetime]:
        return _require_timezone(value) if value is not None else value

    @model_validator(mode="after")
    def validate_object(self) -> "CanonicalObject":
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be earlier than created_at")
        if self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("valid_to cannot be earlier than valid_from")
        field_names = [item.field_name for item in self.field_provenance]
        if len(field_names) != len(set(field_names)):
            raise ValueError("field provenance entries must have unique field names")
        return self


class ObjectVersion(FrozenContract):
    version_id: NonEmptyStr
    object_id: NonEmptyStr
    version: int = Field(ge=1)
    state_version: int = Field(ge=0)
    snapshot: dict[str, Any]
    evidence_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    created_at: datetime

    _validate_created_at = field_validator("created_at")(_require_timezone)


class Relation(FrozenContract):
    relation_id: NonEmptyStr
    user_id: NonEmptyStr
    workspace_id: NonEmptyStr
    source_object_id: NonEmptyStr
    relation_type: RelationType
    target_object_id: NonEmptyStr
    confidence: float = Field(ge=0, le=1)
    epistemic_type: EpistemicType
    evidence_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    validity: AssertionValidity = AssertionValidity.ACTIVE
    valid_from: datetime
    valid_to: Optional[datetime] = None

    _validate_valid_from = field_validator("valid_from")(_require_timezone)

    @field_validator("valid_to")
    @classmethod
    def validate_valid_to_timezone(cls, value: Optional[datetime]) -> Optional[datetime]:
        return _require_timezone(value) if value is not None else value

    @model_validator(mode="after")
    def validate_relation(self) -> "Relation":
        if self.source_object_id == self.target_object_id:
            raise ValueError("relation source and target must differ")
        if self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("valid_to cannot be earlier than valid_from")
        return self


class MeetingPayload(FrozenContract):
    external_memory_id: NonEmptyStr
    started_at: datetime
    ended_at: Optional[datetime] = None
    transcript_version: int = Field(ge=1)
    participant_refs: tuple[NonEmptyStr, ...] = ()
    project_links: tuple[NonEmptyStr, ...] = ()
    processing_status: NonEmptyStr

    _validate_started_at = field_validator("started_at")(_require_timezone)

    @field_validator("ended_at")
    @classmethod
    def validate_ended_at_timezone(cls, value: Optional[datetime]) -> Optional[datetime]:
        return _require_timezone(value) if value is not None else value

    @model_validator(mode="after")
    def validate_times(self) -> "MeetingPayload":
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError("ended_at cannot be earlier than started_at")
        return self


class TopicPayload(FrozenContract):
    canonical_label: NonEmptyStr
    aliases: tuple[NonEmptyStr, ...] = ()
    first_seen_at: datetime
    last_seen_at: datetime
    resolution_status: NonEmptyStr
    scope_project_ids: tuple[NonEmptyStr, ...] = ()

    _validate_first_seen_at = field_validator("first_seen_at")(_require_timezone)
    _validate_last_seen_at = field_validator("last_seen_at")(_require_timezone)

    @model_validator(mode="after")
    def validate_times(self) -> "TopicPayload":
        if self.last_seen_at < self.first_seen_at:
            raise ValueError("last_seen_at cannot be earlier than first_seen_at")
        return self


class CommitmentPayload(FrozenContract):
    committed_by: NonEmptyStr
    committed_to: Optional[NonEmptyStr] = None
    action: NonEmptyStr
    committed_at: datetime
    due_at: Optional[datetime] = None
    fulfillment_status: FulfillmentStatus
    completion_evidence_ids: tuple[NonEmptyStr, ...] = ()

    _validate_committed_at = field_validator("committed_at")(_require_timezone)

    @field_validator("due_at")
    @classmethod
    def validate_due_at_timezone(cls, value: Optional[datetime]) -> Optional[datetime]:
        return _require_timezone(value) if value is not None else value

    @model_validator(mode="after")
    def validate_completion(self) -> "CommitmentPayload":
        if self.fulfillment_status is FulfillmentStatus.COMPLETED and not self.completion_evidence_ids:
            raise ValueError("completed commitments require completion evidence")
        return self


class DecisionPayload(FrozenContract):
    decision_owner: Optional[NonEmptyStr] = None
    decision: NonEmptyStr
    rationale: str = ""
    alternatives: tuple[NonEmptyStr, ...] = ()
    effective_status: NonEmptyStr
    effective_from: datetime

    _validate_effective_from = field_validator("effective_from")(_require_timezone)


class IssuePayload(FrozenContract):
    subtype: NonEmptyStr
    severity: NonEmptyStr
    affected_object_ids: tuple[NonEmptyStr, ...] = ()
    owner: Optional[NonEmptyStr] = None
    resolution_status: NonEmptyStr
    resolution_evidence_ids: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def validate_resolution(self) -> "IssuePayload":
        if self.resolution_status == "resolved" and not self.resolution_evidence_ids:
            raise ValueError("resolved issues require resolution evidence")
        return self
