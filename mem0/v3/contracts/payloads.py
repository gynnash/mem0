"""Strongly typed semantic payloads carried by a memory changeset."""

from datetime import datetime
from typing import Any, Optional

from pydantic import Field, field_validator, model_validator

from mem0.v3.contracts.base import FrozenContract, NonEmptyStr
from mem0.v3.domain.enums import (
    AssertionValidity,
    EpistemicType,
    FulfillmentStatus,
    Polarity,
    RetentionStatus,
    WorkflowStatus,
)
from mem0.v3.domain.models import FieldProvenance


def _require_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value


class ObjectMutationPayload(FrozenContract):
    """Versioned canonical-object state patch.

    Identity and lifecycle fields are explicit. ``attributes`` is the only
    extension bag and may not contain physical persistence instructions.
    """

    canonical_key: Optional[NonEmptyStr] = None
    title: Optional[NonEmptyStr] = None
    description: Optional[str] = None
    primary_project_id: Optional[NonEmptyStr] = None
    validity: Optional[AssertionValidity] = None
    workflow_status: Optional[WorkflowStatus] = None
    retention_status: Optional[RetentionStatus] = None
    importance: Optional[float] = Field(default=None, ge=0, le=1)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    salience: Optional[float] = Field(default=None, ge=0, le=1)
    attributes: dict[str, Any] = Field(default_factory=dict)
    field_provenance: tuple[FieldProvenance, ...] = ()
    schema_version: Optional[int] = Field(default=None, ge=1)
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None

    # Meeting identity and processing fields are first-class because they are
    # used by deterministic resolution and transcript replay protection.
    external_memory_id: Optional[NonEmptyStr] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    transcript_version: Optional[int] = Field(default=None, ge=1)
    participant_refs: tuple[NonEmptyStr, ...] = ()
    project_links: tuple[NonEmptyStr, ...] = ()
    processing_status: Optional[NonEmptyStr] = None

    # Topic fields are first-class because a topic is promoted only after the
    # precision-first resolver has established a stable identity.
    canonical_label: Optional[NonEmptyStr] = None
    aliases: tuple[NonEmptyStr, ...] = ()
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    resolution_status: Optional[NonEmptyStr] = None
    scope_project_ids: tuple[NonEmptyStr, ...] = ()

    # Decision, commitment and issue fields are explicit because they drive
    # downstream action and must not be accepted as arbitrary JSON keys.
    decision_owner: Optional[NonEmptyStr] = None
    decision: Optional[NonEmptyStr] = None
    rationale: Optional[str] = None
    alternatives: tuple[NonEmptyStr, ...] = ()
    effective_status: Optional[NonEmptyStr] = None
    effective_from: Optional[datetime] = None
    committed_by: Optional[NonEmptyStr] = None
    committed_to: Optional[NonEmptyStr] = None
    action: Optional[NonEmptyStr] = None
    committed_at: Optional[datetime] = None
    due_at: Optional[datetime] = None
    fulfillment_status: Optional[FulfillmentStatus] = None
    completion_evidence_ids: tuple[NonEmptyStr, ...] = ()
    subtype: Optional[NonEmptyStr] = None
    severity: Optional[NonEmptyStr] = None
    affected_object_ids: tuple[NonEmptyStr, ...] = ()
    owner: Optional[NonEmptyStr] = None
    resolution_status: Optional[NonEmptyStr] = None
    resolution_evidence_ids: tuple[NonEmptyStr, ...] = ()

    @field_validator(
        "valid_from",
        "valid_to",
        "started_at",
        "ended_at",
        "first_seen_at",
        "last_seen_at",
        "effective_from",
        "committed_at",
        "due_at",
    )
    @classmethod
    def validate_datetimes(cls, value: Optional[datetime]) -> Optional[datetime]:
        return _require_timezone(value) if value is not None else value

    @model_validator(mode="after")
    def validate_ranges(self) -> "ObjectMutationPayload":
        if self.valid_from and self.valid_to and self.valid_to < self.valid_from:
            raise ValueError("valid_to cannot be earlier than valid_from")
        if self.started_at and self.ended_at and self.ended_at < self.started_at:
            raise ValueError("ended_at cannot be earlier than started_at")
        if self.first_seen_at and self.last_seen_at and self.last_seen_at < self.first_seen_at:
            raise ValueError("last_seen_at cannot be earlier than first_seen_at")
        return self


class AssertionCreatePayload(FrozenContract):
    subject_object_ref: NonEmptyStr
    predicate: NonEmptyStr
    value: Any
    asserted_by_entity_id: Optional[NonEmptyStr] = None
    epistemic_type: EpistemicType
    modality: NonEmptyStr
    polarity: Polarity
    confidence: float = Field(ge=0, le=1)
    asserted_at: datetime
    validity: AssertionValidity = AssertionValidity.ACTIVE

    _validate_asserted_at = field_validator("asserted_at")(_require_timezone)


class RelationCreatePayload(FrozenContract):
    confidence: float = Field(ge=0, le=1)
    epistemic_type: EpistemicType
    validity: AssertionValidity = AssertionValidity.ACTIVE
    valid_from: datetime
    valid_to: Optional[datetime] = None

    _validate_valid_from = field_validator("valid_from")(_require_timezone)

    @field_validator("valid_to")
    @classmethod
    def validate_valid_to(cls, value: Optional[datetime]) -> Optional[datetime]:
        return _require_timezone(value) if value is not None else value

    @model_validator(mode="after")
    def validate_range(self) -> "RelationCreatePayload":
        if self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("valid_to cannot be earlier than valid_from")
        return self
