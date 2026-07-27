"""Structured contracts for source-local transcript understanding."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import Field, field_validator, model_validator

from mem0.v3.contracts.base import FrozenContract, NonEmptyStr


def _require_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value


class ClaimType(str, Enum):
    DECISION = "decision"
    COMMITMENT = "commitment"
    CONDITION = "condition"
    OBJECTION = "objection"
    BLOCKER = "blocker"
    TASK = "task"
    GOAL = "goal"
    PREFERENCE = "preference"


class ClaimModality(str, Enum):
    STATED = "stated"
    PROMISED = "promised"
    PLANNED = "planned"
    CONDITIONAL = "conditional"
    UNCERTAIN = "uncertain"


class ClaimLifecycleSignal(str, Enum):
    NONE = "none"
    RESOLVED = "resolved"
    REOPENED = "reopened"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"


class TranscriptSegment(FrozenContract):
    segment_id: NonEmptyStr
    speaker_ref: Optional[NonEmptyStr] = None
    text: NonEmptyStr
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> "TranscriptSegment":
        if self.end_ms < self.start_ms:
            raise ValueError("segment end_ms cannot be earlier than start_ms")
        return self


class MeetingExtractionInput(FrozenContract):
    user_id: NonEmptyStr
    workspace_id: NonEmptyStr
    memory_id: NonEmptyStr
    transcript_version: int = Field(ge=1)
    transcript_content_hash: Optional[str] = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    title: NonEmptyStr
    started_at: datetime
    ended_at: Optional[datetime] = None
    participant_refs: tuple[NonEmptyStr, ...] = ()
    segments: tuple[TranscriptSegment, ...] = Field(min_length=1)

    _validate_started_at = field_validator("started_at")(_require_timezone)

    @field_validator("ended_at")
    @classmethod
    def validate_ended_at(cls, value: Optional[datetime]) -> Optional[datetime]:
        return _require_timezone(value) if value is not None else value

    @model_validator(mode="after")
    def validate_input(self) -> "MeetingExtractionInput":
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError("meeting ended_at cannot be earlier than started_at")
        segment_ids = [item.segment_id for item in self.segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("transcript segment_id values must be unique")
        return self


class EvidenceSpan(FrozenContract):
    segment_id: NonEmptyStr
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> "EvidenceSpan":
        if self.end_char <= self.start_char:
            raise ValueError("evidence end_char must be greater than start_char")
        return self


class ExtractedProjectMention(FrozenContract):
    mention: NonEmptyStr
    evidence_spans: tuple[EvidenceSpan, ...] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class ExtractedClaim(FrozenContract):
    claim_id: NonEmptyStr
    claim_type: ClaimType
    text: NonEmptyStr
    owner_mention: Optional[NonEmptyStr] = None
    due_at: Optional[datetime] = None
    condition: Optional[NonEmptyStr] = None
    negated: bool = False
    modality: ClaimModality
    lifecycle_signal: ClaimLifecycleSignal = ClaimLifecycleSignal.NONE
    project_mentions: tuple[NonEmptyStr, ...] = ()
    object_mentions: tuple[NonEmptyStr, ...] = ()
    evidence_spans: tuple[EvidenceSpan, ...] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)

    @field_validator("due_at")
    @classmethod
    def validate_due_at(cls, value: Optional[datetime]) -> Optional[datetime]:
        return _require_timezone(value) if value is not None else value


class SessionTopicCandidate(FrozenContract):
    candidate_id: NonEmptyStr
    label: NonEmptyStr
    explicit_name: bool = False
    project_mentions: tuple[NonEmptyStr, ...] = ()
    object_anchors: tuple[NonEmptyStr, ...] = ()
    evidence_spans: tuple[EvidenceSpan, ...] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class LocalExtractionResult(FrozenContract):
    extraction_version: NonEmptyStr
    claims: tuple[ExtractedClaim, ...] = ()
    project_mentions: tuple[ExtractedProjectMention, ...] = ()
    topic_candidates: tuple[SessionTopicCandidate, ...] = ()
    warnings: tuple[NonEmptyStr, ...] = ()
