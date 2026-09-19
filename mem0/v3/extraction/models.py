"""Structured contracts for source-local transcript understanding."""

from datetime import datetime
import json
from enum import Enum
from typing import Optional

from pydantic import Field, ValidationError, field_validator, model_validator

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


class TaskExecutionIntent(str, Enum):
    ASSIGNED = "assigned"
    SELF_COMMITTED = "self_committed"


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


class EvidenceUnit(FrozenContract):
    evidence_unit_id: NonEmptyStr
    segment_id: NonEmptyStr
    text: NonEmptyStr


class UnitBackedEpisodicEvidence(FrozenContract):
    evidence_id: NonEmptyStr
    content: NonEmptyStr
    primary_speaker_ref: Optional[NonEmptyStr] = None
    evidence_unit_ids: tuple[NonEmptyStr, ...] = Field(
        min_length=1,
        description="Exact source unit IDs. Prefer 1 to 4 consecutive units; the kernel splits longer or discontinuous citations without dropping units.",
    )
    confidence: float = Field(ge=0, le=1)


class EpisodicEvidence(FrozenContract):
    evidence_id: NonEmptyStr
    content: NonEmptyStr
    primary_speaker_ref: Optional[NonEmptyStr] = None
    source_spans: tuple[EvidenceSpan, ...] = Field(min_length=1, max_length=4)
    confidence: float = Field(ge=0, le=1)


class UnitBackedEntityMention(FrozenContract):
    mention: NonEmptyStr
    episodic_evidence_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class ExtractedEntityMention(FrozenContract):
    mention: NonEmptyStr
    episodic_evidence_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class UnitBackedExtractedProjectMention(FrozenContract):
    mention: NonEmptyStr
    episodic_evidence_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class UnitBackedExtractedClaim(FrozenContract):
    claim_id: NonEmptyStr
    claim_type: ClaimType
    text: NonEmptyStr
    owner_mention: Optional[NonEmptyStr] = None
    action: Optional[NonEmptyStr] = None
    task_intent: Optional[TaskExecutionIntent] = None
    due_at: Optional[datetime] = None
    condition: Optional[NonEmptyStr] = None
    negated: bool = False
    modality: ClaimModality
    lifecycle_signal: ClaimLifecycleSignal = ClaimLifecycleSignal.NONE
    project_mentions: tuple[NonEmptyStr, ...] = ()
    object_mentions: tuple[NonEmptyStr, ...] = ()
    episodic_evidence_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)

    @field_validator("due_at")
    @classmethod
    def validate_due_at(cls, value: Optional[datetime]) -> Optional[datetime]:
        return _require_timezone(value) if value is not None else value

    @model_validator(mode="after")
    def validate_task_fields(self) -> "UnitBackedExtractedClaim":
        _validate_task_fields(self)
        return self


class UnitBackedSessionTopicCandidate(FrozenContract):
    candidate_id: NonEmptyStr
    label: NonEmptyStr
    explicit_name: bool = False
    project_mentions: tuple[NonEmptyStr, ...] = ()
    object_anchors: tuple[NonEmptyStr, ...] = ()
    episodic_evidence_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class UnitBackedLocalExtractionResult(FrozenContract):
    extraction_version: NonEmptyStr
    episodic_evidence: tuple[UnitBackedEpisodicEvidence, ...] = ()
    claims: tuple[UnitBackedExtractedClaim, ...] = ()
    entity_mentions: tuple[UnitBackedEntityMention, ...] = ()
    project_mentions: tuple[UnitBackedExtractedProjectMention, ...] = ()
    topic_candidates: tuple[UnitBackedSessionTopicCandidate, ...] = ()
    warnings: tuple[NonEmptyStr, ...] = ()


class ExtractedProjectMention(FrozenContract):
    mention: NonEmptyStr
    episodic_evidence_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class ExtractedClaim(FrozenContract):
    claim_id: NonEmptyStr
    claim_type: ClaimType
    text: NonEmptyStr
    owner_mention: Optional[NonEmptyStr] = None
    action: Optional[NonEmptyStr] = None
    task_intent: Optional[TaskExecutionIntent] = None
    due_at: Optional[datetime] = None
    condition: Optional[NonEmptyStr] = None
    negated: bool = False
    modality: ClaimModality
    lifecycle_signal: ClaimLifecycleSignal = ClaimLifecycleSignal.NONE
    project_mentions: tuple[NonEmptyStr, ...] = ()
    object_mentions: tuple[NonEmptyStr, ...] = ()
    episodic_evidence_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)

    @field_validator("due_at")
    @classmethod
    def validate_due_at(cls, value: Optional[datetime]) -> Optional[datetime]:
        return _require_timezone(value) if value is not None else value

    @model_validator(mode="after")
    def validate_task_fields(self) -> "ExtractedClaim":
        _validate_task_fields(self)
        return self


def _validate_task_fields(claim) -> None:
    task_fields = (claim.action, claim.task_intent)
    if claim.claim_type is ClaimType.TASK:
        if claim.owner_mention is None or any(value is None for value in task_fields):
            raise ValueError(
                "task claim requires owner_mention, action and task_intent"
            )
        if claim.modality not in {
            ClaimModality.PROMISED,
            ClaimModality.PLANNED,
            ClaimModality.CONDITIONAL,
        }:
            raise ValueError(
                "task claim requires promised, planned or conditional modality"
            )
    elif any(value is not None for value in task_fields):
        raise ValueError("action and task_intent are only valid for task claims")


class SessionTopicCandidate(FrozenContract):
    candidate_id: NonEmptyStr
    label: NonEmptyStr
    explicit_name: bool = False
    project_mentions: tuple[NonEmptyStr, ...] = ()
    object_anchors: tuple[NonEmptyStr, ...] = ()
    episodic_evidence_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class LocalExtractionResult(FrozenContract):
    extraction_version: NonEmptyStr
    episodic_evidence: tuple[EpisodicEvidence, ...] = ()
    claims: tuple[ExtractedClaim, ...] = ()
    entity_mentions: tuple[ExtractedEntityMention, ...] = ()
    project_mentions: tuple[ExtractedProjectMention, ...] = ()
    topic_candidates: tuple[SessionTopicCandidate, ...] = ()
    warnings: tuple[NonEmptyStr, ...] = ()


ISOLATION_WARNING_PREFIX = "extraction_quarantine:"


def isolation_warning(field, index, item_id, reason):
    return ISOLATION_WARNING_PREFIX + json.dumps({
        "field": field, "index": index, "item_id": item_id, "reason": reason,
    }, sort_keys=True)


def extraction_isolation_report(warnings):
    return tuple(json.loads(value[len(ISOLATION_WARNING_PREFIX):])
                 for value in warnings if value.startswith(ISOLATION_WARNING_PREFIX))


class PartialUnitBackedLocalExtractionResult(UnitBackedLocalExtractionResult):
    """Strict envelope with independently validated items; used only at model ingress."""

    @model_validator(mode="before")
    @classmethod
    def isolate_invalid_items(cls, value):
        if not isinstance(value, dict):
            return value
        value = dict(value)
        warnings = value.get("warnings", ())
        if not isinstance(warnings, (list, tuple)) or not all(isinstance(w, str) and w for w in warnings):
            raise ValueError("extraction warnings must be a sequence of nonempty strings")
        warnings = [w for w in warnings if not w.startswith(ISOLATION_WARNING_PREFIX)]
        contracts = {
            "episodic_evidence": UnitBackedEpisodicEvidence,
            "claims": UnitBackedExtractedClaim,
            "entity_mentions": UnitBackedEntityMention,
            "project_mentions": UnitBackedExtractedProjectMention,
            "topic_candidates": UnitBackedSessionTopicCandidate,
        }
        invalid_evidence_ids = set()
        for field, contract in contracts.items():
            items = value.get(field, ())
            if not isinstance(items, (list, tuple)):
                raise ValueError(f"extraction {field} must be a sequence")
            accepted = []
            for index, item in enumerate(items):
                try:
                    accepted.append(contract.model_validate(item))
                except ValidationError:
                    item_id = next((item.get(key) for key in ("evidence_id", "claim_id", "candidate_id")
                                    if isinstance(item.get(key), str)), None) if isinstance(item, dict) else None
                    warnings.append(isolation_warning(field, index, item_id, "invalid_item_schema"))
                    if field == "episodic_evidence" and item_id is not None:
                        invalid_evidence_ids.add(item_id)
            value[field] = accepted
        value["episodic_evidence"] = [item for item in value["episodic_evidence"]
                                       if item.evidence_id not in invalid_evidence_ids]
        value["warnings"] = warnings
        return value
