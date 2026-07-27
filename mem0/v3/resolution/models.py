"""Pure resolution inputs and decisions for global memory alignment."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import Field

from mem0.v3.contracts.base import FrozenContract, NonEmptyStr
from mem0.v3.domain import MemoryObjectType


class ProjectLinkStatus(str, Enum):
    LINKED = "linked"
    MULTI_LINKED = "multi_linked"
    MULTI_CANDIDATE = "multi_candidate"
    UNRESOLVED = "unresolved"
    RETRACTED = "retracted"


class ProjectAnchorType(str, Enum):
    EXPLICIT_PROJECT_MENTION = "explicit_project_mention"
    LINKED_OBJECT = "linked_object"
    LOCAL_UNIQUE_ANCHOR = "local_unique_anchor"
    MEETING_PRIMARY = "meeting_primary"
    OWNER = "owner"
    RECENCY = "recency"
    SEMANTIC_SIMILARITY = "semantic_similarity"


class ProjectCandidate(FrozenContract):
    project_object_id: NonEmptyStr
    score: float = Field(ge=0, le=1)
    anchor_types: tuple[ProjectAnchorType, ...] = ()
    anchor_refs: tuple[NonEmptyStr, ...] = ()
    evidence_ids: tuple[NonEmptyStr, ...] = ()


class ProjectLink(FrozenContract):
    project_object_id: NonEmptyStr
    role: NonEmptyStr
    confidence: float = Field(ge=0, le=1)


class ProjectLinkDecision(FrozenContract):
    task_object_ref: NonEmptyStr
    decision: ProjectLinkStatus
    primary_project_object_id: Optional[NonEmptyStr] = None
    confidence: float = Field(ge=0, le=1)
    margin_to_second_candidate: float = Field(ge=0, le=1)
    anchor_types: tuple[ProjectAnchorType, ...] = ()
    evidence_ids: tuple[NonEmptyStr, ...] = ()
    candidate_project_ids: tuple[NonEmptyStr, ...] = ()
    project_links: tuple[ProjectLink, ...] = ()
    resolver_version: NonEmptyStr


class TopicResolutionStatus(str, Enum):
    LINKED = "linked"
    NEW_TOPIC = "new_topic"
    MULTI_CANDIDATE = "multi_candidate"
    UNRESOLVED = "unresolved"
    MERGED = "merged"
    RETRACTED = "retracted"


class TopicCandidateMatch(FrozenContract):
    topic_object_id: NonEmptyStr
    score: float = Field(ge=0, le=1)
    lock_version: Optional[int] = Field(default=None, ge=0)
    last_seen_at: Optional[datetime] = None
    scope_project_ids: tuple[NonEmptyStr, ...] = ()
    explicit_name_match: bool = False
    project_anchor_match: bool = False
    object_anchor_match: bool = False
    scope_compatible: bool = True


class TopicResolutionDecision(FrozenContract):
    session_candidate_id: NonEmptyStr
    decision: TopicResolutionStatus
    topic_object_id: Optional[NonEmptyStr] = None
    confidence: float = Field(ge=0, le=1)
    candidate_topic_ids: tuple[NonEmptyStr, ...] = ()
    resolver_version: NonEmptyStr
    shadow_only: bool = True


class EntityResolutionStatus(str, Enum):
    LINKED = "linked"
    NEW_SESSION_ENTITY = "new_session_entity"
    UNRESOLVED = "unresolved"


class EntityCandidate(FrozenContract):
    entity_object_id: NonEmptyStr
    score: float = Field(ge=0, le=1)
    same_external_binding: bool = False
    user_confirmed_alias: bool = False


class EntityResolutionDecision(FrozenContract):
    participant_ref: NonEmptyStr
    decision: EntityResolutionStatus
    entity_object_id: Optional[NonEmptyStr] = None
    confidence: float = Field(ge=0, le=1)
    resolver_version: NonEmptyStr


class ObjectLinkCandidate(FrozenContract):
    object_id: NonEmptyStr
    object_type: MemoryObjectType
    canonical_key: NonEmptyStr
    title: NonEmptyStr
    lock_version: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    anchor_types: tuple[NonEmptyStr, ...] = ()
    attributes: dict[str, Any] = Field(default_factory=dict)
    current_state: dict[str, Any] = Field(default_factory=dict)
    field_provenance: tuple[dict[str, Any], ...] = ()
    workflow_status: Optional[NonEmptyStr] = None


class MeetingObjectState(FrozenContract):
    object_id: NonEmptyStr
    canonical_key: NonEmptyStr
    lock_version: int = Field(ge=0)
    transcript_version: int = Field(ge=1)
    transcript_content_hash: Optional[str] = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )


class AlignmentContext(FrozenContract):
    base_state_version: int = Field(ge=0)
    meeting_object: Optional[MeetingObjectState] = None
    object_candidates_by_claim: dict[str, tuple[ObjectLinkCandidate, ...]] = Field(
        default_factory=dict
    )
    project_candidates_by_claim: dict[str, tuple[ProjectCandidate, ...]] = Field(
        default_factory=dict
    )
    project_candidates_by_mention: dict[str, tuple[ProjectCandidate, ...]] = Field(
        default_factory=dict
    )
    topic_matches_by_candidate: dict[str, tuple[TopicCandidateMatch, ...]] = Field(
        default_factory=dict
    )
    entity_candidates_by_participant: dict[str, tuple[EntityCandidate, ...]] = Field(
        default_factory=dict
    )
    now: datetime
