"""Stable domain enums used by V3 contracts."""

from enum import Enum


class LifecycleOperation(str, Enum):
    CREATE = "create"
    CONFIRM = "confirm"
    ENRICH = "enrich"
    UPDATE = "update"
    RESOLVE = "resolve"
    REOPEN = "reopen"
    SUPERSEDE = "supersede"
    CONTRADICT = "contradict"
    MERGE = "merge"
    SPLIT = "split"
    RETRACT = "retract"
    ARCHIVE = "archive"
    DELETE = "delete"


class MemoryObjectType(str, Enum):
    ENTITY = "entity"
    PROJECT = "project"
    MEETING = "meeting"
    TOPIC = "topic"
    GOAL = "goal"
    DECISION = "decision"
    COMMITMENT = "commitment"
    TASK = "task"
    ISSUE = "issue"
    PREFERENCE = "preference"


class RelationType(str, Enum):
    DISCUSSES = "discusses"
    DISCUSSES_TOPIC = "discusses_topic"
    PARTICIPATED_IN = "participated_in"
    MENTIONED_IN = "mentioned_in"
    COMMITTED_BY = "committed_by"
    OWNED_BY = "owned_by"
    BELONGS_TO_PROJECT = "belongs_to_project"
    MERGED_FROM = "merged_from"


class RetractionTargetType(str, Enum):
    OBJECT = "object"
    ASSERTION = "assertion"
    RELATION = "relation"
    EVIDENCE = "evidence"


class EvidenceValidity(str, Enum):
    ACTIVE = "active"
    INVALIDATED = "invalidated"
    DELETED = "deleted"


class AssertionValidity(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CONTRADICTED = "contradicted"
    RETRACTED = "retracted"


class EpistemicType(str, Enum):
    OBSERVED = "observed"
    REPORTED = "reported"
    INFERRED = "inferred"
    HYPOTHESIS = "hypothesis"
    USER_CONFIRMED = "user_confirmed"


class Polarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class WorkflowStatus(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class RetentionStatus(str, Enum):
    WORKING = "working"
    LONG_TERM = "long_term"
    ARCHIVED = "archived"
    DELETED = "deleted"


class FulfillmentStatus(str, Enum):
    OPEN = "open"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    UNKNOWN_NO_EVIDENCE = "unknown_no_evidence"
