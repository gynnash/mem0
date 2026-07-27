"""Stable cross-repository contracts for the V3 memory kernel."""

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
from mem0.v3.contracts.commit import MemoryCommitReceipt
from mem0.v3.contracts.payloads import (
    AssertionCreatePayload,
    ObjectMutationPayload,
    RelationCreatePayload,
)
from mem0.v3.contracts.snapshot import MemoryReadSnapshot
from mem0.v3.contracts.tools import ToolDiagnostics, ToolEnvelope, ToolStatus
from mem0.v3.contracts.versions import (
    CHANGESET_SCHEMA_VERSION,
    CURRENT_CONTRACT_VERSIONS,
    KERNEL_API_VERSION,
    MINIMUM_STORAGE_SCHEMA_VERSION,
    TOOL_CONTRACT_VERSION,
    ContractVersions,
)

__all__ = [
    "AssertionMutation",
    "AssertionCreatePayload",
    "CHANGESET_SCHEMA_VERSION",
    "CURRENT_CONTRACT_VERSIONS",
    "ContractVersions",
    "DomainEvent",
    "EvidenceCreate",
    "KERNEL_API_VERSION",
    "MINIMUM_STORAGE_SCHEMA_VERSION",
    "MemoryCommitReceipt",
    "MemoryReadSnapshot",
    "ObjectMutation",
    "ObjectMutationPayload",
    "RelationMutation",
    "RelationCreatePayload",
    "RetractionMutation",
    "SourceRef",
    "TOOL_CONTRACT_VERSION",
    "ToolDiagnostics",
    "ToolEnvelope",
    "ToolStatus",
    "ValidatedMemoryChangeSet",
]
