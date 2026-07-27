from datetime import datetime, timezone
import importlib.metadata

import pytest
from pydantic import ValidationError

try:
    importlib.metadata.version("mem0pin")
except importlib.metadata.PackageNotFoundError:
    _metadata_version = importlib.metadata.version
    importlib.metadata.version = lambda package: "0.0.0" if package == "mem0pin" else _metadata_version(package)

from mem0.v3 import MemoryChangeDraft, MemoryPlanner
from mem0.v3.contracts import (
    AssertionMutation,
    CURRENT_CONTRACT_VERSIONS,
    EvidenceCreate,
    MemoryCommitReceipt,
    MemoryReadSnapshot,
    ObjectMutation,
    RelationMutation,
    SourceRef,
    ToolDiagnostics,
    ToolEnvelope,
    ToolStatus,
    ValidatedMemoryChangeSet,
)
from mem0.v3.domain import Evidence, LifecycleOperation, MemoryObjectType


def _object_mutation(logical_ref: str = "meeting:42") -> ObjectMutation:
    occurred_at = datetime.now(timezone.utc)
    return ObjectMutation(
        logical_ref=logical_ref,
        operation=LifecycleOperation.CREATE,
        object_type=MemoryObjectType.MEETING,
        evidence_ids=("evidence:42",),
        payload={
            "external_memory_id": "42",
            "canonical_key": "meeting:42",
            "title": "Launch review",
            "valid_from": occurred_at,
            "started_at": occurred_at,
            "transcript_version": 1,
            "processing_status": "extracted",
        },
    )


def _draft(**overrides) -> MemoryChangeDraft:
    values = {
        "changeset_id": "changeset:42",
        "user_id": "user:7",
        "workspace_id": "workspace:7",
        "source_ref": SourceRef(source_type="memory", source_id="42", memory_id="42"),
        "base_state_version": 10,
        "object_mutations": (_object_mutation(),),
    }
    values.update(overrides)
    return MemoryChangeDraft(**values)


def test_planner_emits_current_versioned_changeset_without_storage_dependency():
    changeset = MemoryPlanner().plan(_draft())

    assert isinstance(changeset, ValidatedMemoryChangeSet)
    assert changeset.kernel_version == CURRENT_CONTRACT_VERSIONS.kernel_api_version
    assert changeset.schema_version == CURRENT_CONTRACT_VERSIONS.changeset_schema_version
    assert changeset.workspace_id == "workspace:7"


def test_changeset_rejects_empty_mutation_set():
    with pytest.raises(ValidationError, match="at least one semantic mutation"):
        MemoryPlanner().plan(_draft(object_mutations=()))


def test_changeset_rejects_duplicate_logical_refs_across_mutations():
    with pytest.raises(ValidationError, match="logical_ref values must be unique"):
        MemoryPlanner().plan(_draft(object_mutations=(_object_mutation(), _object_mutation())))


def test_changeset_rejects_physical_storage_instructions():
    with pytest.raises(ValidationError, match="forbidden physical instruction key"):
        ObjectMutation(
            logical_ref="meeting:42",
            operation=LifecycleOperation.CREATE,
            object_type=MemoryObjectType.MEETING,
            evidence_ids=("evidence:42",),
            payload={
                "external_memory_id": "42",
                "canonical_key": "meeting:42",
                "title": "Launch review",
                "valid_from": datetime.now(timezone.utc),
                "started_at": datetime.now(timezone.utc),
                "transcript_version": 1,
                "processing_status": "extracted",
                "attributes": {"sql": "UPDATE memory_v3_objects"},
            },
        )


def test_object_payload_rejects_fields_owned_by_another_object_type():
    with pytest.raises(ValidationError, match="owned by another object type"):
        ObjectMutation(
            logical_ref="decision:wrong-shape",
            operation=LifecycleOperation.CREATE,
            object_type=MemoryObjectType.DECISION,
            evidence_ids=("evidence:42",),
            payload={
                "canonical_key": "decision:wrong-shape",
                "title": "Launch decision",
                "valid_from": datetime.now(timezone.utc),
                "decision": "Ship Friday",
                "effective_status": "effective",
                "effective_from": datetime.now(timezone.utc),
                "transcript_version": 1,
            },
        )


def test_snapshot_requires_timezone_and_projection_not_ahead_of_mysql():
    with pytest.raises(ValidationError, match="timezone-aware"):
        MemoryReadSnapshot(
            user_id="user:7",
            workspace_id="workspace:7",
            as_of_event_id=10,
            projection_checkpoint=9,
            created_at=datetime(2026, 7, 26),
        )

    with pytest.raises(ValidationError, match="cannot exceed"):
        MemoryReadSnapshot(
            user_id="user:7",
            workspace_id="workspace:7",
            as_of_event_id=10,
            projection_checkpoint=11,
            created_at=datetime.now(timezone.utc),
        )


def test_contracts_are_frozen_and_reject_unknown_fields():
    receipt = MemoryCommitReceipt(
        operation_key="operation:42",
        changeset_id="changeset:42",
        committed_state_version=11,
        persisted_ids={"meeting:42": "object:99"},
        audit_event_ids=("audit:11",),
        outbox_event_ids=("outbox:11",),
    )

    with pytest.raises(ValidationError, match="frozen"):
        receipt.committed_state_version = 12

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        MemoryCommitReceipt(
            operation_key="operation:42",
            changeset_id="changeset:42",
            committed_state_version=11,
            arbitrary_sql="UPDATE anything",
        )


def test_tool_envelope_carries_snapshot_watermark_diagnostics():
    envelope = ToolEnvelope[dict[str, str]](
        status=ToolStatus.SUCCESS,
        data={"object_id": "object:99"},
        evidence_refs=("evidence:42",),
        diagnostics=ToolDiagnostics(tool="get_current_state", as_of_event_id=10),
    )

    assert envelope.diagnostics.source_of_truth == "mysql"
    assert envelope.diagnostics.as_of_event_id == 10


def test_evidence_create_must_match_changeset_scope():
    evidence = Evidence(
        evidence_id="evidence:99",
        user_id="user:other",
        workspace_id="workspace:7",
        source_type="transcript",
        source_id="99",
        content="A source-backed statement.",
        content_hash="0" * 64,
        recorded_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
    )

    with pytest.raises(ValidationError, match="user_id must match"):
        MemoryPlanner().plan(_draft(evidence_creates=(EvidenceCreate(logical_ref="evidence:99", evidence=evidence),)))


@pytest.mark.parametrize(
    "mutation_overrides",
    [
        {
            "assertion_mutations": (
                AssertionMutation(
                    logical_ref="assertion:old",
                    assertion_id="assertion:old",
                    operation=LifecycleOperation.UPDATE,
                    evidence_ids=("evidence:42",),
                    payload={
                        "subject_object_ref": "object:1",
                        "predicate": "status",
                        "value": "active",
                        "epistemic_type": "observed",
                        "modality": "stated",
                        "polarity": "positive",
                        "confidence": 1,
                        "asserted_at": datetime.now(timezone.utc),
                    },
                ),
            )
        },
        {
            "relation_mutations": (
                RelationMutation(
                    logical_ref="relation:old",
                    relation_id="relation:old",
                    operation=LifecycleOperation.UPDATE,
                    source_object_ref="object:1",
                    target_object_ref="object:2",
                    relation_type="discusses",
                    evidence_ids=("evidence:42",),
                    payload={
                        "confidence": 1,
                        "epistemic_type": "observed",
                        "valid_from": datetime.now(timezone.utc),
                    },
                ),
            )
        },
    ],
)
def test_assertions_and_relations_are_replaced_instead_of_updated(mutation_overrides):
    with pytest.raises(ValueError, match="immutable"):
        MemoryPlanner().plan(_draft(**mutation_overrides))


def test_relation_type_must_be_registered():
    with pytest.raises(ValidationError, match="relation_type"):
        RelationMutation(
            logical_ref="relation:unknown",
            operation=LifecycleOperation.CREATE,
            source_object_ref="meeting:42",
            target_object_ref="project:apollo",
            relation_type="invented_relation",
            evidence_ids=("evidence:42",),
            payload={
                "confidence": 1,
                "epistemic_type": "observed",
                "valid_from": datetime.now(timezone.utc),
            },
        )
