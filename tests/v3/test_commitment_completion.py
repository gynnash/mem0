import hashlib
from datetime import datetime, timezone

from mem0.v3 import ControlledMemoryMaintenance
from mem0.v3.domain import (
    Evidence,
    FulfillmentStatus,
    LifecycleOperation,
    WorkflowStatus,
)


NOW = datetime(2026, 7, 26, 12, tzinfo=timezone.utc)


def test_controlled_completion_emits_semantic_changeset_only():
    content = "User completed Todo 42"
    evidence = Evidence(
        evidence_id="evidence:todo:42:complete",
        user_id="7",
        workspace_id="8",
        source_type="product_action",
        source_id="todo:42:complete",
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        recorded_at=NOW,
        created_at=NOW,
    )

    changeset = ControlledMemoryMaintenance().resolve_commitment(
        changeset_id="changeset:todo:42:complete",
        user_id="7",
        workspace_id="8",
        base_state_version=19,
        commitment_object_id="object:commitment:1",
        commitment_expected_version=3,
        completion_evidence=evidence,
    )

    assert changeset.source_ref.source_type == "product_action"
    assert changeset.evidence_creates[0].evidence == evidence
    mutation = changeset.object_mutations[0]
    assert mutation.operation is LifecycleOperation.RESOLVE
    assert mutation.expected_version == 3
    assert mutation.payload.workflow_status is WorkflowStatus.COMPLETED
    assert mutation.payload.fulfillment_status is FulfillmentStatus.COMPLETED
    assert mutation.payload.completion_evidence_ids == (
        "commitment:completion_evidence",
    )
    assert changeset.domain_events[0].event_type == "memory.commitment_resolved"
    assert (
        changeset.domain_events[1].event_type
        == "memory.dependencies_invalidated"
    )
    assert (
        changeset.domain_events[1].aggregate_ref
        == "object:commitment:1"
    )
