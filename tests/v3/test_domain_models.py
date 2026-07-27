from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from mem0.v3 import ControlledMemoryMaintenance, MemoryChangeDraft, MemoryPlanner
from mem0.v3.contracts import ObjectMutation, SourceRef
from mem0.v3.domain import (
    CommitmentPayload,
    Evidence,
    FulfillmentStatus,
    LifecycleOperation,
    MemoryObjectType,
)


NOW = datetime(2026, 7, 26, tzinfo=timezone.utc)


def _draft(mutation: ObjectMutation) -> MemoryChangeDraft:
    return MemoryChangeDraft(
        changeset_id="changeset:1",
        user_id="user:1",
        workspace_id="workspace:1",
        source_ref=SourceRef(source_type="memory", source_id="1", memory_id="1"),
        base_state_version=4,
        object_mutations=(mutation,),
    )


def test_evidence_requires_complete_ordered_span_and_timezone():
    with pytest.raises(ValidationError, match="supplied together"):
        Evidence(
            evidence_id="evidence:1",
            user_id="user:1",
            workspace_id="workspace:1",
            source_type="transcript",
            source_id="1",
            start_ms=10,
            content="Decision was confirmed.",
            content_hash="0" * 64,
            recorded_at=NOW,
            created_at=NOW,
        )

    with pytest.raises(ValidationError, match="timezone-aware"):
        Evidence(
            evidence_id="evidence:1",
            user_id="user:1",
            workspace_id="workspace:1",
            source_type="transcript",
            source_id="1",
            content="Decision was confirmed.",
            content_hash="0" * 64,
            recorded_at=datetime(2026, 7, 26),
            created_at=NOW,
        )


def test_completed_commitment_requires_completion_evidence():
    with pytest.raises(ValidationError, match="completion evidence"):
        CommitmentPayload(
            committed_by="person:1",
            action="Send the proposal",
            committed_at=NOW,
            due_at=NOW + timedelta(days=1),
            fulfillment_status=FulfillmentStatus.COMPLETED,
        )


def test_default_planner_rules_require_evidence():
    mutation = ObjectMutation(
        logical_ref="decision:1",
        operation=LifecycleOperation.CREATE,
        object_type=MemoryObjectType.DECISION,
        payload={
            "canonical_key": "decision:1",
            "title": "Shipping month",
            "valid_from": datetime.now(timezone.utc),
            "decision": "Ship in September",
            "effective_status": "effective",
            "effective_from": datetime.now(timezone.utc),
        },
    )

    with pytest.raises(ValueError, match="requires evidence"):
        MemoryPlanner().plan(_draft(mutation))


def test_default_planner_rules_enforce_deterministic_meeting_identity():
    mutation = ObjectMutation(
        logical_ref="meeting:1",
        operation=LifecycleOperation.CREATE,
        object_type=MemoryObjectType.MEETING,
        evidence_ids=("evidence:1",),
        payload={
            "external_memory_id": "1",
            "canonical_key": "meeting:wrong",
            "title": "Meeting",
            "valid_from": NOW,
            "started_at": NOW,
            "transcript_version": 1,
            "processing_status": "extracted",
        },
    )

    with pytest.raises(ValueError, match="canonical_key"):
        MemoryPlanner().plan(_draft(mutation))


def test_default_planner_rules_require_optimistic_version_for_updates():
    mutation = ObjectMutation(
        logical_ref="decision:1:update",
        operation=LifecycleOperation.UPDATE,
        object_type=MemoryObjectType.DECISION,
        object_id="object:1",
        evidence_ids=("evidence:2",),
        payload={"decision": "Ship in October"},
    )

    with pytest.raises(ValueError, match="expected object version"):
        MemoryPlanner().plan(_draft(mutation))


def test_merge_is_rejected_for_ingestion_but_allowed_for_controlled_maintenance():
    mutation = ObjectMutation(
        logical_ref="topic:merge",
        operation=LifecycleOperation.MERGE,
        object_type=MemoryObjectType.TOPIC,
        object_id="topic:target",
        expected_version=1,
        evidence_ids=("evidence:1",),
        payload={"attributes": {"merged_source_object_ids": ["topic:source"]}},
    )
    with pytest.raises(ValueError, match="controlled correction"):
        MemoryPlanner().plan(_draft(mutation))

    changeset = ControlledMemoryMaintenance().merge_objects(
        changeset_id="changeset:merge",
        user_id="user:1",
        workspace_id="workspace:1",
        base_state_version=4,
        object_type=MemoryObjectType.TOPIC,
        target_object_id="topic:target",
        target_expected_version=1,
        source_object_ids=("topic:source",),
        evidence_ids=("evidence:1",),
        valid_from=NOW,
        reason="confirmed duplicate",
    )
    assert changeset.object_mutations[0].operation is LifecycleOperation.MERGE
    assert changeset.retractions[0].target_id == "topic:source"


def test_controlled_split_requires_two_new_children():
    def child(ref):
        return ObjectMutation(
            logical_ref=ref,
            operation=LifecycleOperation.CREATE,
            object_type=MemoryObjectType.TOPIC,
            evidence_ids=("evidence:1",),
                payload={
                    "canonical_key": ref,
                    "title": ref,
                    "valid_from": NOW,
                    "canonical_label": ref,
                    "first_seen_at": NOW,
                    "last_seen_at": NOW,
                    "resolution_status": "new_topic",
                },
            )
    changeset = ControlledMemoryMaintenance().split_object(
        changeset_id="changeset:split",
        user_id="user:1",
        workspace_id="workspace:1",
        base_state_version=4,
        object_type=MemoryObjectType.TOPIC,
        source_object_id="topic:source",
        source_expected_version=1,
        child_mutations=(child("topic:a"), child("topic:b")),
        evidence_ids=("evidence:1",),
        reason="two independent topics",
    )
    assert changeset.object_mutations[0].operation is LifecycleOperation.SPLIT
    assert len(changeset.object_mutations) == 3
