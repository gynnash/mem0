from datetime import datetime, timezone

from mem0.v3 import ControlledMemoryMaintenance
from mem0.v3.contracts import AssertionCreatePayload, RelationCreatePayload
from mem0.v3.domain import (
    EpistemicType,
    FieldProvenance,
    LifecycleOperation,
    MemoryObjectType,
    Polarity,
    RelationType,
    RetractionTargetType,
)


NOW = datetime(2026, 7, 26, tzinfo=timezone.utc)


def test_recompute_object_support_preserves_user_locked_provenance():
    changeset = ControlledMemoryMaintenance().recompute_object_support(
        changeset_id="changeset:dependency:object:1",
        user_id="7",
        workspace_id="7",
        base_state_version=9,
        object_type=MemoryObjectType.COMMITMENT,
        object_id="obj_1",
        expected_version=3,
        evidence_ids=("ev_active",),
        field_provenance=(
            FieldProvenance(
                field_name="action",
                evidence_ids=("ev_active",),
                user_confirmed=True,
                user_locked=True,
            ),
        ),
    )

    mutation = changeset.object_mutations[0]
    assert mutation.operation is LifecycleOperation.UPDATE
    assert mutation.evidence_ids == ("ev_active",)
    assert mutation.payload.field_provenance[0].user_locked is True
    assert changeset.expected_object_versions == {"obj_1": 3}


def test_replace_assertion_support_retracts_immutable_predecessor():
    changeset = ControlledMemoryMaintenance().replace_assertion_support(
        changeset_id="changeset:dependency:assertion:1",
        user_id="7",
        workspace_id="7",
        base_state_version=9,
        assertion_id="ast_1",
        evidence_ids=("ev_active",),
        payload=AssertionCreatePayload(
            subject_object_ref="obj_1",
            predicate="action",
            value="Send the proposal",
            epistemic_type=EpistemicType.OBSERVED,
            modality="asserted",
            polarity=Polarity.POSITIVE,
            confidence=0.9,
            asserted_at=NOW,
        ),
    )

    assert changeset.assertion_mutations[0].operation is LifecycleOperation.CREATE
    assert changeset.retractions[0].target_type is RetractionTargetType.ASSERTION
    assert changeset.retractions[0].target_id == "ast_1"


def test_replace_relation_support_retracts_immutable_predecessor():
    changeset = ControlledMemoryMaintenance().replace_relation_support(
        changeset_id="changeset:dependency:relation:1",
        user_id="7",
        workspace_id="7",
        base_state_version=9,
        relation_id="rel_1",
        source_object_id="obj_1",
        target_object_id="obj_2",
        relation_type=RelationType.BELONGS_TO_PROJECT,
        evidence_ids=("ev_active",),
        payload=RelationCreatePayload(
            confidence=0.8,
            epistemic_type=EpistemicType.OBSERVED,
            valid_from=NOW,
        ),
    )

    assert changeset.relation_mutations[0].operation is LifecycleOperation.CREATE
    assert changeset.retractions[0].target_type is RetractionTargetType.RELATION
    assert changeset.retractions[0].target_id == "rel_1"


def test_retract_source_evidence_emits_one_invalidation_per_evidence():
    changeset = ControlledMemoryMaintenance().retract_source_evidence(
        changeset_id="changeset:source:42:deleted",
        user_id="7",
        workspace_id="7",
        base_state_version=9,
        evidence_ids=("ev_1", "ev_2", "ev_1"),
        source_id="42",
        reason="source_memory_deleted",
    )

    assert [item.target_id for item in changeset.retractions] == [
        "ev_1",
        "ev_2",
    ]
    assert all(
        item.target_type is RetractionTargetType.EVIDENCE
        for item in changeset.retractions
    )
    assert [item.aggregate_ref for item in changeset.domain_events] == [
        "ev_1",
        "ev_2",
    ]
