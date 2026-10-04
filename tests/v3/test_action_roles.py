import json
from datetime import datetime, timezone

import pytest

from mem0.v3.domain import MemoryObjectType
from mem0.v3.extraction import (
    ExtractedClaim, LocalExtractionService, MeetingExtractionInput,
    TranscriptSegment, UnitBackedLocalExtractionResult,
)
from mem0.v3.resolution import AlignmentContext, GlobalAlignmentService, ObjectLinkCandidate


NOW = datetime(2026, 7, 26, 9, tzinfo=timezone.utc)


class FakeModel:
    def __init__(self, claims):
        self.claims = claims

    def generate_structured(self, *, request, response_model):
        self.schema = response_model.model_json_schema()
        self.request = request
        segments = json.loads(request.messages[1].content)["transcript_segments"]
        return response_model.model_validate({
            "extraction_version": "local-extraction/roles-v2",
            "episodic_evidence": [
                {"evidence_id": f"e{index}", "content": segment["evidence_units"][0]["text"],
                 "evidence_unit_ids": [segment["evidence_units"][0]["evidence_unit_id"]], "confidence": 0.97}
                for index, segment in enumerate(segments, 1)
            ],
            "claims": self.claims,
        })


def _source(first_text="Bob, please review the proposal."):
    return MeetingExtractionInput(
        user_id="7", workspace_id="8", memory_id="42", transcript_version=1,
        title="Review planning", started_at=NOW,
        participant_refs=("Alice", "Bob"),
        segments=(
            TranscriptSegment(segment_id="s1", speaker_ref="Alice",
                              text=first_text, start_ms=0, end_ms=1000),
            TranscriptSegment(segment_id="s2", speaker_ref="Bob",
                              text="I can review it tomorrow.", start_ms=1000, end_ms=2000),
        ),
    )


def _claim(**changes):
    return {
        "claim_id": "task-1", "claim_type": "task", "text": "Review the proposal",
        "asserted_by_speaker_ref": "Alice", "initiator_mention": "Alice",
        "owner_mention": "Bob", "action": "Review the proposal",
        "task_intent": "requested", "modality": "stated",
        "episodic_evidence_ids": ["e1"], "confidence": 0.97, **changes,
    }


def _plan(claim, *, candidate=None, first_text="Bob, please review the proposal."):
    model = FakeModel([claim])
    source = _source(first_text)
    extraction = LocalExtractionService(model).extract(source)
    context = AlignmentContext(
        base_state_version=0, now=NOW,
        object_candidates_by_claim={claim["claim_id"]: (candidate,)} if candidate else {},
    )
    return GlobalAlignmentService().plan(source=source, extraction=extraction, context=context)


@pytest.mark.parametrize("intent,expected", [
    ("requested", "proposed"), ("assigned", "proposed"),
    ("self_committed", "accepted"), (None, "proposed"),
])
def test_action_intent_does_not_imply_execution_has_started(intent, expected):
    changeset = _plan(_claim(task_intent=intent, modality="promised" if intent == "self_committed" else "stated"))
    task = next(item for item in changeset.object_mutations if item.object_type is MemoryObjectType.TASK)
    assert task.payload.workflow_status.value == expected
    assert task.payload.attributes["execution_intent"] == intent


def test_statement_initiator_and_executor_are_independent():
    changeset = _plan(_claim(task_intent="assigned"))
    entities = {item.payload.title: item.logical_ref for item in changeset.object_mutations
                if item.object_type is MemoryObjectType.ENTITY}
    task = next(item for item in changeset.object_mutations if item.object_type is MemoryObjectType.TASK)
    assertion = changeset.assertion_mutations[0]
    assert assertion.payload.asserted_by_entity_id == entities["Alice"]
    assert task.payload.attributes["owner_entity_id"] == entities["Bob"]
    assert task.payload.attributes["initiator_entity_id"] == entities["Alice"]
    assert assertion.payload.value["task_intent"] == "assigned"
    assert assertion.payload.value["action"] == "Review the proposal"


def test_reported_commitment_preserves_reporter_separately_from_executor():
    changeset = _plan(_claim(claim_type="commitment", task_intent="self_committed", modality="promised",
                            initiator_mention=None), first_text="Bob told me he committed to reviewing the proposal.")
    entities = {item.payload.title: item.logical_ref for item in changeset.object_mutations
                if item.object_type is MemoryObjectType.ENTITY}
    commitment = next(item for item in changeset.object_mutations if item.object_type is MemoryObjectType.COMMITMENT)
    assert changeset.assertion_mutations[0].payload.asserted_by_entity_id == entities["Alice"]
    assert changeset.assertion_mutations[0].payload.epistemic_type.value == "reported"
    assert commitment.payload.committed_by == entities["Bob"]
    assert commitment.payload.action == "Review the proposal"


def test_missing_roles_remain_unknown_without_discarding_the_request():
    changeset = _plan(_claim(owner_mention=None, initiator_mention=None, asserted_by_speaker_ref=None))
    task = next(item for item in changeset.object_mutations if item.object_type is MemoryObjectType.TASK)
    assert task.payload.attributes["owner_entity_id"] is None
    assert task.payload.attributes["initiator_entity_id"] is None
    assert changeset.assertion_mutations[0].payload.asserted_by_entity_id is None
    assert not any(item.relation_type.value == "owned_by" for item in changeset.relation_mutations)


def test_speaker_outside_the_claims_citations_is_isolated():
    model = FakeModel([_claim(asserted_by_speaker_ref="Bob")])
    extraction = LocalExtractionService(model).extract(_source())
    assert not extraction.claims
    assert len(extraction.episodic_evidence) == 2
    assert any("speaker_outside_source" in value for value in extraction.warnings)


def test_alignment_also_rejects_an_uncited_speaker_when_called_directly():
    source = _source()
    extraction = LocalExtractionService(FakeModel([_claim()])).extract(source)
    extraction = extraction.model_copy(update={
        "claims": (ExtractedClaim.model_validate(_claim(asserted_by_speaker_ref="Bob")),),
    })
    with pytest.raises(ValueError, match="speaker is outside cited source"):
        GlobalAlignmentService().plan(source=source, extraction=extraction,
                                      context=AlignmentContext(base_state_version=0, now=NOW))


@pytest.mark.parametrize("intent", ["requested", "assigned", "self_committed"])
def test_completion_is_an_independently_evidenced_lifecycle_signal(intent):
    changeset = _plan(_claim(task_intent=intent, lifecycle_signal="resolved"),
                     first_text="Bob has completed the proposal review.")
    task = next(item for item in changeset.object_mutations if item.object_type is MemoryObjectType.TASK)
    assert task.payload.workflow_status.value == "completed"
    assert task.payload.attributes["completion_evidence_ids"] == task.evidence_ids
    assert changeset.assertion_mutations[0].payload.value["task_intent"] == intent


@pytest.mark.parametrize("status", ["accepted", "in_progress", "completed", "cancelled"])
def test_repeated_request_does_not_reset_an_existing_action_state(status):
    candidate = ObjectLinkCandidate(
        object_id="task:existing", object_type="task", canonical_key="task:existing",
        title="Review the proposal", lock_version=3, confidence=0.99,
        anchor_types=("explicit_reference",), workflow_status=status,
    )
    changeset = _plan(_claim(), candidate=candidate)
    task = next(item for item in changeset.object_mutations if item.object_type is MemoryObjectType.TASK)
    assert task.object_id == "task:existing"
    assert task.expected_version == 3
    assert "workflow_status" not in task.payload.model_fields_set
    assert "completion_evidence_ids" not in task.payload.attributes


def test_user_locked_status_is_preserved():
    candidate = ObjectLinkCandidate(
        object_id="task:existing", object_type="task", canonical_key="task:existing",
        title="Review the proposal", lock_version=3, confidence=0.99,
        anchor_types=("explicit_reference",), workflow_status="proposed",
        field_provenance=({"field_name": "workflow_status", "user_locked": True},),
    )
    changeset = _plan(_claim(lifecycle_signal="resolved"), candidate=candidate)
    task = next(item for item in changeset.object_mutations if item.object_type is MemoryObjectType.TASK)
    assert "workflow_status" not in task.payload.model_fields_set
    assert any("user_locked_field_preserved:workflow_status" in value for value in changeset.warnings)


def test_user_locked_executor_is_preserved_without_an_incoming_ownership_relation():
    candidate = ObjectLinkCandidate(
        object_id="task:existing", object_type="task", canonical_key="task:existing",
        title="Review the proposal", lock_version=3, confidence=0.99,
        anchor_types=("explicit_reference",), workflow_status="proposed",
        attributes={"owner_entity_id": "entity:confirmed"},
        field_provenance=({"field_name": "owner_entity_id", "user_locked": True},),
    )
    changeset = _plan(_claim(), candidate=candidate)
    task = next(item for item in changeset.object_mutations if item.object_type is MemoryObjectType.TASK)
    assert "owner_entity_id" not in task.payload.attributes
    assert "owner_mention" not in task.payload.attributes
    assert not any(item.relation_type.value == "owned_by" for item in changeset.relation_mutations)


def test_unresolved_roles_do_not_clear_existing_canonical_responsibility():
    candidate = ObjectLinkCandidate(
        object_id="task:existing", object_type="task", canonical_key="task:existing",
        title="Review the proposal", lock_version=3, confidence=0.99,
        anchor_types=("explicit_reference",), workflow_status="accepted",
        attributes={"owner_entity_id": "entity:confirmed", "execution_intent": "self_committed"},
    )
    changeset = _plan(_claim(owner_mention=None, task_intent=None), candidate=candidate)
    task = next(item for item in changeset.object_mutations if item.object_type is MemoryObjectType.TASK)
    assert "owner_entity_id" not in task.payload.attributes
    assert "execution_intent" not in task.payload.attributes


@pytest.mark.parametrize("intent", ["requested", "assigned"])
def test_requests_and_assignments_cannot_be_typed_as_commitments(intent):
    with pytest.raises(ValueError, match="not a commitment"):
        ExtractedClaim.model_validate(_claim(claim_type="commitment", task_intent=intent))


@pytest.mark.parametrize("claim_type", ["task", "commitment"])
def test_negated_completion_is_invalid(claim_type):
    with pytest.raises(ValueError, match="cannot establish completion"):
        ExtractedClaim.model_validate(_claim(claim_type=claim_type, task_intent="self_committed",
                                            negated=True, lifecycle_signal="resolved"))


@pytest.mark.parametrize("modality,negated", [("uncertain", False), ("promised", True)])
def test_uncertain_or_negated_self_commitment_does_not_establish_acceptance(modality, negated):
    changeset = _plan(_claim(task_intent="self_committed", modality=modality, negated=negated))
    task = next(item for item in changeset.object_mutations if item.object_type is MemoryObjectType.TASK)
    assert task.payload.workflow_status.value == "proposed"
    if negated:
        assert not any(item.relation_type.value == "owned_by" for item in changeset.relation_mutations)


def test_extra_fields_are_forbidden_in_both_claim_contracts():
    with pytest.raises(ValueError):
        ExtractedClaim.model_validate(_claim(guessed_owner="Alice"))
    with pytest.raises(ValueError):
        UnitBackedLocalExtractionResult.model_validate({"extraction_version": "v2", "unexpected": True})
    schema = UnitBackedLocalExtractionResult.model_json_schema()
    assert schema["$defs"]["UnitBackedExtractedClaim"]["additionalProperties"] is False
    assert "asserted_by_speaker_ref" in schema["$defs"]["UnitBackedExtractedClaim"]["properties"]
