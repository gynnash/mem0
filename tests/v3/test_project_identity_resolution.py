"""Reserved project identities are distinct from withdrawn factual state."""

from datetime import datetime, timezone

import pytest

from mem0.v3.domain import LifecycleOperation
from mem0.v3.extraction import (
    EpisodicEvidence, EvidenceSpan, ExtractedProjectMention,
    LocalExtractionResult, MeetingExtractionInput, TranscriptSegment,
)
from mem0.v3.resolution import AlignmentContext, GlobalAlignmentService


NOW = datetime(2026, 10, 4, tzinfo=timezone.utc)
KEY = GlobalAlignmentService.project_canonical_key("Apollo")


def _plan(*, validity="retracted", confidence=0.99, identity_updates=None):
    source = MeetingExtractionInput(user_id="1", workspace_id="1", memory_id="43",
        transcript_version=2, title="Apollo review", started_at=NOW, participant_refs=(),
        segments=(TranscriptSegment(segment_id="s1", text="Apollo is discussed today.", start_ms=0, end_ms=1000),))
    extraction = LocalExtractionResult(extraction_version="fixture/v1",
        episodic_evidence=(EpisodicEvidence(evidence_id="episode-new", content=source.segments[0].text,
            source_spans=(EvidenceSpan(segment_id="s1", start_char=0, end_char=len(source.segments[0].text)),), confidence=0.99),),
        project_mentions=(ExtractedProjectMention(mention="Apollo", episodic_evidence_ids=("episode-new",), confidence=confidence),))
    identity = dict(user_id="1", workspace_id="1", object_id="project-fixed", canonical_key=KEY,
        lock_version=3, validity=validity, assertion_ids=("old-assertion",), relation_ids=("old-relation",))
    identity.update(identity_updates or {})
    context = AlignmentContext(base_state_version=10, now=NOW, project_identities_by_mention={"Apollo": identity})
    return GlobalAlignmentService().plan(source=source, extraction=extraction, context=context)


def test_retracted_project_uses_same_identity_with_new_evidence_and_explicit_retirements():
    draft = _plan()
    project = next(item for item in draft.object_mutations if item.object_type.value == "project")
    assert project.operation is LifecycleOperation.REOPEN
    assert project.object_id == "project-fixed"
    assert project.expected_version == 3
    assert project.payload.canonical_key == KEY
    assert project.payload.valid_from == NOW
    assert set(project.payload.attributes) == {"identity_aliases", "resolution_status", "resolver_version"}
    new_evidence = {item.logical_ref for item in draft.evidence_creates}
    assert set(project.evidence_ids) <= new_evidence
    assert {(item.target_type.value, item.target_id) for item in draft.retractions} == {
        ("assertion", "old-assertion"), ("relation", "old-relation"),
    }
    assert all(set(item.evidence_ids) <= new_evidence for item in draft.retractions)
    assert any(event.event_type == "memory.dependencies_invalidated" for event in draft.domain_events)


def test_exact_active_identity_missed_by_recall_is_confirmed_without_creation():
    draft = _plan(validity="active")
    project = next(item for item in draft.object_mutations if item.object_type.value == "project")
    assert project.operation is LifecycleOperation.CONFIRM
    assert project.object_id == "project-fixed"
    assert project.payload.valid_from is None
    assert not draft.retractions


@pytest.mark.parametrize("validity", ["superseded", "merged"])
def test_non_reopenable_identity_is_not_revived(validity):
    draft = _plan(validity=validity)
    assert not any(item.object_type.value == "project" for item in draft.object_mutations)


def test_weak_project_mention_does_not_reestablish_state():
    draft = _plan(confidence=0.94)
    assert not any(item.object_type.value == "project" for item in draft.object_mutations)
    assert not draft.retractions


@pytest.mark.parametrize("retention", ["deleted", "forgotten"])
def test_user_removed_identity_is_not_reestablished(retention):
    draft = _plan(identity_updates={"retention_status": retention})
    assert not any(item.object_type.value == "project" for item in draft.object_mutations)
    assert not draft.retractions


@pytest.mark.parametrize("updates", [{"user_id": "2"}, {"workspace_id": "2"}, {"canonical_key": "project:other"}])
def test_identity_scope_and_key_must_match_new_source(updates):
    with pytest.raises(ValueError, match="identity does not match"):
        _plan(identity_updates=updates)
