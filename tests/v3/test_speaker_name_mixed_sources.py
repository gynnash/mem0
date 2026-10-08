import hashlib
from datetime import datetime, timezone

import pytest

from mem0.v3.domain import Evidence
from mem0.v3.speaker_names import (
    SpeakerIdentityScopeError, SpeakerNameScope, plan_speaker_name_correction,
)


NOW = datetime(2026, 10, 8, tzinfo=timezone.utc)


def _evidence(evidence_id, source_id, speaker, memory_id="10"):
    content = f"{speaker} discussed the task."
    return Evidence(evidence_id=evidence_id, user_id="7", workspace_id="7",
                    source_type="transcript", source_id=source_id, memory_id=memory_id,
                    transcript_version=100, speaker_id=speaker, start_ms=0, end_ms=2000,
                    content=content, content_hash=hashlib.sha256(content.encode()).hexdigest(),
                    recorded_at=NOW, created_at=NOW)


def test_mixed_actor_is_held_for_review_while_independent_source_is_corrected():
    changed = _evidence("ev1", "100:0", "Speaker-1")
    other = _evidence("ev2", "100:1", "Speaker-2")
    another_recording = _evidence("ev3", "200:0", "Speaker-3", memory_id="11")
    person = {"object_id": "person-old", "object_type": "entity", "lock_version": 1,
              "evidence_ids": ["ev1"], "payload": {"title": "Speaker-1", "attributes": {
                  "speaker_ref": "Speaker-1"}}}
    other_person = {"object_id": "person-other", "object_type": "entity", "lock_version": 1,
                    "evidence_ids": ["ev2"], "payload": {"title": "Speaker-2", "attributes": {
                        "speaker_ref": "Speaker-2"}}}
    mixed_task = {"object_id": "task-mixed", "object_type": "task", "lock_version": 1,
                  "evidence_ids": ["ev1", "ev2"], "payload": {"title": "Shared task", "attributes": {
                      "owner_entity_id": "person-old", "owner_speaker_ref": "Speaker-1"}}}
    owned_task = {**mixed_task, "object_id": "task-owned", "evidence_ids": ["ev1"],
                  "payload": {"title": "Owned task", "attributes": {
                      "owner_entity_id": "person-old", "owner_speaker_ref": "Speaker-1"}}}
    independent = {"assertion_id": "independent", "evidence_ids": ["ev1"], "payload": {
        "subject_object_ref": "person-old", "predicate": "spoke", "value": {
            "asserted_by_speaker_ref": "Speaker-1"}, "asserted_by_entity_id": "person-old",
        "epistemic_type": "reported", "modality": "stated", "polarity": "positive",
        "confidence": 0.8, "asserted_at": NOW}}
    mixed = {**independent, "assertion_id": "mixed", "evidence_ids": ["ev1", "ev2"]}
    dependent = {**independent, "assertion_id": "task-dependent", "evidence_ids": ["ev2"],
                 "payload": {**independent["payload"], "subject_object_ref": "task-mixed",
                             "value": "The task remains open.", "asserted_by_entity_id": None}}
    replaced_dependent = {**dependent, "assertion_id": "task-replaced-dependent",
                          "evidence_ids": ["ev1"]}
    cross_recording_dependent = {**dependent, "assertion_id": "cross-recording-dependent",
                                 "evidence_ids": ["ev3"]}
    related = {"relation_id": "task-owner", "source_object_id": "task-mixed",
               "target_object_id": "person-old", "relation_type": "owned_by",
               "evidence_ids": ["ev1", "ev2"], "payload": {"confidence": 0.8}}
    unchanged_relation = {**related, "relation_id": "task-other", "relation_type": "discusses",
                          "evidence_ids": ["ev2"]}
    unrelated_relation = {**related, "relation_id": "person-unrelated",
                          "source_object_id": "person-old", "target_object_id": "person-other",
                          "relation_type": "discusses", "evidence_ids": ["ev1"],
                          "payload": {"confidence": 0.8, "epistemic_type": "reported",
                                      "valid_from": NOW}}
    cross_recording_relation = {**related, "relation_id": "cross-recording-relation",
                                "evidence_ids": ["ev3"], "relation_type": "discusses"}
    kwargs = dict(
        operation_key="speaker-name-sync:10:100:1", user_id="7", workspace_id="7",
        base_state_version=1, scopes=(SpeakerNameScope("10", 100, (0,), ("Speaker-1",),
            "Milo", "Speaker-1", "speaker:14"),),
        evidence=(changed, other, another_recording),
        objects=(person, other_person, mixed_task, owned_task),
        assertions=(independent, mixed, dependent, replaced_dependent,
                    cross_recording_dependent),
        relations=(related, unchanged_relation, unrelated_relation,
                   cross_recording_relation), now=NOW, correct_identity=True)
    result = plan_speaker_name_correction(**kwargs)
    assert result.model_dump(mode="json") == plan_speaker_name_correction(**kwargs).model_dump(mode="json")

    held = {(item.target_type.value, item.target_id) for item in result.retractions}
    assert len(held) == len(result.retractions)
    assert {("assertion", "mixed"), ("assertion", "task-dependent"),
            ("assertion", "task-replaced-dependent"),
            ("assertion", "cross-recording-dependent"), ("object", "task-mixed"),
            ("relation", "task-owner"), ("relation", "task-other"),
            ("relation", "cross-recording-relation")} <= held
    assert not any(item.assertion_id == "mixed" for item in result.assertion_mutations)
    assert not any(item.assertion_id == "task-dependent" for item in result.assertion_mutations)
    assert not result.assertion_mutations or all(
        item.payload.subject_object_ref != "task-mixed" for item in result.assertion_mutations)
    assert not any(item.object_id == "task-mixed" for item in result.object_mutations)
    assert any(item.object_id == "person-old" and item.evidence_ids == (
        result.evidence_creates[0].evidence.evidence_id,) for item in result.object_mutations)
    assert any(item.object_id == "task-owned" and
               item.payload.attributes["owner_entity_id"] == "entity:speaker:14"
               for item in result.object_mutations)
    assert not any(item.relation_id == "task-owner" for item in result.relation_mutations)
    assert all(item.source_object_ref != "task-mixed" and
               item.target_object_ref != "task-mixed" for item in result.relation_mutations)
    assert any(item.relation_type == "discusses" for item in result.relation_mutations)
    assert any(item.payload.asserted_by_entity_id == "entity:speaker:14"
               for item in result.assertion_mutations)
    assert {item["field_key"] for item in result.domain_events[0].payload["review_required"]} == {
        "assertion:mixed#reporter", "object:task-mixed#executor"}


def test_stable_speaker_ref_uses_unique_evidence_supported_entity():
    changed = _evidence("ev1", "100:0", "speaker:23")
    source = {"object_id": "person-supported", "object_type": "entity", "lock_version": 1,
              "evidence_ids": ["ev1"], "payload": {"title": "Previous speaker", "attributes": {
                  "speaker_ref": "speaker:23"}}}
    other = {**source, "object_id": "person-other", "evidence_ids": ["ev2"]}
    reported = {"assertion_id": "reported", "evidence_ids": ["ev1"], "payload": {
        "subject_object_ref": "person-supported", "predicate": "spoke", "value": {
            "asserted_by_speaker_ref": "speaker:23"},
        "asserted_by_entity_id": "person-supported", "epistemic_type": "reported",
        "modality": "stated", "polarity": "positive", "confidence": 0.8,
        "asserted_at": NOW}}
    kwargs = dict(operation_key="speaker-name-sync:10:100:1", user_id="7", workspace_id="7",
                  base_state_version=1, scopes=(SpeakerNameScope("10", 100, (0,), ("Speaker-1",),
                      "Milo", "Speaker-1", "speaker:14", ("speaker:23",)),),
                  evidence=(changed,), assertions=(reported,), relations=(), now=NOW,
                  correct_identity=True)

    result = plan_speaker_name_correction(objects=(source, other), **kwargs)
    assert result is not None
    assert any(item.object_id == "person-supported" for item in result.object_mutations)
    assert any(item.payload.asserted_by_entity_id == "entity:speaker:14"
               for item in result.assertion_mutations)

    with pytest.raises(SpeakerIdentityScopeError, match="multiple possible source entities"):
        plan_speaker_name_correction(objects=(source, {**other, "evidence_ids": ["ev1"]}), **kwargs)
