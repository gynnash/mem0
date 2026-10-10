from mem0.v3.retrieval.weekly_signals import authorized_version, business_changes, current_open_state, field_support, omit_lowest_priority_matter


def test_metadata_does_not_make_a_business_transition():
    original = {"object_id": "task:1", "version": 1, "workflow_status": "open", "evidence_ids": ["e1"]}
    changed = {**original, "version": 2, "state_version": 20, "evidence_ids": ["e1", "e2"]}
    assert len(business_changes([original, changed], visible_evidence_ids={"e1", "e2"})) == 1


def test_missing_field_is_unknown_and_explicit_null_can_clear_a_field():
    base = {"object_id": "task:1", "version": 1, "attributes": {"deadline": "Friday"}, "evidence_ids": ["e1"]}
    missing = {"object_id": "task:1", "version": 2, "attributes": {}, "evidence_ids": ["e2"]}
    assert len(business_changes([base, missing], visible_evidence_ids={"e1", "e2"})) == 1
    cleared = {**missing, "attributes": {"deadline": None}, "field_provenance": [{"field_name": "deadline", "evidence_ids": ["e2"]}]}
    change = business_changes([base, cleared], visible_evidence_ids={"e1", "e2"})[-1]["changes"][0]
    assert change["before"] == "Friday" and change["after"] is None and change["kind"] == "transition"


def test_new_field_has_no_invented_previous_state_and_keeps_correction_notes():
    base = {"object_id": "task:1", "version": 1, "attributes": {}, "evidence_ids": ["e1"]}
    new = {**base, "version": 2, "attributes": {"deadline": "Friday"}, "evidence_ids": ["e2"]}
    change = business_changes([base, new], visible_evidence_ids={"e1", "e2"})[-1]["changes"][0]
    assert change["kind"] == "newly_observed" and change["before_refs"] == ()
    new["field_provenance"] = [{"field_name": "deadline", "evidence_ids": ["e2"], "user_confirmed": True, "corrected_at": "2026-10-01"}]
    authorized = authorized_version(new, {"e2"})
    assert authorized["field_provenance"][-1]["corrected_at"] == "2026-10-01"
    assert business_changes([base, authorized], visible_evidence_ids={"e1", "e2"})[-1]["changes"][0]["kind"] == "correction"


def test_explicit_empty_or_unavailable_field_support_never_falls_back():
    row = {"object_id": "task:1", "workflow_status": "completed", "evidence_ids": ["e1"],
        "field_provenance": [{"field_name": "workflow_status", "evidence_ids": []}]}
    assert not field_support(row, "workflow_status")
    authorized = authorized_version(row, {"e1"})
    assert "workflow_status" not in authorized
    assert "workflow_status" in authorized["unavailable_fields"]


def test_budget_selection_omits_whole_matter_and_keeps_coverage():
    threads = [{"kind": "date", "date": "2026-10-01", "source_refs": ["E1"]},
        {"ref": "O1", "open_state": "completion_unknown", "changes": []},
        {"ref": "O2", "open_state": "open", "changes": [{"before_refs": ["E1"], "after_refs": ["E2"]}]},
        {"kind": "assertion", "subject": "O1"}, {"kind": "relation", "from": "O1", "to": "O2"}]
    assert omit_lowest_priority_matter(threads) == [threads[0], threads[2]]


def test_missing_lifecycle_support_is_unknown_and_people_are_not_open_work():
    row = {"object_id": "task:1", "object_type": "task", "workflow_status": "open", "evidence_ids": ["e1"],
        "attributes": {"resolution_status": "resolved"},
        "field_provenance": [{"field_name": "resolution_status", "evidence_ids": ["unavailable"]}]}
    assert current_open_state(authorized_version(row, {"e1"})) == "completion_unknown"
    assert current_open_state({"object_type": "entity", "workflow_status": "active"}) == "not_applicable"
    assert current_open_state({"object_type": "task", "workflow_status": "completed"}) == "terminal"
