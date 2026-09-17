from copy import deepcopy

from mem0.v3.retrieval.monthly_selection import plan_monthly_candidates


def obj(identifier, kind="commitment", **values):
    return {"object_id": identifier, "object_type": kind, "title": identifier,
            "validity": "active", "workflow_status": "accepted", "attributes": {},
            "evidence_ids": [f"e:{identifier}"], "version": 2, **values}


def plan(**values):
    return plan_monthly_candidates(**{
        "objects": [], "previous_versions": [], "assertions": [], "relations": [],
        "source_occurrences": [], "source_candidates": [], "user_context": {},
        "window": {"utc_start": "2026-08-01T00:00:00Z", "utc_end": "2026-09-01T00:00:00Z"},
        "snapshot_event_id": 50, **values})


def occurrence(ref, root, week="2026-08-03", **values):
    return {"candidate_ref": ref, "root_source_ref": root, "week_key": week,
            "evidence_ids": [f"e:{root}"], **values}


def item(result, ref):
    return next(value for value in result["items"] if value["candidate_ref"] == ref)


def test_important_single_decision_precedes_repeated_trivia_and_keeps_before_after():
    decision = obj("d", "decision", attributes={"decision": "Friday", "decision_changed": True})
    previous = obj("d", "decision", attributes={"decision": "Monday"}, version=1, evidence_ids=["e:before"])
    priority = {"assertion_id": "a:priority", "subject_object_id": "d", "predicate": "priority",
                "value": "high", "epistemic_type": "user_confirmed", "validity": "active",
                "evidence_ids": ["e:priority"]}
    result = plan(objects=[obj("trivia", "issue"), decision], previous_versions=[previous],
                  assertions=[priority], source_occurrences=[
                      occurrence("object:trivia", f"memory:{number}") for number in range(1, 21)])
    selected = result["items"][0]
    assert selected["candidate_ref"] == "object:d"
    assert selected["tier"] == "material_change"
    assert selected["required"] is True
    assert {"e:d", "e:before", "e:priority"} <= set(selected["required_evidence_ids"])
    assert "decision_changed" in selected["reason_codes"]
    assert result["selection_version"] == 1 and result["scan_complete"] is True


def test_title_change_or_unsupported_importance_does_not_promote_decision():
    current = obj("d", "decision", title="renamed", importance=1,
                  attributes={"decision_changed": True})
    before = obj("d", "decision", title="old", version=1,
                 attributes={"decision_changed": True})
    result = item(plan(objects=[current], previous_versions=[before]), "object:d")
    assert result["tier"] == "necessary_context"
    assert result["required"] is False


def test_derivatives_do_not_inflate_recurrence_and_recordings_are_not_events():
    occurrences = [occurrence("object:x", "memory:1") for _ in range(8)]
    result = item(plan(objects=[obj("x", "issue", attributes={"occurrence_count": 8})],
                       source_occurrences=occurrences), "object:x")
    assert result["independent_source_count"] == 1
    assert result["tier"] == "necessary_context"
    occurrences.append(occurrence("object:x", "memory:2", "2026-08-10"))
    result = item(plan(objects=[obj("x")], source_occurrences=occurrences), "object:x")
    assert result["tier"] == "independent_recurrence"
    assert result["independent_recording_count"] == 2
    assert result["verified_occurrence_count"] == 0
    assert result["reason_codes"] == ["repeated_mentions"]


def test_repeated_event_requires_distinct_explicit_supported_event_identity():
    occurrences = [occurrence("object:x", "memory:1", event_id="event:1"),
                   occurrence("object:x", "memory:2", event_id="event:1")]
    same = item(plan(objects=[obj("x")], source_occurrences=occurrences), "object:x")
    assert same["verified_occurrence_count"] == 1
    assert "repeated_occurrences" not in same["reason_codes"]
    occurrences.append(occurrence("object:x", "memory:3", event_id="event:2"))
    different = item(plan(objects=[obj("x")], source_occurrences=occurrences), "object:x")
    assert different["verified_occurrence_count"] == 2
    assert "repeated_occurrences" in different["reason_codes"]


def test_shared_project_or_same_name_does_not_invent_dependency_or_merge_identity():
    a = obj("a", title="Alex", primary_project_id="p", attributes={"dependency_project_ids": ["q"]})
    b = obj("b", title="Alex", primary_project_id="q")
    result = plan(objects=[a, b])
    assert {row["candidate_ref"] for row in result["items"]} == {"object:a", "object:b"}
    assert all(row["tier"] == "necessary_context" for row in result["items"])
    assert all(row["relation_ids"] == [] for row in result["items"])


def test_supported_relation_keeps_endpoints_direction_evidence_and_hypothesis():
    relation = {"relation_id": "r:1", "source_object_id": "a", "target_object_id": "b",
                "relation_type": "depends_on", "validity": "active", "epistemic_type": "hypothesis",
                "confidence": .6, "evidence_ids": ["e:relation"]}
    result = item(plan(objects=[obj("a"), obj("b")], relations=[relation]), "object:a")
    assert result["tier"] == "meaningful_connection"
    assert result["object_ids"] == ["a", "b"]
    assert result["relation_ids"] == ["r:1"]
    assert {"e:a", "e:b", "e:relation"} <= set(result["required_evidence_ids"])
    assert result["epistemic_type"] == "hypothesis"
    assert result["required"] is False


def test_correction_owner_and_opposing_assertion_are_indivisible_dependencies():
    current = obj("x", workflow_status="cancelled", attributes={"owner_entity_id": "owner"},
                  evidence_ids=["e:correction"], field_provenance=[
                      {"field_name": "workflow_status", "evidence_ids": ["e:correction"]}])
    before = obj("x", version=1, evidence_ids=["e:old"])
    negative = {"assertion_id": "a:no", "subject_object_id": "x", "predicate": "commitment",
                "value": "withdrawn", "polarity": "negative", "validity": "active",
                "epistemic_type": "reported", "evidence_ids": ["e:negative"]}
    result = item(plan(objects=[current, obj("owner", "entity")], previous_versions=[before],
                       assertions=[negative]), "object:x")
    assert {"x", "owner"} == set(result["object_ids"])
    assert {"e:correction", "e:old", "e:owner", "e:negative"} <= set(result["required_evidence_ids"])
    assert "e:correction" in result["mandatory_state_refs"]
    assert "a:no" in result["assertion_ids"]


def test_missing_owner_or_relation_support_rejects_whole_candidate():
    result = plan(objects=[obj("x", attributes={"owner_entity_id": "missing"})])
    assert result["items"] == []
    assert result["unavailable_candidates"][0]["candidate_ref"] == "object:x"
    relation = {"relation_id": "r", "source_object_id": "a", "target_object_id": "b",
                "relation_type": "depends_on", "validity": "active", "evidence_ids": []}
    result = plan(objects=[obj("a"), obj("b")], relations=[relation])
    assert result["items"] == []


def test_owner_dependency_cycle_does_not_create_trustworthy_current_state():
    result = plan(objects=[obj("a", attributes={"owner_entity_id": "b"}),
                           obj("b", attributes={"owner_entity_id": "a"})])
    assert result["items"] == []
    assert all(row["reason"] == "cyclic_dependency" for row in result["unavailable_candidates"])


def test_unmodeled_sources_remain_typed_coverage_candidates_and_not_new_facts():
    sources = [{"candidate_ref": f"source:{kind}:7", "source_type": kind, "source_id": "7",
                "evidence_ids": [f"e:{kind}"], "root_source_ref": f"{kind}:7", "week_key": "2026-08-03"}
               for kind in ("memory", "memo")]
    result = plan(source_candidates=sources)
    assert result["unmodeled_source_refs"] == ["source:memo:7", "source:memory:7"]
    assert len(result["items"]) == 2
    assert all(row["tier"] == "necessary_context" for row in result["items"])


def test_selection_is_input_order_invariant_and_does_not_mutate_inputs():
    args = {"objects": [obj("a"), obj("b")], "source_occurrences": [
        occurrence("object:a", "memory:1"), occurrence("object:a", "memory:2")]}
    original = deepcopy(args)
    first = plan(**args)
    second = plan(**{name: list(reversed(values)) for name, values in args.items()})
    assert first == second
    assert args == original


def test_source_coverage_includes_old_reference_current_correction_and_speaker_identity():
    current = obj("x", workflow_status="cancelled", evidence_ids=["e:withdrawn"])
    before = obj("x", version=1, evidence_ids=["e:original"])
    source = {"source_type": "memory", "source_id": "38", "evidence_ids": ["e:original"],
              "object_ids": ["speaker"], "root_source_ref": "memory:38"}
    result = item(plan(objects=[current, obj("speaker", "entity")], previous_versions=[before],
                       source_candidates=[source]), "source:memory:38")
    assert result["object_ids"] == ["speaker", "x"]
    assert {"e:original", "e:withdrawn", "e:speaker"} <= set(result["required_evidence_ids"])
    assert "e:withdrawn" in result["mandatory_state_refs"]


def test_source_with_unavailable_identity_is_not_a_safe_coverage_package():
    result = plan(source_candidates=[{"source_type": "memory", "source_id": "38",
                                      "evidence_ids": ["e:recording"], "object_ids": ["missing"]}])
    assert result["items"] == []
    assert result["unavailable_candidates"][0]["reason"] == "missing_or_unavailable_object"


def test_derivative_dates_or_recording_aliases_do_not_inflate_independent_weeks():
    result = item(plan(objects=[obj("x")], source_occurrences=[
        occurrence("object:x", "memory:1", "2026-08-03", memory_id="1"),
        occurrence("object:x", "memory:1", "2026-08-10", memory_id="derivative:1")]), "object:x")
    assert result["supporting_week_keys"] == ["2026-08-03"]
    assert result["independent_recording_count"] == 1


def test_missing_closure_of_user_priority_reports_required_unavailable():
    priority = {"assertion_id": "priority", "subject_object_id": "x", "predicate": "priority",
                "value": "high", "epistemic_type": "user_confirmed", "evidence_ids": ["e:priority"]}
    result = plan(objects=[obj("x", attributes={"owner_entity_id": "missing"})], assertions=[priority])
    assert result["unavailable_candidates"] == [{"candidate_ref": "object:x",
                                                "reason": "missing_or_unavailable_object", "required": True}]


def test_supported_blocker_of_active_goal_is_material_without_inventing_priority():
    relation = {"relation_id": "r", "source_object_id": "issue", "target_object_id": "goal",
                "relation_type": "blocks", "validity": "active", "epistemic_type": "reported",
                "evidence_ids": ["e:blocking"]}
    result = item(plan(objects=[obj("issue", "issue"), obj("goal", "goal")],
                       relations=[relation]), "object:issue")
    assert result["tier"] == "material_change"
    assert "affects_active_goal" in result["reason_codes"]
    assert result["required"] is False


def test_structured_source_priority_keeps_original_support_and_uncertainty():
    source = {"source_type": "summary", "source_id": "7", "evidence_ids": ["e:original"],
              "structured_signals": [{"kind": "decision", "priority": "high", "user_confirmed": True,
                                      "epistemic_type": "reported", "evidence_ids": ["e:original"]}]}
    result = item(plan(source_candidates=[source]), "source:summary:7")
    assert result["tier"] == "material_change" and result["required"] is True
    assert result["required_evidence_ids"] == ["e:original"]
    source["structured_signals"][0]["epistemic_type"] = "hypothesis"
    result = item(plan(source_candidates=[source]), "source:summary:7")
    assert result["required"] is False and result["epistemic_type"] == "hypothesis"


def test_unsupported_structured_hint_never_becomes_required_but_source_is_preserved():
    source = {"source_type": "summary", "source_id": "7", "evidence_ids": ["e:original"],
              "structured_signals": [{"kind": "decision", "priority": "high", "user_confirmed": True}]}
    result = item(plan(source_candidates=[source]), "source:summary:7")
    assert result["tier"] == "necessary_context" and result["required"] is False


def test_negative_hypothesis_assertion_and_all_provenance_support_are_preserved():
    current = obj("x", field_provenance=[{"field_name": "condition", "evidence_ids": ["e:condition"]}])
    assertion = {"assertion_id": "negative", "subject_object_id": "x", "predicate": "outcome",
                 "polarity": "negative", "epistemic_type": "hypothesis", "evidence_ids": ["e:opposition"]}
    result = item(plan(objects=[current], assertions=[assertion]), "object:x")
    assert {"e:x", "e:condition", "e:opposition"} <= set(result["required_evidence_ids"])
    assert result["epistemic_type"] == "hypothesis"


def test_owner_self_description_is_not_a_dependency_cycle():
    current = obj("x", attributes={"owner_entity_id": "owner"})
    assertion = {"assertion_id": "identity", "subject_object_id": "owner",
                 "asserted_by_entity_id": "owner", "predicate": "name", "value": "Alex",
                 "epistemic_type": "user_confirmed", "evidence_ids": ["e:identity"]}
    result = item(plan(objects=[current, obj("owner", "entity")], assertions=[assertion]), "object:x")
    assert "identity" in result["assertion_ids"]
    assert "e:identity" in result["required_evidence_ids"]


def test_unsupported_package_reason_is_stable_under_assertion_order():
    assertions = [
        {"assertion_id": "a", "subject_object_id": "x", "asserted_by_entity_id": "missing",
         "evidence_ids": ["e:a"]},
        {"assertion_id": "b", "subject_object_id": "x", "evidence_ids": []},
    ]
    assert plan(objects=[obj("x")], assertions=assertions) == plan(
        objects=[obj("x")], assertions=list(reversed(assertions)))


def test_evidence_units_from_same_source_remain_separate_packages_and_one_root():
    sources = [{"candidate_ref": f"evidence:segment:{number}", "source_type": "memory", "source_id": "38",
                "evidence_ids": [f"segment:{number}"], "root_source_ref": "memory:38", "week_key": "2026-08-03"}
               for number in (1, 2)]
    result = plan(source_candidates=sources)
    assert [row["candidate_ref"] for row in result["items"]] == ["evidence:segment:1", "evidence:segment:2"]
    assert all(row["source_refs"] == ["source:memory:38"] for row in result["items"])
    assert all(row["independent_source_count"] == 1 for row in result["items"])
    assert result["unmodeled_source_refs"] == ["source:memory:38"]
    assert [row["required_evidence_ids"] for row in result["items"]] == [["segment:1"], ["segment:2"]]


def test_domain_owner_string_and_completion_support_are_mandatory():
    current = obj("x", workflow_status="completed", attributes={
        "committed_by": "owner", "completion_evidence_ids": ["e:done"]})
    result = item(plan(objects=[current, obj("owner", "entity")]), "object:x")
    assert result["object_ids"] == ["owner", "x"]
    assert "e:done" in result["required_evidence_ids"]
    assert "e:done" in result["mandatory_state_refs"]


def test_attribute_completion_and_reopening_are_material_without_changing_workflow_status():
    for before_state, current_state in (("open", "completed"), ("completed", "open")):
        before = obj("x", version=1, attributes={"fulfillment_status": before_state}, evidence_ids=["e:before"])
        current = obj("x", attributes={"fulfillment_status": current_state})
        result = item(plan(objects=[current], previous_versions=[before]), "object:x")
        assert result["tier"] == "material_change"
        assert "current_action_state_changed" in result["reason_codes"]
        assert {"e:before", "e:x"} <= set(result["required_evidence_ids"])
