"""Pure monthly selection policy; callers authorize and load every returned ID.

This produces whole support packages, not facts, summaries, or storage writes.
The input must be a completed authorized scan. Evidence IDs are obligations for
the caller to revalidate, never proof that an arbitrary ID is authorized.
"""

from collections import defaultdict

from .service import MemoryQueryService


_TIERS = ("material_change", "meaningful_connection", "independent_recurrence", "necessary_context")
_CONNECTIONS = {"depends_on", "blocks", "constrains", "conflicts_with", "contradicts"}
_OWNERSHIP = {"owned_by", "committed_by"}
_HIGH_PRIORITY = {"high", "critical", "urgent", "highest"}
_TERMINAL = {"completed", "cancelled", "resolved", "retracted", "superseded", "withdrawn", "retired", "closed"}
_OWNER_FIELDS = ("owner_entity_id", "owner_object_id", "committed_by_entity_id",
                 "corrected_by_object_id", "superseded_by_object_id")


class _Unsupported(ValueError):
    pass


def _refs(values):
    if isinstance(values, str):
        values = (values,)
    return {str(value) for value in values or () if value is not None and str(value)}


def _support(row):
    result = _refs(row.get("evidence_ids"))
    for name, values in (row.get("attributes") or {}).items():
        if name.endswith("_evidence_ids"):
            result.update(_refs(values))
    for field in row.get("field_provenance") or ():
        result.update(_refs(field.get("evidence_ids")))
    return result


def _effective(row):
    return row.get("validity", "active") == "active" and row.get("retention_status") != "deleted"


def _index(rows, key):
    result = {}
    for row in rows:
        identifier = str(row[key])
        if identifier in result and result[identifier] != row:
            raise ValueError(f"conflicting monthly {key}: {identifier}")
        result[identifier] = row
    return result


def _semantic(row):
    attributes = dict(row.get("attributes") or {})
    # These flags suggest a query, but do not establish a substantive change.
    for name in ("decision_changed", "last_changed_at", "occurrence_count"):
        attributes.pop(name, None)
    return {"object_id": row["object_id"], "workflow_status": row.get("workflow_status"),
            "validity": row.get("validity"), "attributes": attributes}


def _epistemic(rows):
    types = {row.get("epistemic_type") for row in rows}
    return "hypothesis" if "hypothesis" in types else "inferred" if "inferred" in types else "supported"


def _action_state_changed(before, after):
    states = lambda row: {"workflow_status": row.get("workflow_status"), **{
        name: (row.get("attributes") or {}).get(name)
        for name in ("fulfillment_status", "resolution_status", "effective_status")}}
    old, new = states(before), states(after)
    return any(old[key] != new[key] and (old[key] in _TERMINAL or new[key] in _TERMINAL)
               for key in old)


def _priority(row, assertions, user_context):
    """Return (supported importance, explicitly confirmed priority, evidence)."""
    important, required, evidence = False, False, set()
    for field in row.get("field_provenance") or ():
        refs = _refs(field.get("evidence_ids"))
        name = field.get("field_name")
        value = row.get(name, (row.get("attributes") or {}).get(name))
        high = name == "priority" and str(value).lower() in _HIGH_PRIORITY
        semantic = name == "importance" and isinstance(value, (int, float)) and value > 0
        if refs and (high or semantic):
            important = True
            required |= high and field.get("user_confirmed") is True
            evidence.update(refs)
    for assertion in assertions:
        if (assertion.get("predicate") not in {"priority", "explicit_priority"}
                or str(assertion.get("value")).lower() not in _HIGH_PRIORITY
                or not _effective(assertion) or not _support(assertion)
                or assertion.get("epistemic_type") in {"inferred", "hypothesis"}):
            continue
        important = True
        required |= assertion.get("epistemic_type") == "user_confirmed"
        evidence.update(_support(assertion))
    for priority in user_context.get("explicit_priorities") or ():
        if (str(priority.get("object_id")) != str(row["object_id"])
                or str(priority.get("priority")).lower() not in _HIGH_PRIORITY
                or priority.get("verified") is not True or not _support(priority)
                or priority.get("epistemic_type") in {"inferred", "hypothesis"}):
            continue
        important = True
        required = True
        evidence.update(_support(priority))
    return important, required, evidence


def plan_monthly_candidates(*, objects, previous_versions, assertions, relations,
                            source_occurrences, source_candidates, user_context,
                            window, snapshot_event_id):
    """Rank complete candidate packages without changing evidence or identities.

    ``required`` means a sourced explicit user priority cannot silently be
    dropped for budget. Ordinary inferred importance never sets this flag.
    Relations retain their IDs; the caller loads their direction and full data.
    Unsupported closures are returned separately, not salvaged by deleting
    inconvenient support. No timestamps or input order affect ranking.
    """
    object_by_id = _index(objects, "object_id")
    assertion_by_id = _index(assertions, "assertion_id")
    relation_by_id = _index(relations, "relation_id")
    by_subject, by_endpoint, previous, occurrences = (defaultdict(list) for _ in range(4))
    for _, row in sorted(assertion_by_id.items()):
        if _effective(row):
            by_subject[str(row["subject_object_id"])].append(row)
    for _, row in sorted(relation_by_id.items()):
        if _effective(row):
            for key in {str(row["source_object_id"]), str(row["target_object_id"])}:
                by_endpoint[key].append(row)
    for row in previous_versions:
        previous[str(row["object_id"])].append(row)
    for row in source_occurrences:
        occurrences[str(row["candidate_ref"])].append(row)

    service = MemoryQueryService()
    decision_signals = {str(row["object_id"]) for row in service.select_decision_changes(objects)}
    conflict_signals = {str(row["object_id"]) for row in service.select_unresolved_conflicts(objects)}
    repeated_signals = {str(row["object_id"]) for row in service.select_repeated_issues(objects)}
    cross_project_signals = {str(row["object_id"]) for row in service.select_cross_project_dependencies(objects)}

    def closure(root):
        obj_ids, assertion_ids, relation_ids, evidence, state = (set() for _ in range(5))
        epistemic_rows, path = [], set()

        def add_object(identifier):
            if identifier in path:
                raise _Unsupported("cyclic_dependency")
            if identifier in obj_ids:
                return
            row = object_by_id.get(identifier)
            if row is None or not _effective(row):
                raise _Unsupported("missing_or_unavailable_object")
            support = _support(row)
            if not support:
                raise _Unsupported("missing_object_support")
            path.add(identifier)
            obj_ids.add(identifier)
            evidence.update(support)
            state.update(support)
            epistemic_rows.append(row)
            for assertion in by_subject[identifier]:
                support = _support(assertion)
                if not support:
                    raise _Unsupported("missing_assertion_support")
                assertion_ids.add(str(assertion["assertion_id"]))
                evidence.update(support)
                state.update(support)
                epistemic_rows.append(assertion)
                asserted_by = assertion.get("asserted_by_entity_id")
                if asserted_by and str(asserted_by) != identifier:
                    add_object(str(asserted_by))
            attributes = row.get("attributes") or {}
            for field in _OWNER_FIELDS:
                if attributes.get(field):
                    add_object(str(attributes[field]))
            for field in ("owner", "committed_by", "decision_owner"):
                owner = attributes.get(field)
                if isinstance(owner, str) and owner:
                    add_object(owner)
                elif isinstance(owner, dict) and (owner.get("object_id") or owner.get("entity_id")):
                    add_object(str(owner.get("object_id") or owner["entity_id"]))
            for relation in by_endpoint[identifier]:
                if (relation.get("relation_type") in _OWNERSHIP
                        and str(relation["source_object_id"]) == identifier):
                    add_relation(relation)
                    add_object(str(relation["target_object_id"]))
            path.remove(identifier)

        def add_relation(row):
            if not _support(row):
                raise _Unsupported("missing_relation_support")
            relation_ids.add(str(row["relation_id"]))
            evidence.update(_support(row))
            epistemic_rows.append(row)

        add_object(root)
        connections = [row for row in by_endpoint[root] if row.get("relation_type") in _CONNECTIONS]
        for relation in connections:
            add_relation(relation)
            add_object(str(relation["source_object_id"]))
            add_object(str(relation["target_object_id"]))
        changed = []
        for before in previous[root]:
            if service.compare_versions(_semantic(before), _semantic(object_by_id[root]))["changes"]:
                if not _support(before):
                    raise _Unsupported("missing_previous_support")
                evidence.update(_support(before))
                changed.append(before)
        return obj_ids, assertion_ids, relation_ids, evidence, state, epistemic_rows, connections, changed

    def package(ref, rows, *, obj_ids=(), assertion_ids=(), relation_ids=(), evidence=(), state=(),
                tier="necessary_context", reasons=(), required=False, epistemic="supported", source_refs=()):
        roots, weeks, recordings, events, support = (set() for _ in range(5))
        root_weeks = defaultdict(set)
        for occurrence in rows:
            if not _support(occurrence) or not occurrence.get("root_source_ref"):
                continue
            roots.add(str(occurrence["root_source_ref"]))
            support.update(_support(occurrence))
            if occurrence.get("week_key"):
                root_weeks[str(occurrence["root_source_ref"])].add(str(occurrence["week_key"]))
            if str(occurrence["root_source_ref"]).startswith("memory:"):
                recordings.add(str(occurrence["root_source_ref"])[7:])
            if occurrence.get("event_id"):
                events.add(str(occurrence["event_id"]))
        # Derivative timestamps cannot turn one original source into weeks of
        # independent support. Ambiguous dates conservatively use the earliest.
        weeks.update(min(values) for values in root_weeks.values() if values)
        reasons = set(reasons)
        if len(roots) >= 2:
            reasons.add("repeated_mentions")
            if len(events) >= 2:
                reasons.add("repeated_occurrences")
            if tier == "necessary_context":
                tier = "independent_recurrence"
        return {"candidate_ref": ref, "tier": tier, "reason_codes": sorted(reasons),
                "required": required, "object_ids": sorted(obj_ids),
                "assertion_ids": sorted(assertion_ids), "relation_ids": sorted(relation_ids),
                "required_evidence_ids": sorted(set(evidence) | support),
                "mandatory_state_refs": sorted(state), "root_source_refs": sorted(roots),
                "supporting_week_keys": sorted(weeks), "independent_source_count": len(roots),
                "independent_recording_count": len(recordings), "verified_occurrence_count": len(events),
                "epistemic_type": epistemic, "source_refs": sorted(source_refs)}

    items, unavailable, unmodeled = [], [], []
    candidate_count = 0
    for identifier, row in sorted(object_by_id.items()):
        if row.get("object_type") in {"entity", "meeting"}:
            continue
        candidate_count += 1
        ref = f"object:{identifier}"
        important, required, priority_support = _priority(row, by_subject[identifier], user_context)
        try:
            obj_ids, assertion_ids, relation_ids, evidence, state, epistemic_rows, connections, changed = closure(identifier)
            evidence.update(priority_support)
            reasons, tier = set(), "necessary_context"
            if connections:
                tier = "meaningful_connection"
                reasons.add("supported_connection")
                if identifier in cross_project_signals:
                    reasons.add("cross_project_dependency")
                if identifier in conflict_signals:
                    reasons.add("supported_conflict")
                for relation in connections:
                    goal = object_by_id[str(relation["target_object_id"])]
                    if (relation.get("relation_type") in {"blocks", "constrains"}
                            and goal.get("object_type") == "goal"
                            and goal.get("workflow_status") not in _TERMINAL
                            and relation.get("epistemic_type") not in {"inferred", "hypothesis"}):
                        tier = "material_change"
                        reasons.add("affects_active_goal")
            status_change = any(_action_state_changed(before, row) for before in changed)
            if important:
                tier = "material_change"
                reasons.add("explicit_priority" if required else "supported_importance")
                if changed and (row.get("object_type") == "decision" or identifier in decision_signals):
                    reasons.add("decision_changed")
            if status_change:
                tier = "material_change"
                reasons.add("current_action_state_changed")
            result = package(ref, occurrences[ref], obj_ids=obj_ids, assertion_ids=assertion_ids,
                             relation_ids=relation_ids, evidence=evidence, state=state, tier=tier,
                             reasons=reasons, required=required, epistemic=_epistemic(epistemic_rows))
            if identifier in repeated_signals and result["independent_source_count"] >= 2:
                result["reason_codes"] = sorted(set(result["reason_codes"]) | {"repeated_issue_mentions"})
            items.append(result)
        except _Unsupported as error:
            unavailable.append({"candidate_ref": ref, "reason": str(error), "required": required})

    objects_by_evidence = defaultdict(set)
    for row in (*objects, *previous_versions):
        for evidence_id in _support(row):
            objects_by_evidence[evidence_id].add(str(row["object_id"]))
    for row in assertions:
        if _effective(row):
            for evidence_id in _support(row):
                objects_by_evidence[evidence_id].add(str(row["subject_object_id"]))
    source_by_ref = {}
    for source in source_candidates:
        source_ref = f"source:{source['source_type']}:{source['source_id']}"
        ref = str(source.get("candidate_ref") or source_ref)
        if ref != source_ref and not (
                ref.startswith("evidence:") and _refs(source.get("evidence_ids")) == {ref[9:]}):
            raise ValueError("source candidate must identify its typed source or one complete evidence unit")
        if ref in source_by_ref and source_by_ref[ref] != source:
            raise ValueError(f"conflicting monthly source: {ref}")
        source_by_ref[ref] = source
    for ref, source in sorted(source_by_ref.items()):
        candidate_count += 1
        source_ref = f"source:{source['source_type']}:{source['source_id']}"
        if not _support(source) or not _effective(source):
            unavailable.append({"candidate_ref": ref, "reason": "missing_source_support", "required": False})
            continue
        signals = [signal for signal in source.get("structured_signals") or ()
                   if signal.get("kind") in {"decision", "blocker", "outcome"}
                   and _support(signal) and _support(signal) <= _support(source)]
        priorities = [signal for signal in signals
                      if str(signal.get("priority")).lower() in _HIGH_PRIORITY
                      and signal.get("epistemic_type") not in {"inferred", "hypothesis"}]
        required = any(signal.get("user_confirmed") is True for signal in priorities)
        try:
            roots = _refs(source.get("object_ids"))
            for signal in signals:
                roots.update(_refs(signal.get("object_ids")))
            for evidence_id in _support(source):
                roots.update(objects_by_evidence[evidence_id])
            for assertion_id in _refs(source.get("assertion_ids")):
                assertion = assertion_by_id.get(assertion_id)
                if not assertion or not _effective(assertion):
                    raise _Unsupported("missing_assertion_support")
                roots.add(str(assertion["subject_object_id"]))
            for relation_id in _refs(source.get("relation_ids")):
                relation = relation_by_id.get(relation_id)
                if not relation or not _effective(relation) or not _support(relation):
                    raise _Unsupported("missing_relation_support")
                roots.update((str(relation["source_object_id"]), str(relation["target_object_id"])))
            obj_ids, assertion_ids, relation_ids, evidence, state = (set() for _ in range(5))
            evidence.update(_support(source))
            epistemic_rows = [source, *signals]
            for root in sorted(roots):
                closed = closure(root)
                for target, values in zip((obj_ids, assertion_ids, relation_ids, evidence, state), closed[:5]):
                    target.update(values)
                epistemic_rows.extend(closed[5])
            for relation_id in _refs(source.get("relation_ids")):
                relation_ids.add(relation_id)
                evidence.update(_support(relation_by_id[relation_id]))
                epistemic_rows.append(relation_by_id[relation_id])
            is_unmodeled = not signals and not any(
                object_by_id[root].get("object_type") != "entity" for root in roots)
            if is_unmodeled:
                unmodeled.append(source_ref)
            reasons = ["unmodeled_source_coverage" if is_unmodeled else "source_context"]
            reasons.extend(f"structured_{signal['kind']}" for signal in signals)
            if priorities:
                reasons.append("explicit_priority" if required else "supported_importance")
            items.append(package(ref, [source, *occurrences[ref]], evidence=evidence,
                                 obj_ids=obj_ids, assertion_ids=assertion_ids, relation_ids=relation_ids,
                                 state=state, reasons=reasons, source_refs=[source_ref], required=required,
                                 tier="material_change" if priorities else "necessary_context",
                                 epistemic=_epistemic(epistemic_rows)))
        except _Unsupported as error:
            unavailable.append({"candidate_ref": ref, "reason": str(error), "required": required})
    items.sort(key=lambda row: (_TIERS.index(row["tier"]), not row["required"],
                               -len(row["supporting_week_keys"]), -row["independent_source_count"],
                               row["candidate_ref"]))
    return {"selection_version": 1, "snapshot_event_id": snapshot_event_id,
            "items": items, "candidate_count": candidate_count, "scan_complete": True,
            "unmodeled_source_refs": sorted(set(unmodeled)),
            "unavailable_candidates": sorted(unavailable, key=lambda row: row["candidate_ref"])}
