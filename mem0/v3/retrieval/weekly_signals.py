"""Pure periodic signal rules. Storage, authorization and publication stay outside."""

POLICY_VERSION = "weekly_signals.v1"
TERMINAL = frozenset({"completed", "resolved", "cancelled", "canceled", "retracted",
    "superseded", "withdrawn", "deleted", "forgotten"})
BUSINESS_FIELDS = ("workflow_status", "validity", "primary_project_id", "valid_from", "valid_to")
BUSINESS_ATTRIBUTES = ("action", "decision", "description", "owner", "owner_entity_id",
    "assignee_entity_id", "due_at", "deadline", "condition", "conditions", "blocker",
    "fulfillment_status", "resolution_status", "effective_status", "goal", "outcome",
    "result", "scope", "dependency_project_ids")


def is_open_loop(item):
    attributes = item.get("attributes") or {}
    return not any(str(value or "").lower() in TERMINAL for value in (
        item.get("workflow_status"), item.get("validity"), item.get("retention_status"),
        *(attributes.get(key) for key in ("fulfillment_status", "resolution_status", "effective_status")),
    ))


def current_open_state(item):
    if item.get("object_type") not in {"commitment", "issue", "task", "goal"}:
        return "not_applicable"
    if not is_open_loop(item):
        return "terminal"
    lifecycle_fields = {"workflow_status", "validity", "attributes.fulfillment_status",
        "attributes.resolution_status", "attributes.effective_status"}
    if item.get("support_unavailable") or lifecycle_fields.intersection(item.get("unavailable_fields") or ()):
        return "completion_unknown"
    return "open" if item.get("workflow_status") in {"open", "in_progress", "active", "accepted"} else "completion_unknown"


def business_values(item):
    values = {key: item[key] for key in BUSINESS_FIELDS if key in item}
    attributes = item.get("attributes") or {}
    values.update({"attributes." + key: attributes[key] for key in BUSINESS_ATTRIBUTES if key in attributes})
    return values


def field_support(item, path):
    provenance = item.get("field_provenance") or ()
    matches = [row for row in provenance if row.get("field_name") in {path, path.removeprefix("attributes.")}]
    if matches:
        return tuple(dict.fromkeys(ref for row in matches for ref in row.get("evidence_ids") or ()))
    return tuple(item.get("evidence_ids") or ())


def field_notes(item, path):
    matches = [row for row in item.get("field_provenance") or ()
        if row.get("field_name") in {path, path.removeprefix("attributes.")}]
    return {key: row[key] for row in matches for key in ("user_confirmed", "corrected_at") if row.get(key)}


def business_changes(versions, *, visible_evidence_ids):
    """Keep intermediate transitions and both supports; never guess a predecessor."""
    visible = set(visible_evidence_ids)
    result = []
    previous = None
    for item in versions:
        if item.get("support_unavailable"):
            previous = None
            continue
        after = business_values(item)
        before = business_values(previous) if previous else {}
        fields = []
        for path in sorted(set(before) | set(after)):
            if path not in after:
                continue  # A missing field is unknown, not a supported clearing.
            if path in set(item.get("unavailable_fields") or ()) | set((previous or {}).get("unavailable_fields") or ()):
                continue
            if before.get(path) == after.get(path):
                continue
            new_refs = field_support(item, path)
            newly_observed = previous is None or path not in before
            old_refs = field_support(previous, path) if not newly_observed else ()
            if not new_refs or not set(new_refs).issubset(visible):
                continue
            if not newly_observed and (not old_refs or not set(old_refs).issubset(visible)):
                continue
            fields.append({"field": path, "before": before.get(path), "after": after.get(path),
                "before_refs": old_refs, "after_refs": new_refs,
                "kind": "correction" if field_notes(item, path).get("corrected_at") else "newly_observed" if newly_observed else "transition"})
        if fields:
            result.append({"object_id": item["object_id"], "from_version": previous.get("version") if previous else None,
                "to_version": item.get("version"), "observed_at": item.get("observed_at"),
                "time_basis": "observation_time_only", "changes": fields})
        previous = item
    return tuple(result)


def authorized_version(item, visible_evidence_ids):
    visible = set(visible_evidence_ids)
    result = {key: item.get(key) for key in ("object_id", "object_type", "version", "lock_version", "observed_at")}
    result["evidence_ids"] = tuple(ref for ref in item.get("evidence_ids") or () if ref in visible)
    result["attributes"], unavailable, provenance = {}, [], []
    for path, value in {"title": item.get("title"), **business_values(item)}.items():
        refs = field_support(item, path)
        if not refs or not set(refs).issubset(visible):
            unavailable.append(path)
            continue
        if path.startswith("attributes."):
            result["attributes"][path.removeprefix("attributes.")] = value
        else:
            result[path] = value
        provenance.append({"field_name": path, "evidence_ids": refs, **field_notes(item, path)})
    result["field_provenance"] = provenance
    result["evidence_ids"] = tuple(dict.fromkeys((*result["evidence_ids"], *(ref for row in provenance for ref in row["evidence_ids"]))))
    result["unavailable_fields"] = unavailable
    result["support_unavailable"] = not result["evidence_ids"]
    return result


def signal_read_plan(*, object_ids, start_at, end_at):
    """A deterministic read plan; the caller executes through authorized tools."""
    return {"activity": {"start_at": start_at, "end_at": end_at, "time_basis": "source", "limit": 100},
        "changes": {"object_ids": tuple(sorted(set(object_ids))), "start_at": start_at,
            "end_at": end_at, "include_predecessor": True, "limit": 100}}


def omit_lowest_priority_matter(threads):
    """Omit one complete matter, including its assertions and relation edges."""
    candidates = [row for row in threads if str(row.get("ref") or "").startswith("O")]
    if not candidates:
        return None
    weakest = min(candidates, key=lambda row: (bool(row.get("changes")), row.get("open_state") == "open", row["ref"]))
    ref = weakest["ref"]
    return [row for row in threads if row.get("ref") != ref and row.get("subject") != ref
        and row.get("from") != ref and row.get("to") != ref]
