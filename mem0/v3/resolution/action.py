"""Apply source action claims to current responsibility as a single state change."""

from mem0.v3.action_roles import ACTION_ROLES, EXECUTOR_FIELDS
from mem0.v3.extraction import ClaimLifecycleSignal


def reconcile_action(*, existing, claim, proposed, assertion_ref):
    if existing is None:
        return proposed, False, ()
    current = {**existing.current_state, **existing.attributes}
    attrs = proposed["attributes"]
    preserved = set()

    def preserve(fields):
        for field in fields:
            proposed.pop(field, None)
            attrs.pop(field, None)
        preserved.update(fields)

    owner_field = "committed_by" if claim.claim_type.value == "commitment" else "owner_entity_id"
    owner = proposed.get(owner_field, attrs.get(owner_field))
    old_owner = current.get(owner_field)
    owner_changed = bool(claim.owner_mention and old_owner and owner != old_owner)
    locked_owner = any(item.get("user_locked") and item.get("field_name") in EXECUTOR_FIELDS
                       for item in existing.field_provenance)
    established = existing.workflow_status in {"accepted", "in_progress", "completed", "cancelled"}
    intent = claim.task_intent.value if claim.task_intent else None
    replacement = claim.lifecycle_signal is ClaimLifecycleSignal.SUPERSEDES
    request_only = established and claim.claim_type.value != "commitment" and not replacement and (
        claim.lifecycle_signal is ClaimLifecycleSignal.NONE or not owner_changed
    ) and (
        intent == "requested" or (intent is None and not claim.owner_mention)
        or (intent == "assigned" and not owner_changed)
    )
    conflict = owner_changed and not request_only and (locked_owner or not replacement)
    warnings = ()
    if locked_owner and owner_changed:
        warnings = (f"user_locked_field_preserved:{owner_field}",)
    if request_only or conflict:
        preserve(EXECUTOR_FIELDS | set(ACTION_ROLES["initiator"]) | {
            "execution_intent", "modality", "negated", "condition", "action", "due_at",
            "committed_at", "title",
        })
    else:
        # Absence of a role is not a new unresolved assignment. A supplied unresolved
        # role keeps its explicit null identity, never borrowing an old entity ID.
        if claim.owner_mention is None or locked_owner:
            preserve(EXECUTOR_FIELDS)
        if claim.initiator_mention is None:
            preserve(ACTION_ROLES["initiator"])
        if intent is None:
            preserve({"execution_intent"})
    locked_initiator = any(item.get("user_locked") and item.get("field_name") in ACTION_ROLES["initiator"]
                           for item in existing.field_provenance)
    if locked_initiator:
        preserve(ACTION_ROLES["initiator"])
    if conflict or (established and claim.lifecycle_signal is ClaimLifecycleSignal.NONE):
        preserve({"workflow_status", "fulfillment_status", "completion_evidence_ids"})
    if conflict:
        attrs.update(has_unresolved_conflict=True, conflict_assertion_refs=tuple(dict.fromkeys((
            *(current.get("conflict_assertion_refs") or ()), assertion_ref,
        ))))
    elif replacement and not locked_owner:
        attrs.update(has_unresolved_conflict=False, conflict_assertion_refs=())
    proposed["field_provenance"] = tuple(
        item for item in proposed["field_provenance"] if item.field_name not in preserved
    )
    return proposed, conflict, warnings
