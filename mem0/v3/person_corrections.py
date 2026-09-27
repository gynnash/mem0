"""Typed, user-confirmed person corrections; no storage or model side effects."""

from datetime import datetime
from typing import Literal

from pydantic import Field, JsonValue

from mem0.v3.contracts import (
    AssertionMutation, DomainEvent, EvidenceCreate, ObjectMutation, SourceRef,
)
from mem0.v3.contracts.base import FrozenContract
from mem0.v3.contracts.payloads import ObjectMutationPayload
from mem0.v3.domain.models import Evidence, FieldProvenance
from mem0.v3.planner import MemoryChangeDraft, MemoryPlanner


class PersonCorrection(FrozenContract):
    target_id: str = Field(min_length=1)
    operation: Literal["update", "complete", "retract", "replace_assertion"]
    field: str = ""
    value: JsonValue = ""
    statement: str = Field(min_length=1, max_length=2000)
    effective_at: datetime | None = None


_FIELDS = {
    "commitment": {"action", "due_at", "description"},
    "decision": {"decision", "rationale", "description", "effective_from"},
    "issue": {"description", "severity"},
    "project": {"description"},
    "task": {"description"},
    "goal": {"description"},
    "preference": {"description"},
}


def correction_patch(intent: PersonCorrection, target):
    patch = _field_patch(intent, target)
    if intent.effective_at:
        patch["asserted_at" if intent.operation == "replace_assertion" else "valid_from"] = intent.effective_at
    if intent.operation != "replace_assertion":
        ObjectMutationPayload.model_validate(patch)
    return patch


def _field_patch(intent: PersonCorrection, target):
    """The preview and the commit use this same operation-to-field mapping."""
    kind = target.get("object_type")
    if intent.operation == "replace_assertion":
        if not target.get("assertion_id"):
            raise ValueError("correction requires an assertion target")
        return {"value": intent.value}
    if intent.operation == "update":
        if intent.field not in _FIELDS.get(kind, set()):
            raise ValueError("unsupported field for the selected object type")
        if intent.value is None and intent.field in {"due_at", "effective_from"}:
            return {intent.field: None}
        if not isinstance(intent.value, str) or not intent.value.strip():
            raise ValueError("a replacement value is required")
        value = intent.value
        if intent.field in {"due_at", "effective_from"}:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if value.tzinfo is None:
                raise ValueError("a resolved date and timezone are required")
        return {intent.field: value}
    if intent.operation == "complete":
        if kind == "commitment":
            return {"workflow_status": "completed", "fulfillment_status": "completed"}
        if kind == "issue":
            return {"workflow_status": "completed", "resolution_status": "resolved"}
        if kind in {"task", "project", "goal"}:
            return {"workflow_status": "completed"}
        raise ValueError("this item does not have a completion state")
    if intent.operation == "retract":
        if kind not in _FIELDS:
            raise ValueError("identity and original evidence cannot be retracted here")
        return {"validity": "retracted"}
    raise ValueError("unsupported correction")


def plan_person_correction(*, changeset_id, user_id, workspace_id, base_state_version,
                           intent: PersonCorrection, target, correction_evidence: Evidence):
    patch = correction_patch(intent, target)
    if str(target.get("object_id") or target.get("assertion_id")) != intent.target_id:
        raise ValueError("correction target mismatch")
    ref = "person:correction:evidence"
    objects, assertions, retractions = (), (), ()
    expected = {}
    if intent.operation == "replace_assertion":
        from mem0.v3.contracts import RetractionMutation
        assertions = (AssertionMutation(logical_ref="person:corrected_assertion", operation="create",
            evidence_ids=(ref,), payload={
                "subject_object_ref": target["subject_object_id"], "predicate": target["predicate"],
                "value": intent.value, "epistemic_type": "user_confirmed",
                "modality": target.get("modality", "asserted"), "polarity": target.get("polarity", "positive"),
                "confidence": 1.0, "asserted_at": intent.effective_at or correction_evidence.recorded_at,
            }),)
        retractions = (RetractionMutation(logical_ref="person:old_assertion", target_type="assertion",
            target_id=intent.target_id, reason="explicit_person_correction", evidence_ids=(ref,)),)
    else:
        expected = {intent.target_id: int(target["lock_version"])}
        if intent.operation == "complete" and target["object_type"] == "commitment":
            patch["completion_evidence_ids"] = (ref,)
        if intent.operation == "complete" and target["object_type"] == "issue":
            patch["resolution_evidence_ids"] = (ref,)
        patch["field_provenance"] = tuple(FieldProvenance(field_name=field, evidence_ids=(ref,),
            user_confirmed=True, user_locked=True, corrected_at=correction_evidence.recorded_at)
            for field in patch if field not in {"completion_evidence_ids", "resolution_evidence_ids"})
        if intent.effective_at:
            patch["valid_from"] = intent.effective_at
        if intent.operation == "update" and intent.value is None:
            patch["attributes"] = {intent.field: patch.pop(intent.field)}
        objects = (ObjectMutation(logical_ref="person:corrected_object", operation="update",
            object_type=target["object_type"], object_id=intent.target_id,
            expected_version=target["lock_version"], evidence_ids=(ref,), payload=patch),)
    return MemoryPlanner().plan(MemoryChangeDraft(
        changeset_id=changeset_id, user_id=user_id, workspace_id=workspace_id,
        source_ref=SourceRef(source_type="person_correction", source_id=correction_evidence.source_id),
        base_state_version=base_state_version, expected_object_versions=expected,
        evidence_creates=(EvidenceCreate(logical_ref=ref, evidence=correction_evidence),),
        object_mutations=objects, assertion_mutations=assertions, retractions=retractions,
        domain_events=(DomainEvent(event_type="memory.dependencies_invalidated",
            aggregate_ref=intent.target_id, payload={"reason": "explicit_person_correction"}),),
    ))


def plan_person_restore(*, changeset_id, user_id, workspace_id, base_state_version,
                        original_intent, original_target, current_target, correction_evidence):
    """Compensate exactly one confirmed operation at its resulting object version."""
    intent = PersonCorrection.model_validate(original_intent)
    if intent.operation == "replace_assertion":
        inverse = PersonCorrection(target_id=current_target["assertion_id"], operation="replace_assertion",
            value=original_target["value"], statement="User undid a prior correction.")
        return plan_person_correction(changeset_id=changeset_id, user_id=user_id, workspace_id=workspace_id,
            base_state_version=base_state_version, intent=inverse, target=current_target,
            correction_evidence=correction_evidence)
    changed = correction_patch(intent, original_target)
    ref = "person:undo:evidence"
    fields, attributes = {}, {}
    for key in changed:
        value = original_target.get(key)
        if key in {"description", "workflow_status", "validity", "valid_from"}:
            fields[key] = value if value is not None else ""
        else:
            # Typed object fields are stored in the attributes projection. Explicit
            # null restores a previously unknown date instead of leaving the new one.
            attributes[key] = value
    if intent.operation == "complete":
        for key in ("completion_evidence_ids", "resolution_evidence_ids"):
            if key in current_target:
                attributes[key] = original_target.get(key, [])
    if attributes:
        fields["attributes"] = attributes
    fields["field_provenance"] = tuple(FieldProvenance(field_name=key, evidence_ids=(ref,),
        user_confirmed=True, user_locked=True, corrected_at=correction_evidence.recorded_at)
        for key in changed)
    oid = current_target["object_id"]
    return MemoryPlanner().plan(MemoryChangeDraft(
        changeset_id=changeset_id, user_id=user_id, workspace_id=workspace_id,
        source_ref=SourceRef(source_type="person_correction_undo", source_id=correction_evidence.source_id),
        base_state_version=base_state_version, expected_object_versions={oid: current_target["lock_version"]},
        evidence_creates=(EvidenceCreate(logical_ref=ref, evidence=correction_evidence),),
        object_mutations=(ObjectMutation(logical_ref="person:restored_object", operation="update",
            object_type=current_target["object_type"], object_id=oid,
            expected_version=current_target["lock_version"], evidence_ids=(ref,), payload=fields),),
        domain_events=(DomainEvent(event_type="memory.dependencies_invalidated", aggregate_ref=oid,
            payload={"reason": "person_correction_undone"}),),
    ))
