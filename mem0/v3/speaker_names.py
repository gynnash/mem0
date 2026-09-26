"""Deterministic, source-scoped name correction. No extraction or model calls."""

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

from mem0.v3.contracts import (
    AssertionMutation, DomainEvent, EvidenceCreate, ObjectMutation,
    ObjectMutationPayload, RelationMutation, RetractionMutation, SourceRef,
)
from mem0.v3.domain import Evidence
from mem0.v3.planner import MemoryChangeDraft, MemoryPlanner


def plan_person_identity_merge(*, operation_key, user_id, workspace_id, base_state_version,
                               source_speaker_ids, target_speaker_id, source_object_ids,
                               target_object_id, target_name, evidence, objects, assertions, relations, now):
    """Migrate a user-confirmed identity without extracting any new facts."""
    source_refs = {f"speaker:{value}" for value in source_speaker_ids}
    if source_object_ids and not target_object_id:
        raise ValueError("A canonical merge target is required")
    target_ref = f"speaker:{target_speaker_id}"
    identity_map = dict.fromkeys(source_object_ids, target_object_id) if target_object_id else {}
    if target_object_id in identity_map or target_ref in source_refs:
        raise ValueError("Merge source and target must differ")
    evidence_map = {}
    creates, retracts, obj_changes, ast_changes, rel_changes = [], [], [], [], []
    for item in evidence:
        if item.user_id != user_id or item.workspace_id != workspace_id:
            raise ValueError("Merge evidence is outside the authorized scope")
        if item.speaker_id not in source_refs:
            continue
        new_id = _id("ev", operation_key, item.evidence_id)
        evidence_map[item.evidence_id] = new_id
        corrected = Evidence.model_validate({**item.model_dump(), "evidence_id": new_id,
                                             "speaker_id": target_ref, "created_at": now})
        creates.append(EvidenceCreate(logical_ref=new_id, evidence=corrected))
        retracts.append(RetractionMutation(logical_ref="ret:" + item.evidence_id,
                        target_type="evidence", target_id=item.evidence_id, reason="person_identity_merge"))

    def refs(values):
        return tuple(dict.fromkeys(evidence_map.get(value, value) for value in values))

    identity_fields = {"subject_object_id", "subject_object_ref", "asserted_by_entity_id", "owner",
                       "committed_by", "committed_to", "decision_owner", "source_object_id", "target_object_id",
                       "entity_id", "entity_ids", "object_id", "object_ids", "participant_ids"}
    evidence_fields = {"evidence_id", "evidence_ids", "completion_evidence_ids", "resolution_evidence_ids",
                       "source_evidence_id", "source_evidence_ids", "evidence_refs"}
    def migrate(value, key=""):
        if isinstance(value, dict):
            return {k: migrate(v, k) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [migrate(v, key) for v in value]
        if isinstance(value, str):
            if key in identity_fields:
                return identity_map.get(value, value)
            if key in evidence_fields:
                return evidence_map.get(value, value)
            if key in {"speaker_ref", "speaker_id", "participant_refs"} and value in source_refs:
                return target_ref
        if key in {"speaker_id", "person_id", "owner_id"} and str(value) in {str(item) for item in source_speaker_ids}:
            return str(target_speaker_id) if isinstance(value, str) else target_speaker_id
        return value

    assertion_map = {}
    for row in assertions:
        payload = migrate(dict(row["payload"]))
        new_refs = refs(row["evidence_ids"])
        if payload == row["payload"] and new_refs == tuple(row["evidence_ids"]):
            continue
        new_id = _id("ast", operation_key, row["assertion_id"])
        assertion_map[row["assertion_id"]] = new_id
        ast_changes.append(AssertionMutation(logical_ref=new_id, assertion_id=new_id, operation="create",
                           evidence_ids=new_refs, payload=payload))
        retracts.append(RetractionMutation(logical_ref="ret:" + row["assertion_id"], target_type="assertion",
                        target_id=row["assertion_id"], reason="person_identity_merge"))
    relation_groups = {}
    for row in relations:
        source = identity_map.get(row["source_object_id"], row["source_object_id"])
        target = identity_map.get(row["target_object_id"], row["target_object_id"])
        semantic = {key: value for key, value in row["payload"].items() if key != "confidence"}
        key = (source, target, row["relation_type"], json.dumps(semantic, sort_keys=True, default=str))
        relation_groups.setdefault(key, []).append(row)
    for (source, target, kind, _), group in relation_groups.items():
        changed = any(source != row["source_object_id"] or target != row["target_object_id"]
                      or refs(row["evidence_ids"]) != tuple(row["evidence_ids"]) for row in group)
        if not changed:
            continue
        row = group[0]
        new_refs = tuple(dict.fromkeys(ref for item in group for ref in refs(item["evidence_ids"])))
        if source != target:
            new_id = _id("rel", operation_key, ":".join(sorted(item["relation_id"] for item in group)))
            payload = {**row["payload"], "confidence": max(item["payload"].get("confidence", 0) for item in group)}
            rel_changes.append(RelationMutation(logical_ref=new_id, relation_id=new_id, operation="create",
                               source_object_ref=source, target_object_ref=target, evidence_ids=new_refs,
                               relation_type=kind, payload=payload))
        for item in group:
            retracts.append(RetractionMutation(logical_ref="ret:" + item["relation_id"], target_type="relation",
                            target_id=item["relation_id"], reason="person_identity_merge"))
    versions = {}
    merge_evidence = tuple(dict.fromkeys(ref for row in objects
                                       if row["object_id"] in {*source_object_ids, target_object_id}
                                       for ref in refs(row["evidence_ids"])))
    for row in objects:
        oid = row["object_id"]
        if oid in source_object_ids:
            versions[oid] = row["lock_version"]
            retracts.append(RetractionMutation(logical_ref="ret:" + oid, target_type="object",
                            target_id=oid, reason="person_identity_merge", evidence_ids=merge_evidence))
            continue
        before = dict(row["payload"])
        after = migrate(before)
        new_refs = refs(row["evidence_ids"])
        if oid == target_object_id:
            after["title"] = target_name
            after["attributes"] = {**after.get("attributes", {}), "speaker_ref": target_ref,
                                   "merged_source_object_ids": list(dict.fromkeys([
                                       *(after.get("attributes", {}).get("merged_source_object_ids") or []),
                                       *source_object_ids,
                                   ]))}
            aliases = list(after["attributes"].get("identity_aliases") or [])
            if not any((alias.get("value") if isinstance(alias, dict) else alias) == target_name for alias in aliases):
                aliases.append({"value": target_name, "confidence": 1, "user_confirmed": True,
                                "evidence_ids": merge_evidence})
            after["attributes"]["identity_aliases"] = aliases
            new_refs = merge_evidence
        patch = {key: value for key, value in after.items()
                 if key in ObjectMutationPayload.model_fields and value != before.get(key)}
        if not patch and new_refs == tuple(row["evidence_ids"]):
            continue
        if row["object_type"] == "meeting":
            patch.update({key: before.get(key, before.get("attributes", {}).get(key))
                          for key in ("canonical_key", "external_memory_id")})
        versions[oid] = row["lock_version"]
        if oid == target_object_id:
            patch["field_provenance"] = [{"field_name": "title", "evidence_ids": merge_evidence,
                                         "user_confirmed": True, "user_locked": True, "corrected_at": now}]
        obj_changes.append(ObjectMutation(logical_ref="obj:" + oid, operation="update",
                           object_id=oid, object_type=row["object_type"], expected_version=row["lock_version"],
                           evidence_ids=new_refs, payload=patch))
    if not (creates or retracts or obj_changes or ast_changes or rel_changes):
        return None
    return MemoryPlanner().plan(MemoryChangeDraft(
        changeset_id=_id("chg", operation_key, "identity"), user_id=user_id, workspace_id=workspace_id,
        source_ref=SourceRef(source_type="person_identity_merge", source_id=operation_key),
        base_state_version=base_state_version, expected_object_versions=versions,
        evidence_creates=tuple(creates), object_mutations=tuple(obj_changes),
        assertion_mutations=tuple(ast_changes), relation_mutations=tuple(rel_changes),
        retractions=tuple(retracts), domain_events=(DomainEvent(
            event_type="memory.person_identity_merged", aggregate_ref=target_object_id or target_ref,
            payload={"reason": "person_identity_merge", "source_object_ids": list(source_object_ids),
                     "source_speaker_ids": list(source_speaker_ids), "target_speaker_id": target_speaker_id,
                     "target_object_id": target_object_id, "evidence_replacements": evidence_map,
                     "assertion_replacements": assertion_map},
        ),),
    ))


@dataclass(frozen=True)
class SpeakerNameScope:
    memory_id: str
    transcript_version: int
    segment_indexes: tuple[int, ...]
    old_names: tuple[str, ...]
    new_name: str
    original_speaker_ref: str | None = None
    speaker_ref: str | None = None

    def contains(self, evidence: Evidence) -> bool:
        if evidence.memory_id != self.memory_id or evidence.transcript_version != self.transcript_version:
            return False
        # Unknown/range-free provenance cannot establish a segment reference.
        match = re.fullmatch(r"(\d+):(\d+)(?:\.\.(\d+):(\d+))?", evidence.source_id)
        if not match or int(match[1]) != self.transcript_version:
            return False
        first, last = int(match[2]), int(match[4] or match[2])
        if match[3] and int(match[3]) != self.transcript_version:
            return False
        selected = set(self.segment_indexes)
        return first <= last and last - first < len(selected) and all(
            index in selected for index in range(first, last + 1)
        )


def replace_names(value: str, names: Mapping[str, str]) -> str:
    """Replace complete names once, so swaps cannot cascade into each other."""
    if not names:
        return value
    if value in names:
        return names[value]
    patterns = []
    for name in sorted(names, key=len, reverse=True):
        if not name:
            continue
        boundary = r"A-Za-z0-9_-" if re.fullmatch(r"(?i)speaker[-_ ]?\d+", name) else r"\w-"
        patterns.append(r"(?<![" + boundary + r"])(?:" + re.escape(name) + r")(?![" + boundary + r"])")
    expression = "|".join(patterns)
    if not expression:
        return value
    return re.sub(expression, lambda match: names[match[0]], value)


_TEXT_FIELDS = frozenset({
    "name", "display_name", "speaker_name", "title", "description", "content",
    "summary", "text", "decision", "rationale", "action", "canonical_label",
})


def correct_display_fields(value, names, *, text_value=False):
    if isinstance(value, str):
        return replace_names(value, names) if text_value else value
    if isinstance(value, dict):
        return {key: correct_display_fields(item, names, text_value=key in _TEXT_FIELDS)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [correct_display_fields(item, names, text_value=text_value) for item in value]
    return value


def _id(kind, operation, old_id):
    return kind + "_" + hashlib.sha256(f"{operation}:{old_id}".encode()).hexdigest()[:56]


def plan_speaker_name_correction(*, operation_key: str, user_id: str, workspace_id: str,
                                 base_state_version: int, scopes: Sequence[SpeakerNameScope],
                                 evidence: Sequence[Evidence], objects: Sequence[Mapping],
                                 assertions: Sequence[Mapping], relations: Sequence[Mapping],
                                 now: datetime):
    """Create replacement versions and migrate references without adding facts.

    Callers supply the authorized current dependency closure, not model output.
    This operation only changes display fields and evidence references. Person
    identity merges use the separate controlled merge operation.
    """
    affected = {}
    creates, retracts, object_changes, assertion_changes, relation_changes = [], [], [], [], []
    remap = {}
    assertion_map = {}
    for item in evidence:
        if item.user_id != user_id or item.workspace_id != workspace_id:
            raise ValueError("Name correction evidence is outside the authorized scope")
        names = {}
        speaker_ref = item.speaker_id
        for scope in scopes:
            if not scope.new_name.strip() or any(not name for name in scope.old_names):
                raise ValueError("Name correction names cannot be empty")
            if scope.contains(item):
                if scope.speaker_ref and item.speaker_id == scope.original_speaker_ref:
                    speaker_ref = scope.speaker_ref
                for old in scope.old_names:
                    if old in names and names[old] != scope.new_name:
                        raise ValueError("Ambiguous name correction scope")
                    names[old] = scope.new_name
        if not names:
            continue
        affected[item.evidence_id] = names
        content = replace_names(item.content, names)
        if content == item.content and speaker_ref == item.speaker_id:
            continue
        new_id = _id("ev", operation_key, item.evidence_id)
        remap[item.evidence_id] = new_id
        corrected = Evidence.model_validate({
            **item.model_dump(), "evidence_id": new_id, "content": content,
            "speaker_id": speaker_ref,
            "content_hash": hashlib.sha256(content.encode()).hexdigest(), "created_at": now,
        })
        creates.append(EvidenceCreate(logical_ref=new_id, evidence=corrected))
        retracts.append(RetractionMutation(logical_ref="ret:" + item.evidence_id,
                        target_type="evidence", target_id=item.evidence_id,
                        reason="speaker_name_correction"))

    def names_for(refs):
        # A multi-source field is safe only if every reference agrees.
        maps = [affected.get(ref) for ref in refs]
        return maps[0] if maps and maps[0] and all(value == maps[0] for value in maps) else {}

    def refs_for(refs):
        return tuple(dict.fromkeys(remap.get(ref, ref) for ref in refs))

    def migrate_refs(value, key=""):
        if isinstance(value, dict):
            return {k: migrate_refs(v, k) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [migrate_refs(v, key) for v in value]
        if isinstance(value, str) and key in {"evidence_id", "evidence_ids", "completion_evidence_ids", "resolution_evidence_ids"}:
            return remap.get(value, value)
        return value

    for row in assertions:
        refs = tuple(row["evidence_ids"])
        payload = dict(row["payload"])
        value = correct_display_fields(payload["value"], names_for(refs), text_value=True)
        new_refs = refs_for(refs)
        if value == payload["value"] and new_refs == refs:
            continue
        payload["value"] = value
        new_id = _id("ast", operation_key, row["assertion_id"])
        assertion_map[row["assertion_id"]] = new_id
        assertion_changes.append(AssertionMutation(
            logical_ref=new_id, assertion_id=new_id, operation="create",
            evidence_ids=new_refs, payload=payload,
        ))
        retracts.append(RetractionMutation(logical_ref="ret:" + row["assertion_id"],
                        target_type="assertion", target_id=row["assertion_id"],
                        reason="speaker_name_correction"))
    for row in relations:
        refs = tuple(row["evidence_ids"])
        new_refs = refs_for(refs)
        if new_refs == refs:
            continue
        new_id = _id("rel", operation_key, row["relation_id"])
        relation_changes.append(RelationMutation(
            logical_ref=new_id, relation_id=new_id, operation="create",
            source_object_ref=row["source_object_id"], target_object_ref=row["target_object_id"],
            relation_type=row["relation_type"], evidence_ids=new_refs, payload=row["payload"],
        ))
        retracts.append(RetractionMutation(logical_ref="ret:" + row["relation_id"],
                        target_type="relation", target_id=row["relation_id"],
                        reason="speaker_name_correction"))
    for row in objects:
        refs = tuple(row["evidence_ids"])
        before = dict(row["payload"])
        names = names_for(refs)
        # A local transcript name must never rename the global People entity.
        if row["object_type"] == "entity":
            names = {}
        after = migrate_refs(correct_display_fields(before, names))
        new_refs = refs_for(refs)
        patch = {key: value for key, value in after.items()
                 if key in ObjectMutationPayload.model_fields and value != before.get(key)}
        if not patch and new_refs == refs:
            continue
        if row["object_type"] == "meeting":
            patch.update({key: before.get(key, before.get("attributes", {}).get(key))
                          for key in ("canonical_key", "external_memory_id")})
        object_changes.append(ObjectMutation(
            logical_ref="obj:" + row["object_id"], operation="update",
            object_type=row["object_type"], object_id=row["object_id"],
            expected_version=row["lock_version"], evidence_ids=new_refs, payload=patch,
        ))
    if not (creates or object_changes or assertion_changes or relation_changes):
        return None
    return MemoryPlanner().plan(MemoryChangeDraft(
        changeset_id=_id("chg", operation_key, "names"), user_id=user_id, workspace_id=workspace_id,
        source_ref=SourceRef(source_type="internal_correction", source_id=operation_key),
        base_state_version=base_state_version,
        expected_object_versions={row.object_id: row.expected_version for row in object_changes},
        evidence_creates=tuple(creates), object_mutations=tuple(object_changes),
        assertion_mutations=tuple(assertion_changes), relation_mutations=tuple(relation_changes),
        retractions=tuple(retracts), domain_events=(DomainEvent(
            event_type="memory.speaker_names_corrected", aggregate_ref=str(workspace_id),
            payload={"reason": "speaker_name_correction", "evidence_replacements": remap,
                     "assertion_replacements": assertion_map},
        ),),
    ))
