import uuid
from copy import deepcopy


def normalize_applied_fact_ids(values, fact_id=None):
    applied_fact_ids = values if isinstance(values, list) else []
    if fact_id:
        applied_fact_ids = [*applied_fact_ids, fact_id]
    return list(dict.fromkeys(str(value) for value in applied_fact_ids if value not in (None, "")))


def normalize_related_memories(values, business_memory_id=None):
    related_memories = []
    if values is not None:
        if isinstance(values, list):
            related_memories.extend(values)
        else:
            related_memories.append(values)
    if business_memory_id:
        related_memories.append(business_memory_id)

    deduped_related_memories = []
    seen = set()
    for related_memory in related_memories:
        if related_memory in (None, ""):
            continue
        dedupe_key = str(related_memory)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped_related_memories.append(related_memory)

    return deduped_related_memories


def normalize_fact_metadata(metadata):
    metadata = deepcopy(metadata or {})
    metadata.setdefault("fact_id", str(uuid.uuid4()))
    metadata["applied_fact_ids"] = normalize_applied_fact_ids(
        metadata.get("applied_fact_ids"),
        metadata.get("fact_id"),
    )
    if not metadata.get("latest_memory_id"):
        metadata["latest_memory_id"] = metadata.get("memory_id")
    metadata["related_memories"] = normalize_related_memories(
        metadata.get("related_memories"),
        metadata.get("memory_id"),
    )
    return metadata


def build_add_with_attr_result(memory_id, memory_text, metadata, event, reason=None, **extra_fields):
    result = {
        "id": memory_id,
        "fact_id": metadata.get("fact_id"),
        "memory_id": metadata.get("memory_id"),
        "latest_memory_id": metadata.get("latest_memory_id"),
        "memory_type": metadata.get("memory_type"),
        "persisted_at": metadata.get("persisted_at"),
        "memory": memory_text,
        "event": event,
    }
    if reason:
        result["reason"] = reason
    result.update(extra_fields)
    return result
