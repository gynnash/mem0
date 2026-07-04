import hashlib
from copy import deepcopy
from datetime import datetime

import pytz

from mem0.memory.fact_pipeline.fact_ingestion import normalize_fact_metadata, normalize_related_memories


def record_memory_history(db, memory_id, old_memory, new_memory, payload, event="UPDATE"):
    db.add_history(
        memory_id,
        old_memory,
        new_memory,
        event,
        created_at=payload.get("created_at"),
        updated_at=payload.get("updated_at"),
        actor_id=payload.get("actor_id"),
        role=payload.get("role"),
    )


def merge_memory_payload(existing_payload, new_payload, *, new_data=None):
    existing_payload = deepcopy(existing_payload or {})
    new_payload = normalize_fact_metadata(deepcopy(new_payload or {}))
    updated_metadata = deepcopy(existing_payload)

    for key, value in new_payload.items():
        if key in {"data", "hash", "created_at", "updated_at", "changetime", "changeto"}:
            continue
        updated_metadata[key] = value

    created_at = existing_payload.get("created_at") or new_payload.get("created_at")
    if created_at:
        updated_metadata["created_at"] = created_at

    if new_data is not None:
        updated_metadata["data"] = new_data
        updated_metadata["hash"] = hashlib.md5(new_data.encode()).hexdigest()

    updated_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()
    updated_metadata["changetime"] = updated_metadata["updated_at"]
    updated_metadata["related_memories"] = normalize_related_memories(
        updated_metadata.get("related_memories"),
        updated_metadata.get("memory_id"),
    )
    return updated_metadata


def update_memory_weight(
    vector_store,
    db,
    memory_id,
    new_weight,
    existing_payload,
    new_fact_id=None,
    new_business_memory_id=None,
    new_persisted_at=None,
):
    updated_metadata = deepcopy(existing_payload)
    previous_memory = updated_metadata.get("data")
    updated_metadata["weight"] = new_weight
    updated_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()
    updated_metadata["changetime"] = updated_metadata["updated_at"]

    if new_persisted_at is not None:
        updated_metadata["persisted_at"] = new_persisted_at

    if new_fact_id and updated_metadata.get("fact_id") != new_fact_id:
        updated_metadata["changeto"] = new_fact_id

    if new_business_memory_id:
        updated_metadata["latest_memory_id"] = new_business_memory_id
        updated_metadata["related_memories"] = normalize_related_memories(
            existing_payload.get("related_memories"),
            new_business_memory_id,
        )

    vector_store.update(vector_id=memory_id, vector=None, payload=updated_metadata)
    record_memory_history(db, memory_id, previous_memory, previous_memory, updated_metadata)
    return updated_metadata


def update_memory_embedding(vector_store, db, memory_id, new_embedding, new_data, existing_payload, new_payload):
    previous_memory = existing_payload.get("data")
    updated_metadata = merge_memory_payload(existing_payload, new_payload, new_data=new_data)
    vector_store.update(vector_id=memory_id, vector=new_embedding, payload=updated_metadata)
    record_memory_history(db, memory_id, previous_memory, new_data, updated_metadata)
    return updated_metadata


def update_memory_status(
    vector_store,
    db,
    memory_id,
    new_status,
    existing_payload,
    change_source_fact_id=None,
    new_memory_id=None,
    new_persisted_at=None,
):
    updated_metadata = deepcopy(existing_payload)
    previous_memory = updated_metadata.get("data")
    updated_metadata["status"] = new_status

    if new_persisted_at is not None:
        updated_metadata["persisted_at"] = new_persisted_at

    if new_memory_id:
        updated_metadata["latest_memory_id"] = new_memory_id
        updated_metadata["related_memories"] = normalize_related_memories(
            existing_payload.get("related_memories"),
            new_memory_id,
        )

    updated_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()
    updated_metadata["changetime"] = updated_metadata["updated_at"]
    if change_source_fact_id:
        updated_metadata["changeto"] = change_source_fact_id

    vector_store.update(vector_id=memory_id, vector=None, payload=updated_metadata)
    record_memory_history(db, memory_id, previous_memory, previous_memory, updated_metadata)
    return updated_metadata
