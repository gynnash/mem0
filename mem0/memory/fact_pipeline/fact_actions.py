import hashlib
import uuid
from copy import deepcopy
from datetime import datetime

import pytz

from mem0.memory.fact_pipeline.fact_ingestion import (
    normalize_applied_fact_ids,
    normalize_fact_metadata,
    normalize_related_memories,
)


def record_memory_history(
    db,
    memory_id,
    old_memory,
    new_memory,
    payload,
    event="UPDATE",
    operation_fact_id=None,
):
    history_id = None
    if operation_fact_id:
        history_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"mem0-history:{memory_id}:{event}:{operation_fact_id}",
            )
        )
    db.add_history(
        memory_id,
        old_memory,
        new_memory,
        event,
        created_at=payload.get("created_at"),
        updated_at=payload.get("updated_at"),
        actor_id=payload.get("actor_id"),
        role=payload.get("role"),
        history_id=history_id,
    )


def merge_memory_payload(existing_payload, new_payload, *, new_data=None):
    existing_payload = deepcopy(existing_payload or {})
    new_payload = normalize_fact_metadata(deepcopy(new_payload or {}))
    updated_metadata = deepcopy(existing_payload)
    canonical_memory_id = existing_payload.get("memory_id") or new_payload.get("memory_id")
    latest_memory_id = (
        new_payload.get("latest_memory_id")
        or new_payload.get("memory_id")
        or existing_payload.get("latest_memory_id")
        or canonical_memory_id
    )

    for key, value in new_payload.items():
        if key in {
            "data",
            "hash",
            "created_at",
            "updated_at",
            "changetime",
            "changeto",
            "memory_id",
            "latest_memory_id",
            "related_memories",
        }:
            continue
        updated_metadata[key] = value

    updated_metadata["memory_id"] = canonical_memory_id
    updated_metadata["latest_memory_id"] = latest_memory_id

    created_at = existing_payload.get("created_at") or new_payload.get("created_at")
    if created_at:
        updated_metadata["created_at"] = created_at

    if new_data is not None:
        updated_metadata["data"] = new_data
        updated_metadata["hash"] = hashlib.md5(new_data.encode()).hexdigest()

    updated_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()
    updated_metadata["changetime"] = updated_metadata["updated_at"]
    related_memories = normalize_related_memories(
        existing_payload.get("related_memories"),
        canonical_memory_id,
    )
    for related_memory in new_payload.get("related_memories") or []:
        related_memories = normalize_related_memories(related_memories, related_memory)
    updated_metadata["related_memories"] = normalize_related_memories(
        related_memories,
        latest_memory_id,
    )
    operation_fact_id = new_payload.get("fact_id")
    updated_metadata["applied_fact_ids"] = normalize_applied_fact_ids(
        existing_payload.get("applied_fact_ids"),
        operation_fact_id,
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
    operation_fact_id=None,
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
        related_memories = normalize_related_memories(
            existing_payload.get("related_memories"),
            existing_payload.get("memory_id"),
        )
        updated_metadata["related_memories"] = normalize_related_memories(
            related_memories,
            new_business_memory_id,
        )

    updated_metadata["applied_fact_ids"] = normalize_applied_fact_ids(
        existing_payload.get("applied_fact_ids"),
        operation_fact_id,
    )

    vector_store.update(vector_id=memory_id, vector=None, payload=updated_metadata)
    record_memory_history(
        db,
        memory_id,
        previous_memory,
        previous_memory,
        updated_metadata,
        operation_fact_id=operation_fact_id,
    )
    return updated_metadata


def update_memory_embedding(vector_store, db, memory_id, new_embedding, new_data, existing_payload, new_payload):
    previous_memory = existing_payload.get("data")
    updated_metadata = merge_memory_payload(existing_payload, new_payload, new_data=new_data)
    vector_store.update(vector_id=memory_id, vector=new_embedding, payload=updated_metadata)
    record_memory_history(
        db,
        memory_id,
        previous_memory,
        new_data,
        updated_metadata,
        operation_fact_id=new_payload.get("fact_id"),
    )
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
        related_memories = normalize_related_memories(
            existing_payload.get("related_memories"),
            existing_payload.get("memory_id"),
        )
        updated_metadata["related_memories"] = normalize_related_memories(
            related_memories,
            new_memory_id,
        )

    updated_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()
    updated_metadata["changetime"] = updated_metadata["updated_at"]
    if change_source_fact_id:
        updated_metadata["changeto"] = change_source_fact_id
    updated_metadata["applied_fact_ids"] = normalize_applied_fact_ids(
        existing_payload.get("applied_fact_ids"),
        change_source_fact_id,
    )

    vector_store.update(vector_id=memory_id, vector=None, payload=updated_metadata)
    record_memory_history(
        db,
        memory_id,
        previous_memory,
        previous_memory,
        updated_metadata,
        operation_fact_id=change_source_fact_id,
    )
    return updated_metadata
