import importlib.metadata

import pytest

pytest.importorskip("posthog")

importlib.metadata.version = lambda _: "0.0.0"

from mem0.memory.main import Memory
from mem0.memory.fact_pipeline import (
    passes_type_specific_candidate_filter,
    normalize_related_memories,
    select_primary_relationships,
    update_memory_embedding,
)


class _FakeVectorStore:
    def __init__(self):
        self.update_calls = []

    def update(self, **kwargs):
        self.update_calls.append(kwargs)


class _FakeDB:
    def __init__(self):
        self.history_calls = []

    def add_history(self, *args, **kwargs):
        self.history_calls.append((args, kwargs))


def _make_memory():
    memory = Memory.__new__(Memory)
    memory.vector_store = _FakeVectorStore()
    memory.db = _FakeDB()
    return memory


def test_normalize_related_memories_deduplicates_and_includes_memory_id():
    related_memories = normalize_related_memories(
        ["memory-0", "memory-1", "memory-2"],
        "memory-1",
    )

    assert related_memories == ["memory-0", "memory-1", "memory-2"]


def test_update_memory_embedding_preserves_created_at_and_records_history():
    memory = _make_memory()

    updated = update_memory_embedding(
        memory.vector_store,
        memory.db,
        "internal-1",
        [0.1, 0.2],
        "new fact",
        {
            "fact_id": "fact-old",
            "memory_id": "memory-1",
            "data": "old fact",
            "created_at": "2024-01-01T00:00:00Z",
            "related_memories": ["memory-1"],
        },
        {
            "fact_id": "fact-new",
            "memory_id": "memory-2",
            "importance": "high",
        },
    )

    assert updated["created_at"] == "2024-01-01T00:00:00Z"
    assert updated["fact_id"] == "fact-new"
    assert updated["memory_id"] == "memory-2"
    assert updated["related_memories"] == ["memory-1", "memory-2"]
    assert memory.vector_store.update_calls[0]["vector_id"] == "internal-1"
    assert memory.db.history_calls[0][0][:4] == ("internal-1", "old fact", "new fact", "UPDATE")


def test_type_specific_candidate_filter_is_more_permissive_for_related_topic_candidates():
    assert passes_type_specific_candidate_filter(
        "Discuss microservices deployment plan",
        {"memory_type": "topic", "category": ["engineering"]},
        {
            "memory_type": "topic",
            "data": "Microservices rollout strategy for deployment",
            "category": ["engineering"],
        },
        0.63,
    )

    assert not passes_type_specific_candidate_filter(
        "Discuss microservices deployment plan",
        {"memory_type": "topic", "category": ["engineering"]},
        {
            "memory_type": "topic",
            "data": "Brainstorm marketing event ideas",
            "category": ["marketing"],
        },
        0.63,
    )


def test_select_primary_relationships_uses_priority_and_score():
    selected = select_primary_relationships(
        [
            {"old_fact_id": "fact-1", "relationship": "IDENTICAL"},
            {"old_fact_id": "fact-2", "relationship": "MORE_COMPLETE"},
            {"old_fact_id": "fact-3", "relationship": "PARAPHRASE"},
        ],
        {
            "fact-1": {"score": 0.91},
            "fact-2": {"score": 0.70},
            "fact-3": {"score": 0.99},
        },
    )

    assert selected == [{"old_fact_id": "fact-2", "relationship": "MORE_COMPLETE"}]
