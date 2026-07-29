"""Storage-independent candidate fusion for memory recall."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from pydantic import Field

from mem0.v3.contracts.base import FrozenContract, NonEmptyStr
from mem0.v3.ports import SearchCandidate


class FusedObjectCandidate(FrozenContract):
    object_id: NonEmptyStr
    score: float = Field(ge=0, le=1)
    exact_match: bool
    sources: frozenset[str]


def fuse_object_candidates(
    *,
    query: str,
    authoritative_objects: Sequence[Mapping[str, Any]],
    semantic_candidates: Sequence[SearchCandidate],
    limit: int,
) -> tuple[FusedObjectCandidate, ...]:
    """Fuse authoritative lexical results with semantic recall candidates.

    Exact title, canonical-key, and alias anchors are deterministic. All
    other scores remain below the V3 semantic auto-link threshold so recall
    cannot silently become identity resolution.
    """

    if limit < 1:
        raise ValueError("candidate fusion limit must be positive")
    normalized_query = _normalized(query)
    authoritative_rank = {
        str(item["object_id"]): index
        for index, item in enumerate(authoritative_objects)
    }
    semantic_rank = {
        str(candidate.object_id): index
        for index, candidate in enumerate(semantic_candidates)
        if candidate.object_id
    }
    authoritative_by_id = {
        str(item["object_id"]): item for item in authoritative_objects
    }
    semantic_by_id = {
        str(candidate.object_id): candidate
        for candidate in semantic_candidates
        if candidate.object_id
    }
    fused = []
    for object_id in dict.fromkeys((*semantic_rank, *authoritative_rank)):
        sources = frozenset(
            source
            for source, values in (
                ("semantic", semantic_rank),
                ("authoritative", authoritative_rank),
            )
            if object_id in values
        )
        memory_object = authoritative_by_id.get(object_id)
        exact = bool(
            memory_object
            and _object_exact_match(memory_object, normalized_query)
        )
        semantic_score = (
            max(0.0, min(1.0, float(semantic_by_id[object_id].score)))
            if object_id in semantic_by_id
            else 0.0
        )
        score = (
            1.0
            if exact
            else max(0.89 if memory_object else 0.0, semantic_score)
        )
        if not exact:
            score = min(0.89, score)
        fused.append(
            FusedObjectCandidate(
                object_id=object_id,
                score=score,
                exact_match=exact,
                sources=sources,
            )
        )
    fused.sort(
        key=lambda item: (
            0 if item.exact_match else 1,
            0 if len(item.sources) > 1 else 1,
            -item.score,
            min(
                authoritative_rank.get(item.object_id, 1_000_000),
                semantic_rank.get(item.object_id, 1_000_000),
            ),
            item.object_id,
        )
    )
    return tuple(fused[:limit])


def fuse_identifiers(
    *,
    authoritative_ids: Sequence[str],
    semantic_ids: Sequence[str],
    limit: int,
) -> tuple[str, ...]:
    """Deduplicate identifiers while favoring agreement between recall lanes."""

    if limit < 1:
        raise ValueError("identifier fusion limit must be positive")
    authoritative_rank = {
        str(value): index for index, value in enumerate(authoritative_ids)
    }
    semantic_rank = {
        str(value): index for index, value in enumerate(semantic_ids)
    }
    identifiers = tuple(
        dict.fromkeys((*semantic_rank, *authoritative_rank))
    )
    return tuple(
        sorted(
            identifiers,
            key=lambda value: (
                0
                if value in authoritative_rank and value in semantic_rank
                else 1,
                min(
                    authoritative_rank.get(value, 1_000_000),
                    semantic_rank.get(value, 1_000_000),
                ),
                value,
            ),
        )[:limit]
    )


def _object_exact_match(
    memory_object: Mapping[str, Any], normalized_query: str
) -> bool:
    if not normalized_query:
        return False
    values = {
        _normalized(memory_object.get("title")),
        _normalized(memory_object.get("canonical_key")),
    }
    attributes = memory_object.get("attributes") or {}
    for key in ("aliases", "identity_aliases", "object_mentions"):
        value = attributes.get(key) or ()
        if isinstance(value, str):
            values.add(_normalized(value))
        else:
            values.update(_normalized(item) for item in value)
    return normalized_query in values


def _normalized(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())
