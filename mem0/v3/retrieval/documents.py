"""Neutral search-document planning for memory domain records."""

from __future__ import annotations

import json
from typing import Any, Literal, Mapping, Sequence

from mem0.v3.contracts.base import FrozenContract, NonEmptyStr


MemorySearchDocumentKind = Literal["object", "evidence", "assertion"]

SEARCHABLE_OBJECT_ATTRIBUTE_KEYS = frozenset(
    {
        "aliases",
        "canonical_label",
        "deadline",
        "identity_aliases",
        "name",
        "object_mentions",
        "organization",
        "owner",
        "product",
        "project_mentions",
        "role",
        "status",
    }
)


class MemorySearchDocument(FrozenContract):
    """Semantic search content without storage, tenant, or model details."""

    document_kind: MemorySearchDocumentKind
    document_id: NonEmptyStr
    object_type: str = ""
    search_text: NonEmptyStr


def plan_object_search_document(
    *, object_id: str, memory_object: Mapping[str, Any]
) -> MemorySearchDocument:
    attributes = memory_object.get("attributes") or {}
    attribute_values = _flatten_searchable_attributes(attributes)
    return MemorySearchDocument(
        document_kind="object",
        document_id=object_id,
        object_type=str(memory_object.get("object_type") or ""),
        search_text=_search_text(
            (
                memory_object.get("title"),
                memory_object.get("description"),
                *attribute_values,
            ),
            fallback="object",
        ),
    )


def plan_evidence_search_document(
    *, evidence_id: str, evidence: Mapping[str, Any]
) -> MemorySearchDocument:
    return MemorySearchDocument(
        document_kind="evidence",
        document_id=evidence_id,
        search_text=_search_text(
            (evidence.get("content"),), fallback="evidence"
        ),
    )


def plan_assertion_search_document(
    *, assertion_id: str, assertion: Mapping[str, Any]
) -> MemorySearchDocument:
    value_text = json.dumps(
        assertion.get("value"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return MemorySearchDocument(
        document_kind="assertion",
        document_id=assertion_id,
        search_text=_search_text(
            (assertion.get("predicate"), value_text),
            fallback="assertion",
        ),
    )


def _flatten_searchable_attributes(
    attributes: Mapping[str, Any],
) -> tuple[str, ...]:
    values: list[str] = []
    for key in sorted(SEARCHABLE_OBJECT_ATTRIBUTE_KEYS):
        value = attributes.get(key)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            values.append(str(value))
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            values.extend(
                str(item)
                for item in value
                if isinstance(item, (str, int, float))
                and not isinstance(item, bool)
            )
    return tuple(values)


def _search_text(values: Sequence[Any], *, fallback: str) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for value in values:
        part = " ".join(str(value or "").split())
        if part and part not in seen:
            seen.add(part)
            parts.append(part)
    text = " ".join(parts)
    return text or fallback
