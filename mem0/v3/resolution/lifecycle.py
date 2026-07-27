"""Deterministic lifecycle selection after a candidate identity is verified."""

from datetime import date, datetime
from enum import Enum
from typing import Any, Mapping, Optional

from mem0.v3.domain import LifecycleOperation
from mem0.v3.resolution.models import ObjectLinkCandidate


class LifecycleResolver:
    _OBSERVATION_FIELDS = {
        "field_provenance",
        "valid_from",
        "confidence",
        "effective_from",
        "committed_at",
    }

    def protect_user_locked_fields(
        self,
        *,
        existing: Optional[ObjectLinkCandidate],
        proposed: Mapping[str, Any],
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        if existing is None:
            return dict(proposed), ()
        locked = {
            str(item.get("field_name")): item
            for item in existing.field_provenance
            if item.get("user_locked") is True and item.get("field_name")
        }
        if not locked:
            return dict(proposed), ()
        current = {
            "title": existing.title,
            "workflow_status": existing.workflow_status,
            **existing.attributes,
        }
        cleaned = dict(proposed)
        warnings = []
        for field_name in locked:
            if field_name in cleaned and cleaned[field_name] != current.get(field_name):
                cleaned.pop(field_name)
                warnings.append(f"user_locked_field_preserved:{field_name}")
        proposed_provenance = tuple(cleaned.get("field_provenance") or ())
        cleaned["field_provenance"] = tuple(
            item
            for item in proposed_provenance
            if getattr(item, "field_name", None) not in locked
            and not (
                isinstance(item, Mapping)
                and str(item.get("field_name")) in locked
            )
        ) + tuple(locked.values())
        return cleaned, tuple(warnings)

    def resolve(
        self,
        *,
        existing: Optional[ObjectLinkCandidate],
        proposed: Mapping[str, Any],
        contradictory: bool = False,
        resolved: bool = False,
        reopen: bool = False,
        supersedes: bool = False,
    ) -> LifecycleOperation:
        if existing is None:
            return LifecycleOperation.CREATE
        if contradictory:
            return LifecycleOperation.CONTRADICT
        if supersedes:
            return LifecycleOperation.SUPERSEDE
        if reopen:
            return LifecycleOperation.REOPEN
        if resolved:
            return LifecycleOperation.RESOLVE
        current = dict(existing.current_state) or {
            "canonical_key": existing.canonical_key,
            "title": existing.title,
            "workflow_status": existing.workflow_status,
            "attributes": existing.attributes,
        }
        current_items = self._semantic_items(current)
        proposed_items = self._semantic_items(proposed)
        if all(
            key in current_items and current_items[key] == value
            for key, value in proposed_items.items()
        ):
            return LifecycleOperation.CONFIRM
        if any(key not in current_items for key in proposed_items):
            return LifecycleOperation.ENRICH
        return LifecycleOperation.UPDATE

    @classmethod
    def _semantic_items(
        cls,
        value: Mapping[str, Any],
        *,
        prefix: str = "",
    ) -> dict[str, Any]:
        items = {}
        for key, item in value.items():
            key = str(key)
            if key in cls._OBSERVATION_FIELDS or item is None:
                continue
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(item, Mapping):
                nested = cls._semantic_items(item, prefix=path)
                if nested:
                    items.update(nested)
                elif item == {}:
                    items[path] = {}
                continue
            items[path] = cls._comparable(item)
        return items

    @classmethod
    def _comparable(cls, value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, Mapping):
            return {
                str(key): cls._comparable(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return tuple(cls._comparable(item) for item in value)
        return value
