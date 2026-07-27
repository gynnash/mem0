"""Storage-independent current-state filtering and evidence packing."""

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence


class MemoryQueryService:
    OPEN_LOOP_OBJECT_TYPES = ("commitment", "issue", "task", "goal")
    PRIOR_CONSTRAINT_OBJECT_TYPES = ("decision", "commitment", "issue")

    @staticmethod
    def compare_versions(
        before: Mapping[str, Any],
        after: Mapping[str, Any],
    ) -> dict[str, Any]:
        ignored = {"version", "state_version", "lock_version"}
        keys = sorted((set(before) | set(after)).difference(ignored))
        changes = {
            key: {"before": before.get(key), "after": after.get(key)}
            for key in keys
            if before.get(key) != after.get(key)
        }
        return {
            "object_id": str(after.get("object_id") or before.get("object_id") or ""),
            "from_version": before.get("version"),
            "to_version": after.get("version"),
            "changes": changes,
            "changed_fields": tuple(changes),
        }

    @staticmethod
    def select_unresolved_conflicts(
        objects: Sequence[Mapping[str, Any]],
    ) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            item
            for item in objects
            if item.get("validity") == "contradicted"
            or bool(
                (item.get("attributes") or {}).get("has_unresolved_conflict")
                or (item.get("attributes") or {}).get("has_active_conflict")
            )
        )

    @staticmethod
    def select_stalled_goals(
        objects: Sequence[Mapping[str, Any]],
        *,
        now: datetime,
        stalled_after_days: int,
    ) -> tuple[Mapping[str, Any], ...]:
        threshold = _aware(now) - timedelta(days=stalled_after_days)
        return tuple(
            item
            for item in objects
            if item.get("object_type") == "goal"
            and item.get("workflow_status") not in {"completed", "cancelled"}
            and (
                changed_at := _aware(
                    (item.get("attributes") or {}).get("last_changed_at")
                    or item.get("valid_from")
                )
            )
            is not None
            and changed_at <= threshold
        )

    @staticmethod
    def select_overdue_commitments(
        objects: Sequence[Mapping[str, Any]],
        *,
        now: datetime,
    ) -> tuple[Mapping[str, Any], ...]:
        now = _aware(now)
        return tuple(
            item
            for item in objects
            if item.get("object_type") == "commitment"
            and (item.get("attributes") or {}).get("fulfillment_status") == "open"
            and item.get("workflow_status") not in {
                "completed",
                "cancelled",
                "resolved",
            }
            and (
                due_at := _aware((item.get("attributes") or {}).get("due_at"))
            )
            is not None
            and due_at < now
        )

    @staticmethod
    def select_unowned_tasks(
        objects: Sequence[Mapping[str, Any]],
    ) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            item
            for item in objects
            if item.get("object_type") == "task"
            and not (
                (item.get("attributes") or {}).get("owner")
                or (item.get("attributes") or {}).get("owner_entity_id")
            )
            and item.get("workflow_status") not in {"completed", "cancelled"}
        )

    @staticmethod
    def select_repeated_issues(
        objects: Sequence[Mapping[str, Any]],
    ) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            item
            for item in objects
            if item.get("object_type") == "issue"
            and int((item.get("attributes") or {}).get("occurrence_count") or 0)
            >= 2
            and (item.get("attributes") or {}).get("resolution_status")
            not in {"resolved", "completed"}
        )

    @staticmethod
    def select_decision_changes(
        objects: Sequence[Mapping[str, Any]],
    ) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            item
            for item in objects
            if item.get("object_type") == "decision"
            and bool(
                (item.get("attributes") or {}).get("supersedes_decision_id")
                or (item.get("attributes") or {}).get("decision_changed")
            )
        )

    @staticmethod
    def select_cross_project_dependencies(
        objects: Sequence[Mapping[str, Any]],
        *,
        project_id: str | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        selected = []
        for item in objects:
            attributes = item.get("attributes") or {}
            source_project = item.get("primary_project_id")
            dependent_projects = tuple(
                str(value)
                for value in attributes.get("dependency_project_ids") or ()
                if value
            )
            cross_project = tuple(
                value for value in dependent_projects if value != source_project
            )
            if not cross_project:
                continue
            if project_id and project_id not in {
                str(source_project or ""),
                *cross_project,
            }:
                continue
            selected.append(item)
        return tuple(selected)

    @staticmethod
    def select_unsupported_beliefs(
        assertions: Sequence[Mapping[str, Any]],
    ) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            item
            for item in assertions
            if item.get("epistemic_type") in {"inferred", "hypothesis"}
            and (
                not item.get("evidence_ids")
                or float(item.get("confidence") or 0) < 0.5
            )
        )

    @staticmethod
    def select_open_loops(
        objects: Sequence[Mapping[str, Any]],
    ) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            item
            for item in objects
            if item.get("workflow_status") not in {"completed", "cancelled"}
            and (item.get("attributes") or {}).get("fulfillment_status")
            != "completed"
            and (item.get("attributes") or {}).get("resolution_status")
            not in {"resolved", "completed"}
        )

    @staticmethod
    def pack_evidence(
        evidence: Sequence[Mapping[str, Any]],
        *,
        max_total_chars: int,
    ) -> tuple[tuple[dict[str, Any], ...], bool]:
        remaining = max_total_chars
        packed = []
        truncated = False
        for record in evidence:
            if remaining <= 0:
                truncated = True
                break
            content = str(record.get("content") or "")
            visible_content = content[:remaining]
            remaining -= len(visible_content)
            item = {
                **dict(record),
                "content": visible_content,
                "content_truncated": len(visible_content) < len(content),
            }
            truncated = truncated or bool(item["content_truncated"])
            packed.append(item)
        if len(packed) < len(evidence):
            truncated = True
        return tuple(packed), truncated


def _aware(value):
    if value in (None, ""):
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
