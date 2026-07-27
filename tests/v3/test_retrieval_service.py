from datetime import datetime, timezone

from mem0.v3.retrieval import MemoryQueryService


def test_open_loop_policy_excludes_resolved_and_completed_objects():
    objects = (
        {
            "object_id": "commitment:open",
            "workflow_status": "in_progress",
            "attributes": {"fulfillment_status": "open"},
        },
        {
            "object_id": "commitment:done",
            "workflow_status": "completed",
            "attributes": {"fulfillment_status": "completed"},
        },
        {
            "object_id": "issue:resolved",
            "workflow_status": "accepted",
            "attributes": {"resolution_status": "resolved"},
        },
    )

    selected = MemoryQueryService().select_open_loops(objects)

    assert tuple(item["object_id"] for item in selected) == (
        "commitment:open",
    )


def test_evidence_packing_is_bounded_without_owning_storage_or_auth():
    packed, truncated = MemoryQueryService().pack_evidence(
        (
            {"evidence_id": "ev_1", "content": "abcdef"},
            {"evidence_id": "ev_2", "content": "ghij"},
        ),
        max_total_chars=8,
    )

    assert [item["content"] for item in packed] == ["abcdef", "gh"]
    assert packed[1]["content_truncated"] is True
    assert truncated is True


def test_state_analysis_is_precision_first_and_storage_independent():
    service = MemoryQueryService()
    objects = (
        {
            "object_id": "commitment:open",
            "object_type": "commitment",
            "workflow_status": "in_progress",
            "attributes": {
                "fulfillment_status": "open",
                "due_at": "2026-07-20T09:00:00+00:00",
            },
        },
        {
            "object_id": "commitment:unknown",
            "object_type": "commitment",
            "workflow_status": "in_progress",
            "attributes": {
                "fulfillment_status": "unknown_no_evidence",
                "due_at": "2026-07-20T09:00:00+00:00",
            },
        },
        {
            "object_id": "task:unowned",
            "object_type": "task",
            "workflow_status": "accepted",
            "attributes": {},
        },
        {
            "object_id": "issue:repeated",
            "object_type": "issue",
            "attributes": {
                "occurrence_count": 2,
                "resolution_status": "open",
            },
        },
    )

    assert tuple(
        item["object_id"]
        for item in service.select_overdue_commitments(
            objects,
            now=datetime(2026, 7, 26, tzinfo=timezone.utc),
        )
    ) == ("commitment:open",)
    assert tuple(
        item["object_id"] for item in service.select_unowned_tasks(objects)
    ) == ("task:unowned",)
    assert tuple(
        item["object_id"] for item in service.select_repeated_issues(objects)
    ) == ("issue:repeated",)


def test_version_comparison_ignores_storage_watermarks():
    comparison = MemoryQueryService().compare_versions(
        {
            "object_id": "decision:1",
            "version": 1,
            "state_version": 10,
            "title": "Launch Monday",
        },
        {
            "object_id": "decision:1",
            "version": 2,
            "state_version": 20,
            "title": "Launch Friday",
        },
    )

    assert comparison["changed_fields"] == ("title",)
    assert comparison["changes"]["title"] == {
        "before": "Launch Monday",
        "after": "Launch Friday",
    }
