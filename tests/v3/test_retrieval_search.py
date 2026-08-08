import pytest

from mem0.v3.ports import SearchCandidate
from mem0.v3.retrieval import (
    fuse_identifiers,
    fuse_object_candidates,
    plan_assertion_search_document,
    plan_evidence_search_document,
    plan_object_search_document,
)


def test_candidate_fusion_prioritizes_exact_anchor_and_caps_semantic_score():
    objects = (
        {
            "object_id": "project:old",
            "title": "Apollo",
            "canonical_key": "project:apollo",
            "attributes": {"aliases": ["Project A"]},
        },
    )
    semantic = (
        SearchCandidate(object_id="project:new", score=0.95),
        SearchCandidate(object_id="project:old", score=0.70),
    )

    fused = fuse_object_candidates(
        query="Apollo",
        authoritative_objects=objects,
        semantic_candidates=semantic,
        limit=10,
    )

    assert [item.object_id for item in fused] == [
        "project:old",
        "project:new",
    ]
    assert fused[0].exact_match is True
    assert fused[0].score == 1.0
    assert fused[0].sources == frozenset({"authoritative", "semantic"})
    assert fused[1].score == 0.89


def test_identifier_fusion_favors_agreement_and_is_stable():
    assert fuse_identifiers(
        authoritative_ids=("evidence:2", "evidence:1"),
        semantic_ids=("evidence:3", "evidence:1"),
        limit=3,
    ) == ("evidence:1", "evidence:2", "evidence:3")


def test_candidate_fusion_rejects_non_positive_limit():
    with pytest.raises(ValueError, match="limit must be positive"):
        fuse_object_candidates(
            query="Apollo",
            authoritative_objects=(),
            semantic_candidates=(),
            limit=0,
        )


def test_object_search_document_uses_only_allowlisted_semantic_fields():
    document = plan_object_search_document(
        object_id="project:1",
        memory_object={
            "object_type": "project",
            "title": "Apollo",
            "description": "Launch program",
            "attributes": {
                "aliases": ["Project A"],
                "owner": "Alex",
                "private_internal_id": "secret-123",
                "provenance": {"raw": "do not index"},
            },
        },
    )

    assert document.document_kind == "object"
    assert document.document_id == "project:1"
    assert document.object_type == "project"
    assert document.search_text == "Apollo Launch program Project A Alex"
    assert "secret-123" not in document.search_text
    assert "do not index" not in document.search_text


def test_object_search_document_deduplicates_repeated_semantic_fields():
    document = plan_object_search_document(
        object_id="project:coffee",
        memory_object={
            "object_type": "project",
            "title": "咖啡项目",
            "description": "  咖啡项目  ",
            "attributes": {
                "canonical_label": "咖啡项目",
                "identity_aliases": ["咖啡项目", "精品咖啡"],
            },
        },
    )

    assert document.search_text == "咖啡项目 精品咖啡"


def test_evidence_and_assertion_search_documents_are_deterministic():
    evidence = plan_evidence_search_document(
        evidence_id="evidence:1",
        evidence={"content": "  Approved   for launch.  "},
    )
    assertion = plan_assertion_search_document(
        assertion_id="assertion:1",
        assertion={
            "predicate": "project.status",
            "value": {"state": "approved", "owner": "Alex"},
        },
    )

    assert evidence.search_text == "Approved for launch."
    assert assertion.search_text == (
        'project.status {"owner":"Alex","state":"approved"}'
    )


def test_empty_search_content_uses_kind_not_internal_identifier():
    document = plan_object_search_document(
        object_id="private-object-id", memory_object={}
    )

    assert document.search_text == "object"
    assert "private-object-id" not in document.search_text
