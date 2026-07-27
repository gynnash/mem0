"""Precision-first Session Topic Candidate resolution."""

from mem0.v3.extraction import SessionTopicCandidate
from mem0.v3.resolution.models import (
    TopicCandidateMatch,
    TopicResolutionDecision,
    TopicResolutionStatus,
)


class TopicResolver:
    def __init__(
        self,
        *,
        link_threshold: float = 0.92,
        margin_threshold: float = 0.15,
        resolver_version: str = "topic-resolver/v1",
    ) -> None:
        self._link_threshold = link_threshold
        self._margin_threshold = margin_threshold
        self._resolver_version = resolver_version

    def resolve(
        self,
        *,
        session_candidate: SessionTopicCandidate,
        matches: tuple[TopicCandidateMatch, ...],
        independent_evidence_count: int = 1,
    ) -> TopicResolutionDecision:
        ordered = tuple(
            sorted(matches, key=lambda item: (-item.score, item.topic_object_id))
        )
        eligible = tuple(
            item
            for item in ordered
            if item.scope_compatible
            and (
                item.explicit_name_match
                or item.project_anchor_match
                or item.object_anchor_match
            )
        )
        if eligible:
            top = eligible[0]
            second_score = ordered[1].score if len(ordered) > 1 else 0
            if top.score >= self._link_threshold and (
                len(ordered) == 1 or top.score - second_score >= self._margin_threshold
            ):
                return self._result(
                    session_candidate,
                    TopicResolutionStatus.LINKED,
                    ordered,
                    topic_object_id=top.topic_object_id,
                    confidence=top.score,
                )
            return self._result(
                session_candidate,
                TopicResolutionStatus.MULTI_CANDIDATE,
                ordered,
                confidence=top.score,
            )
        has_creation_anchor = bool(
            session_candidate.explicit_name
            or session_candidate.project_mentions
            or session_candidate.object_anchors
            or independent_evidence_count >= 2
        )
        if has_creation_anchor and session_candidate.confidence >= self._link_threshold:
            return self._result(
                session_candidate,
                TopicResolutionStatus.NEW_TOPIC,
                ordered,
                confidence=session_candidate.confidence,
            )
        return self._result(
            session_candidate,
            TopicResolutionStatus.UNRESOLVED,
            ordered,
            confidence=session_candidate.confidence,
        )

    def _result(
        self,
        session_candidate,
        status,
        matches,
        *,
        topic_object_id=None,
        confidence,
    ):
        return TopicResolutionDecision(
            session_candidate_id=session_candidate.candidate_id,
            decision=status,
            topic_object_id=topic_object_id,
            confidence=confidence,
            candidate_topic_ids=tuple(item.topic_object_id for item in matches),
            resolver_version=self._resolver_version,
            shadow_only=True,
        )

