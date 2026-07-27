"""Precision-first task/object to project resolution."""

from mem0.v3.resolution.models import (
    ProjectAnchorType,
    ProjectCandidate,
    ProjectLink,
    ProjectLinkDecision,
    ProjectLinkStatus,
)


_RELIABLE_ANCHORS = {
    ProjectAnchorType.EXPLICIT_PROJECT_MENTION,
    ProjectAnchorType.LINKED_OBJECT,
    ProjectAnchorType.LOCAL_UNIQUE_ANCHOR,
}


class ProjectResolver:
    def __init__(
        self,
        *,
        link_threshold: float = 0.90,
        margin_threshold: float = 0.15,
        resolver_version: str = "project-resolver/v1",
    ) -> None:
        self._link_threshold = link_threshold
        self._margin_threshold = margin_threshold
        self._resolver_version = resolver_version

    def resolve(
        self, *, task_object_ref: str, candidates: tuple[ProjectCandidate, ...]
    ) -> ProjectLinkDecision:
        ordered = tuple(
            sorted(candidates, key=lambda item: (-item.score, item.project_object_id))
        )
        if not ordered:
            return self._empty(task_object_ref, ProjectLinkStatus.UNRESOLVED)
        reliable = tuple(
            item
            for item in ordered
            if set(item.anchor_types).intersection(_RELIABLE_ANCHORS)
        )
        if not reliable:
            return self._decision(
                task_object_ref, ProjectLinkStatus.UNRESOLVED, ordered, ()
            )
        high_confidence = tuple(
            item for item in reliable if item.score >= self._link_threshold
        )
        if len(high_confidence) > 1 and self._have_independent_anchors(
            high_confidence
        ):
            return self._decision(
                task_object_ref,
                ProjectLinkStatus.MULTI_LINKED,
                ordered,
                high_confidence,
            )
        top = reliable[0]
        second_score = ordered[1].score if len(ordered) > 1 else 0.0
        margin = max(0.0, top.score - second_score)
        if top.score < self._link_threshold:
            return self._decision(
                task_object_ref, ProjectLinkStatus.MULTI_CANDIDATE, ordered, ()
            )
        if len(ordered) > 1 and margin < self._margin_threshold:
            return self._decision(
                task_object_ref, ProjectLinkStatus.MULTI_CANDIDATE, ordered, ()
            )
        return self._decision(
            task_object_ref, ProjectLinkStatus.LINKED, ordered, (top,)
        )

    @staticmethod
    def _have_independent_anchors(
        candidates: tuple[ProjectCandidate, ...],
    ) -> bool:
        seen = set()
        for candidate in candidates:
            refs = set(candidate.anchor_refs)
            if not refs or refs.intersection(seen):
                return False
            seen.update(refs)
        return True

    def _empty(self, task_object_ref: str, status: ProjectLinkStatus):
        return ProjectLinkDecision(
            task_object_ref=task_object_ref,
            decision=status,
            confidence=0,
            margin_to_second_candidate=0,
            resolver_version=self._resolver_version,
        )

    def _decision(self, task_object_ref, status, candidates, linked):
        top_score = candidates[0].score if candidates else 0
        second_score = candidates[1].score if len(candidates) > 1 else 0
        anchors = tuple(
            dict.fromkeys(anchor for candidate in linked for anchor in candidate.anchor_types)
        )
        evidence_ids = tuple(
            dict.fromkeys(value for candidate in linked for value in candidate.evidence_ids)
        )
        return ProjectLinkDecision(
            task_object_ref=task_object_ref,
            decision=status,
            primary_project_object_id=(
                linked[0].project_object_id if linked else None
            ),
            confidence=top_score,
            margin_to_second_candidate=max(0, top_score - second_score),
            anchor_types=anchors,
            evidence_ids=evidence_ids,
            candidate_project_ids=tuple(
                item.project_object_id for item in candidates
            ),
            project_links=tuple(
                ProjectLink(
                    project_object_id=item.project_object_id,
                    role="primary" if index == 0 else "secondary",
                    confidence=item.score,
                )
                for index, item in enumerate(linked)
            ),
            resolver_version=self._resolver_version,
        )
