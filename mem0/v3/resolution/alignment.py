"""Global alignment that emits a domain draft, never physical write commands."""

import hashlib
from typing import Dict, Optional, Tuple

from mem0.v3.contracts import (
    AssertionMutation,
    DomainEvent,
    EvidenceCreate,
    ObjectMutation,
    RelationMutation,
    SourceRef,
    ValidatedMemoryChangeSet,
)
from mem0.v3.domain import (
    EpistemicType,
    Evidence,
    FieldProvenance,
    FulfillmentStatus,
    LifecycleOperation,
    MemoryObjectType,
    Polarity,
)
from mem0.v3.extraction import (
    ClaimType,
    ClaimLifecycleSignal,
    EvidenceSpan,
    LocalExtractionResult,
    MeetingExtractionInput,
    SessionTopicCandidate,
)
from mem0.v3.planner import MemoryChangeDraft, MemoryPlanner
from mem0.v3.resolution.lifecycle import LifecycleResolver
from mem0.v3.resolution.entity import EntityResolver
from mem0.v3.resolution.meeting import MeetingResolver
from mem0.v3.resolution.models import (
    AlignmentContext,
    ObjectLinkCandidate,
    ProjectLinkStatus,
    TopicResolutionStatus,
)
from mem0.v3.resolution.project import ProjectResolver
from mem0.v3.resolution.topic import TopicResolver


_CLAIM_OBJECT_TYPES = {
    ClaimType.DECISION: MemoryObjectType.DECISION,
    ClaimType.COMMITMENT: MemoryObjectType.COMMITMENT,
    ClaimType.OBJECTION: MemoryObjectType.ISSUE,
    ClaimType.BLOCKER: MemoryObjectType.ISSUE,
    ClaimType.TASK: MemoryObjectType.TASK,
    ClaimType.GOAL: MemoryObjectType.GOAL,
    ClaimType.PREFERENCE: MemoryObjectType.PREFERENCE,
}
_RELIABLE_OBJECT_ANCHORS = {
    "explicit_reference",
    "external_binding",
    "unique_alias",
    "linked_object",
}


def _stable_token(*values: object, length: int = 24) -> str:
    material = "\x1f".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:length]


def _normalized(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


class GlobalAlignmentService:
    """Convert validated local extraction into a validated semantic changeset."""

    def __init__(
        self,
        *,
        planner: Optional[MemoryPlanner] = None,
        meeting_resolver: Optional[MeetingResolver] = None,
        project_resolver: Optional[ProjectResolver] = None,
        topic_resolver: Optional[TopicResolver] = None,
        lifecycle_resolver: Optional[LifecycleResolver] = None,
        entity_resolver: Optional[EntityResolver] = None,
    ) -> None:
        self._planner = planner or MemoryPlanner()
        self._meeting_resolver = meeting_resolver or MeetingResolver()
        self._project_resolver = project_resolver or ProjectResolver()
        self._topic_resolver = topic_resolver or TopicResolver()
        self._lifecycle_resolver = lifecycle_resolver or LifecycleResolver()
        self._entity_resolver = entity_resolver or EntityResolver()

    def plan(
        self,
        *,
        source: MeetingExtractionInput,
        extraction: LocalExtractionResult,
        context: AlignmentContext,
    ) -> ValidatedMemoryChangeSet:
        evidence_creates, evidence_by_span = self._build_evidence(source, extraction)
        all_evidence_refs = tuple(item.logical_ref for item in evidence_creates)
        meeting_evidence = all_evidence_refs or self._fallback_evidence(
            source, evidence_creates, evidence_by_span
        )
        meeting_operation, meeting_id, meeting_version = self._meeting_resolver.resolve(
            memory_id=source.memory_id,
            transcript_version=source.transcript_version,
            transcript_content_hash=source.transcript_content_hash,
            existing=context.meeting_object,
        )
        meeting_ref = f"meeting:{source.memory_id}"
        meeting_mutation = ObjectMutation(
            logical_ref=meeting_ref,
            operation=meeting_operation,
            object_type=MemoryObjectType.MEETING,
            object_id=meeting_id,
            expected_version=meeting_version,
            evidence_ids=meeting_evidence,
            payload={
                "external_memory_id": source.memory_id,
                "canonical_key": meeting_ref,
                "title": source.title,
                "valid_from": source.started_at,
                "started_at": source.started_at,
                "ended_at": source.ended_at,
                "transcript_version": source.transcript_version,
                "participant_refs": source.participant_refs,
                "processing_status": "extracted",
                "confidence": 1,
                "attributes": {
                    "transcript_content_hash": source.transcript_content_hash,
                },
                "field_provenance": (
                    FieldProvenance(
                        field_name="title", evidence_ids=meeting_evidence
                    ),
                ),
            },
        )
        object_mutations = [meeting_mutation]
        assertion_mutations = []
        relation_mutations = []
        domain_events = []
        alignment_warnings = list(extraction.warnings)
        participant_entity_refs = {}
        project_refs_by_mention = {}

        for participant_ref in source.participant_refs:
            entity_decision = self._entity_resolver.resolve(
                participant_ref=participant_ref,
                candidates=context.entity_candidates_by_participant.get(
                    participant_ref, ()
                ),
            )
            if entity_decision.entity_object_id is not None:
                entity_ref = entity_decision.entity_object_id
            else:
                entity_ref = (
                    f"entity:{source.memory_id}:"
                    f"{_stable_token(participant_ref)}"
                )
                object_mutations.append(
                    ObjectMutation(
                        logical_ref=entity_ref,
                        operation=LifecycleOperation.CREATE,
                        object_type=MemoryObjectType.ENTITY,
                        evidence_ids=meeting_evidence,
                        payload={
                            "canonical_key": entity_ref,
                            "title": participant_ref,
                            "valid_from": source.started_at,
                            "confidence": entity_decision.confidence,
                            "attributes": {
                                "identity_scope": "meeting_session",
                                "speaker_ref": participant_ref,
                                "identity_aliases": (
                                    {
                                        "value": participant_ref,
                                        "confidence": entity_decision.confidence,
                                        "user_confirmed": False,
                                    },
                                ),
                                "resolver_version": entity_decision.resolver_version,
                            },
                        },
                    )
                )
            participant_entity_refs[_normalized(participant_ref)] = entity_ref
            relation_mutations.append(
                self._relation(
                    logical_ref=(
                        f"relation:participant:{source.memory_id}:"
                        f"{_stable_token(participant_ref)}"
                    ),
                    source_ref=entity_ref,
                    target_ref=meeting_ref,
                    relation_type="participated_in",
                    evidence_ids=meeting_evidence,
                    valid_from=source.started_at,
                    confidence=entity_decision.confidence,
                    epistemic_type=EpistemicType.OBSERVED,
                )
            )

        for mention in extraction.project_mentions:
            normalized_mention = _normalized(mention.mention)
            if normalized_mention in project_refs_by_mention:
                continue
            mention_evidence = self._evidence_refs(
                mention.evidence_spans, evidence_by_span
            )
            candidates = context.project_candidates_by_mention.get(
                mention.mention, ()
            )
            decision = self._project_resolver.resolve(
                task_object_ref=f"project-mention:{mention.mention}",
                candidates=candidates,
            )
            if decision.decision is ProjectLinkStatus.LINKED:
                project_ref = decision.primary_project_object_id
            elif not candidates and mention.confidence >= 0.95:
                project_ref = f"project:{_stable_token(normalized_mention)}"
                object_mutations.append(
                    ObjectMutation(
                        logical_ref=project_ref,
                        operation=LifecycleOperation.CREATE,
                        object_type=MemoryObjectType.PROJECT,
                        evidence_ids=mention_evidence,
                        payload={
                            "canonical_key": project_ref,
                            "title": mention.mention,
                            "valid_from": source.started_at,
                            "confidence": mention.confidence,
                            "attributes": {
                                "identity_aliases": (mention.mention,),
                                "resolution_status": "explicit_local_mention",
                                "resolver_version": "global-alignment/v1",
                            },
                        },
                    )
                )
            else:
                alignment_warnings.append(
                    f"project_mention_unresolved:{mention.mention}"
                )
                continue
            project_refs_by_mention[normalized_mention] = project_ref
            relation_mutations.append(
                self._relation(
                    logical_ref=(
                        f"relation:meeting-project:{source.memory_id}:"
                        f"{_stable_token(project_ref)}"
                    ),
                    source_ref=project_ref,
                    target_ref=meeting_ref,
                    relation_type="mentioned_in",
                    evidence_ids=mention_evidence,
                    valid_from=source.started_at,
                    confidence=mention.confidence,
                    epistemic_type=EpistemicType.OBSERVED,
                )
            )

        for claim in extraction.claims:
            claim_evidence = self._evidence_refs(claim.evidence_spans, evidence_by_span)
            owner_ref = participant_entity_refs.get(_normalized(claim.owner_mention))
            assertion_ref = f"assertion:{source.memory_id}:{claim.claim_id}"
            if claim.claim_type is ClaimType.CONDITION:
                assertion_mutations.append(
                    self._claim_assertion(
                        logical_ref=assertion_ref,
                        subject_ref=meeting_ref,
                        claim=claim,
                        evidence_ids=claim_evidence,
                        asserted_at=source.started_at,
                        asserted_by_entity_id=owner_ref,
                    )
                )
                continue

            object_type = _CLAIM_OBJECT_TYPES.get(claim.claim_type)
            if object_type is None:
                continue
            candidate = self._select_object_candidate(
                context.object_candidates_by_claim.get(claim.claim_id, ()),
                object_type,
            )
            object_ref = f"{object_type.value}:{source.memory_id}:{claim.claim_id}"
            payload = self._claim_payload(
                source=source,
                claim=claim,
                canonical_key=(
                    candidate.canonical_key if candidate is not None else object_ref
                ),
                evidence_ids=claim_evidence,
                owner_ref=owner_ref,
            )
            payload, lock_warnings = self._lifecycle_resolver.protect_user_locked_fields(
                existing=candidate,
                proposed=payload,
            )
            alignment_warnings.extend(lock_warnings)
            operation = self._lifecycle_resolver.resolve(
                existing=candidate,
                proposed=payload,
                contradictory=(
                    claim.lifecycle_signal is ClaimLifecycleSignal.CONTRADICTS
                ),
                supersedes=(
                    claim.lifecycle_signal is ClaimLifecycleSignal.SUPERSEDES
                ),
                resolved=(
                    claim.lifecycle_signal is ClaimLifecycleSignal.RESOLVED
                ),
                reopen=(
                    claim.lifecycle_signal is ClaimLifecycleSignal.REOPENED
                ),
            )
            assertion_mutations.append(
                self._claim_assertion(
                    logical_ref=assertion_ref,
                    subject_ref=object_ref,
                    claim=claim,
                    evidence_ids=claim_evidence,
                    asserted_at=source.started_at,
                    asserted_by_entity_id=owner_ref,
                )
            )
            mutation_payload = payload
            if candidate is not None and operation is LifecycleOperation.CONTRADICT:
                conflict_refs = tuple(
                    dict.fromkeys(
                        (
                            *(candidate.attributes.get("conflict_assertion_refs") or ()),
                            assertion_ref,
                        )
                    )
                )
                mutation_payload = {
                    "attributes": {
                        "has_unresolved_conflict": True,
                        "conflict_assertion_refs": conflict_refs,
                    }
                }
            object_mutations.append(
                ObjectMutation(
                    logical_ref=object_ref,
                    operation=operation,
                    object_type=object_type,
                    object_id=candidate.object_id if candidate is not None else None,
                    expected_version=(
                        candidate.lock_version if candidate is not None else None
                    ),
                    evidence_ids=claim_evidence,
                    payload=mutation_payload,
                )
            )
            relation_mutations.append(
                self._relation(
                    logical_ref=f"relation:meeting-claim:{source.memory_id}:{claim.claim_id}",
                    source_ref=object_ref,
                    target_ref=meeting_ref,
                    relation_type="mentioned_in",
                    evidence_ids=claim_evidence,
                    valid_from=source.started_at,
                    confidence=claim.confidence,
                    epistemic_type=EpistemicType.OBSERVED,
                )
            )
            if owner_ref is not None:
                relation_type = (
                    "committed_by"
                    if claim.claim_type is ClaimType.COMMITMENT
                    else "owned_by"
                )
                relation_mutations.append(
                    self._relation(
                        logical_ref=(
                            f"relation:owner:{source.memory_id}:{claim.claim_id}:"
                            f"{_stable_token(owner_ref)}"
                        ),
                        source_ref=object_ref,
                        target_ref=owner_ref,
                        relation_type=relation_type,
                        evidence_ids=claim_evidence,
                        valid_from=source.started_at,
                        confidence=claim.confidence,
                        epistemic_type=EpistemicType.REPORTED,
                    )
                )
            project_decision = self._project_resolver.resolve(
                task_object_ref=object_ref,
                candidates=context.project_candidates_by_claim.get(claim.claim_id, ()),
            )
            if project_decision.decision in {
                ProjectLinkStatus.LINKED,
                ProjectLinkStatus.MULTI_LINKED,
            }:
                for link in project_decision.project_links:
                    relation_mutations.append(
                        self._relation(
                            logical_ref=(
                                f"relation:project:{source.memory_id}:{claim.claim_id}:"
                                f"{link.project_object_id}"
                            ),
                            source_ref=object_ref,
                            target_ref=link.project_object_id,
                            relation_type="belongs_to_project",
                            evidence_ids=(
                                project_decision.evidence_ids or claim_evidence
                            ),
                            valid_from=source.started_at,
                            confidence=link.confidence,
                        )
                    )
            else:
                linked_project_refs = tuple(
                    dict.fromkeys(
                        project_refs_by_mention[_normalized(mention)]
                        for mention in claim.project_mentions
                        if _normalized(mention) in project_refs_by_mention
                    )
                )
                for project_ref in linked_project_refs:
                    relation_mutations.append(
                        self._relation(
                            logical_ref=(
                                f"relation:project:{source.memory_id}:"
                                f"{claim.claim_id}:{_stable_token(project_ref)}"
                            ),
                            source_ref=object_ref,
                            target_ref=project_ref,
                            relation_type="belongs_to_project",
                            evidence_ids=claim_evidence,
                            valid_from=source.started_at,
                            confidence=claim.confidence,
                            epistemic_type=EpistemicType.REPORTED,
                        )
                    )
            if candidate is not None and operation is not LifecycleOperation.CONFIRM:
                domain_events.append(
                    DomainEvent(
                        event_type="memory.dependencies_invalidated",
                        aggregate_ref=candidate.object_id,
                        payload={
                            "reason": "canonical_object_changed",
                            "resolver_version": "global-alignment/v1",
                        },
                    )
                )

        for topic in extraction.topic_candidates:
            self._append_topic(
                source=source,
                topic=topic,
                context=context,
                meeting_ref=meeting_ref,
                evidence_by_span=evidence_by_span,
                object_mutations=object_mutations,
                relation_mutations=relation_mutations,
            )

        expected_versions = {
            mutation.object_id: mutation.expected_version
            for mutation in object_mutations
            if mutation.object_id is not None and mutation.expected_version is not None
        }
        return self._planner.plan(
            MemoryChangeDraft(
                changeset_id=(
                    f"changeset:memory:{source.memory_id}:"
                    f"transcript:{source.transcript_version}"
                ),
                user_id=source.user_id,
                workspace_id=source.workspace_id,
                source_ref=SourceRef(
                    source_type="memory",
                    source_id=source.memory_id,
                    memory_id=source.memory_id,
                    transcript_version=source.transcript_version,
                ),
                base_state_version=context.base_state_version,
                expected_object_versions=expected_versions,
                evidence_creates=tuple(evidence_creates),
                object_mutations=tuple(object_mutations),
                assertion_mutations=tuple(assertion_mutations),
                relation_mutations=tuple(relation_mutations),
                domain_events=tuple(domain_events),
                warnings=tuple(alignment_warnings),
            )
        )

    @staticmethod
    def _select_object_candidate(
        candidates: tuple[ObjectLinkCandidate, ...], object_type: MemoryObjectType
    ) -> Optional[ObjectLinkCandidate]:
        eligible = tuple(
            sorted(
                (
                    item
                    for item in candidates
                    if item.object_type is object_type
                    and item.confidence >= 0.92
                    and set(item.anchor_types).intersection(_RELIABLE_OBJECT_ANCHORS)
                ),
                key=lambda item: (-item.confidence, item.object_id),
            )
        )
        if not eligible:
            return None
        if len(eligible) > 1 and eligible[0].confidence - eligible[1].confidence < 0.15:
            return None
        return eligible[0]

    @staticmethod
    def _claim_payload(
        *, source, claim, canonical_key, evidence_ids, owner_ref=None
    ):
        payload = {
            "canonical_key": canonical_key,
            "title": claim.text,
            "valid_from": source.started_at,
            "confidence": claim.confidence,
            "attributes": {
                "claim_type": claim.claim_type.value,
                "condition": claim.condition,
                "negated": claim.negated,
                "modality": claim.modality.value,
                "project_mentions": claim.project_mentions,
                "object_mentions": claim.object_mentions,
                "owner_mention": claim.owner_mention,
            },
            "field_provenance": (
                FieldProvenance(field_name="title", evidence_ids=evidence_ids),
            ),
        }
        if claim.claim_type is ClaimType.DECISION:
            payload.update(
                {
                    "decision_owner": owner_ref or claim.owner_mention,
                    "decision": claim.text,
                    "effective_status": "effective",
                    "effective_from": source.started_at,
                }
            )
        elif claim.claim_type is ClaimType.COMMITMENT:
            is_completed = (
                claim.lifecycle_signal is ClaimLifecycleSignal.RESOLVED
            )
            payload.update(
                {
                    "committed_by": owner_ref or "unresolved",
                    "action": claim.text,
                    "committed_at": source.started_at,
                    "due_at": claim.due_at,
                    "fulfillment_status": (
                        FulfillmentStatus.COMPLETED
                        if is_completed
                        else FulfillmentStatus.OPEN
                    ),
                    "completion_evidence_ids": (
                        evidence_ids if is_completed else ()
                    ),
                    "workflow_status": (
                        "completed" if is_completed else "in_progress"
                    ),
                }
            )
        elif claim.claim_type is ClaimType.TASK:
            is_completed = (
                claim.lifecycle_signal is ClaimLifecycleSignal.RESOLVED
            )
            payload["attributes"].update(
                {
                    "action": claim.action,
                    "owner_entity_id": owner_ref,
                    "execution_intent": claim.task_intent.value,
                    "due_at": claim.due_at,
                }
            )
            payload["workflow_status"] = (
                "completed" if is_completed else "in_progress"
            )
        elif claim.claim_type in {ClaimType.BLOCKER, ClaimType.OBJECTION}:
            payload.update(
                {
                    "subtype": claim.claim_type.value,
                    "severity": "unknown",
                    "owner": owner_ref or claim.owner_mention,
                    "resolution_status": (
                        "resolved"
                        if claim.lifecycle_signal is ClaimLifecycleSignal.RESOLVED
                        else "open"
                    ),
                    "resolution_evidence_ids": (
                        evidence_ids
                        if claim.lifecycle_signal is ClaimLifecycleSignal.RESOLVED
                        else ()
                    ),
                    "workflow_status": (
                        "completed"
                        if claim.lifecycle_signal is ClaimLifecycleSignal.RESOLVED
                        else "in_progress"
                    ),
                }
            )
        return payload

    @staticmethod
    def _claim_assertion(
        *,
        logical_ref,
        subject_ref,
        claim,
        evidence_ids,
        asserted_at,
        asserted_by_entity_id,
    ):
        return AssertionMutation(
            logical_ref=logical_ref,
            operation=LifecycleOperation.CREATE,
            evidence_ids=evidence_ids,
            payload={
                "subject_object_ref": subject_ref,
                "predicate": claim.claim_type.value,
                "value": {
                    "text": claim.text,
                    "condition": claim.condition,
                    "owner_mention": claim.owner_mention,
                    "due_at": (
                        claim.due_at.isoformat()
                        if claim.due_at is not None
                        else None
                    ),
                    "lifecycle_signal": claim.lifecycle_signal.value,
                },
                "asserted_by_entity_id": asserted_by_entity_id,
                "epistemic_type": EpistemicType.REPORTED,
                "modality": claim.modality.value,
                "polarity": (
                    Polarity.NEGATIVE if claim.negated else Polarity.POSITIVE
                ),
                "confidence": claim.confidence,
                "asserted_at": asserted_at,
            },
        )

    def _append_topic(
        self,
        *,
        source,
        topic: SessionTopicCandidate,
        context,
        meeting_ref,
        evidence_by_span,
        object_mutations,
        relation_mutations,
    ):
        evidence_ids = self._evidence_refs(topic.evidence_spans, evidence_by_span)
        decision = self._topic_resolver.resolve(
            session_candidate=topic,
            matches=context.topic_matches_by_candidate.get(topic.candidate_id, ()),
        )
        if decision.decision is TopicResolutionStatus.LINKED:
            topic_ref = decision.topic_object_id
            match = next(
                item
                for item in context.topic_matches_by_candidate.get(
                    topic.candidate_id, ()
                )
                if item.topic_object_id == topic_ref
            )
            if match.lock_version is None:
                raise ValueError(
                    "linked topic candidate requires lock_version"
                )
            last_seen_at = max(
                value
                for value in (match.last_seen_at, source.started_at)
                if value is not None
            )
            scope_project_ids = tuple(
                dict.fromkeys(
                    (*match.scope_project_ids, *topic.project_mentions)
                )
            )
            object_mutations.append(
                ObjectMutation(
                    logical_ref=(
                        f"topic:update:{source.memory_id}:"
                        f"{_stable_token(topic.candidate_id, topic_ref)}"
                    ),
                    operation=LifecycleOperation.UPDATE,
                    object_type=MemoryObjectType.TOPIC,
                    object_id=topic_ref,
                    expected_version=match.lock_version,
                    evidence_ids=evidence_ids,
                    payload={
                        "last_seen_at": last_seen_at,
                        "resolution_status": (
                            TopicResolutionStatus.LINKED.value
                        ),
                        "scope_project_ids": scope_project_ids,
                        "field_provenance": (
                            FieldProvenance(
                                field_name="last_seen_at",
                                evidence_ids=evidence_ids,
                            ),
                            FieldProvenance(
                                field_name="scope_project_ids",
                                evidence_ids=evidence_ids,
                            ),
                        ),
                    },
                )
            )
        else:
            prefix = (
                "topic"
                if decision.decision is TopicResolutionStatus.NEW_TOPIC
                else "session-topic"
            )
            topic_ref = (
                f"{prefix}:{source.memory_id}:"
                f"{_stable_token(topic.candidate_id, topic.label)}"
            )
            object_mutations.append(
                ObjectMutation(
                    logical_ref=topic_ref,
                    operation=LifecycleOperation.CREATE,
                    object_type=MemoryObjectType.TOPIC,
                    evidence_ids=evidence_ids,
                    payload={
                        "canonical_key": topic_ref,
                        "title": topic.label,
                        "valid_from": source.started_at,
                        "canonical_label": topic.label,
                        "first_seen_at": source.started_at,
                        "last_seen_at": source.started_at,
                        "resolution_status": decision.decision.value,
                        "scope_project_ids": topic.project_mentions,
                        "confidence": topic.confidence,
                        "attributes": {
                            "shadow_only": True,
                            "session_candidate_id": topic.candidate_id,
                            "resolver_version": decision.resolver_version,
                        },
                    },
                )
            )
        if topic_ref is not None:
            relation_mutations.append(
                self._relation(
                    logical_ref=(
                        f"relation:meeting-topic:{source.memory_id}:"
                        f"{topic.candidate_id}"
                    ),
                    source_ref=meeting_ref,
                    target_ref=topic_ref,
                    relation_type="discusses_topic",
                    evidence_ids=evidence_ids,
                    valid_from=source.started_at,
                    confidence=topic.confidence,
                )
            )

    @staticmethod
    def _relation(
        *,
        logical_ref,
        source_ref,
        target_ref,
        relation_type,
        evidence_ids,
        valid_from,
        confidence,
        epistemic_type=EpistemicType.INFERRED,
    ):
        return RelationMutation(
            logical_ref=logical_ref,
            operation=LifecycleOperation.CREATE,
            source_object_ref=source_ref,
            target_object_ref=target_ref,
            relation_type=relation_type,
            evidence_ids=evidence_ids,
            payload={
                "confidence": confidence,
                "epistemic_type": epistemic_type,
                "valid_from": valid_from,
            },
        )

    @staticmethod
    def _evidence_refs(
        spans: tuple[EvidenceSpan, ...], evidence_by_span: Dict[Tuple[str, int, int], str]
    ) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                evidence_by_span[(span.segment_id, span.start_char, span.end_char)]
                for span in spans
            )
        )

    @staticmethod
    def _build_evidence(source, extraction):
        segment_by_id = {item.segment_id: item for item in source.segments}
        spans = []
        spans.extend(span for claim in extraction.claims for span in claim.evidence_spans)
        spans.extend(
            span
            for mention in extraction.project_mentions
            for span in mention.evidence_spans
        )
        spans.extend(
            span
            for topic in extraction.topic_candidates
            for span in topic.evidence_spans
        )
        evidence = []
        by_span = {}
        for span in spans:
            key = (span.segment_id, span.start_char, span.end_char)
            if key in by_span:
                continue
            segment = segment_by_id[span.segment_id]
            content = segment.text[span.start_char : span.end_char]
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            logical_ref = (
                f"evidence:{source.memory_id}:{source.transcript_version}:"
                f"{_stable_token(*key, digest)}"
            )
            by_span[key] = logical_ref
            evidence.append(
                EvidenceCreate(
                    logical_ref=logical_ref,
                    evidence=Evidence(
                        evidence_id=logical_ref,
                        user_id=source.user_id,
                        workspace_id=source.workspace_id,
                        source_type="transcript_segment",
                        source_id=span.segment_id,
                        memory_id=source.memory_id,
                        transcript_version=source.transcript_version,
                        speaker_id=segment.speaker_ref,
                        start_ms=segment.start_ms,
                        end_ms=segment.end_ms,
                        content=content,
                        content_hash=digest,
                        recorded_at=source.started_at,
                        created_at=source.started_at,
                    ),
                )
            )
        return evidence, by_span

    @staticmethod
    def _fallback_evidence(source, evidence_creates, evidence_by_span):
        segment = source.segments[0]
        span = EvidenceSpan(
            segment_id=segment.segment_id, start_char=0, end_char=len(segment.text)
        )
        content = segment.text
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        logical_ref = (
            f"evidence:{source.memory_id}:{source.transcript_version}:"
            f"{_stable_token(span.segment_id, span.start_char, span.end_char, digest)}"
        )
        evidence_by_span[(span.segment_id, span.start_char, span.end_char)] = logical_ref
        evidence_creates.append(
            EvidenceCreate(
                logical_ref=logical_ref,
                evidence=Evidence(
                    evidence_id=logical_ref,
                    user_id=source.user_id,
                    workspace_id=source.workspace_id,
                    source_type="transcript_segment",
                    source_id=segment.segment_id,
                    memory_id=source.memory_id,
                    transcript_version=source.transcript_version,
                    speaker_id=segment.speaker_ref,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    content=content,
                    content_hash=digest,
                    recorded_at=source.started_at,
                    created_at=source.started_at,
                ),
            )
        )
        return (logical_ref,)
