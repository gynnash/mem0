"""Controlled correction planning; never exposed to ordinary Feature Agents."""

from datetime import datetime
from typing import Optional

from mem0.v3.contracts import (
    AssertionCreatePayload,
    AssertionMutation,
    DomainEvent,
    EvidenceCreate,
    ObjectMutation,
    ObjectMutationPayload,
    RelationCreatePayload,
    RelationMutation,
    RetractionMutation,
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
    RelationType,
    RetractionTargetType,
    WorkflowStatus,
)
from mem0.v3.planner import MemoryChangeDraft, MemoryPlanner


class ControlledMemoryMaintenance:
    def __init__(self, planner: Optional[MemoryPlanner] = None) -> None:
        self._planner = planner or MemoryPlanner()

    def resolve_commitment(
        self,
        *,
        changeset_id: str,
        user_id: str,
        workspace_id: str,
        base_state_version: int,
        commitment_object_id: str,
        commitment_expected_version: int,
        completion_evidence: Evidence,
    ) -> ValidatedMemoryChangeSet:
        """Plan a trusted product-action completion without persisting it.

        Summora is responsible for proving the Todo-to-Commitment binding is
        unique and current before calling this controlled semantic operation.
        """

        evidence_ref = "commitment:completion_evidence"
        return self._planner.plan(
            MemoryChangeDraft(
                changeset_id=changeset_id,
                user_id=user_id,
                workspace_id=workspace_id,
                source_ref=SourceRef(
                    source_type="product_action",
                    source_id=completion_evidence.source_id,
                    memory_id=completion_evidence.memory_id,
                ),
                base_state_version=base_state_version,
                expected_object_versions={
                    commitment_object_id: commitment_expected_version
                },
                evidence_creates=(
                    EvidenceCreate(
                        logical_ref=evidence_ref,
                        evidence=completion_evidence,
                    ),
                ),
                object_mutations=(
                    ObjectMutation(
                        logical_ref="commitment:resolve",
                        operation=LifecycleOperation.RESOLVE,
                        object_type=MemoryObjectType.COMMITMENT,
                        object_id=commitment_object_id,
                        expected_version=commitment_expected_version,
                        evidence_ids=(evidence_ref,),
                        payload={
                            "workflow_status": WorkflowStatus.COMPLETED,
                            "fulfillment_status": FulfillmentStatus.COMPLETED,
                            "completion_evidence_ids": (evidence_ref,),
                            "field_provenance": (
                                FieldProvenance(
                                    field_name="fulfillment_status",
                                    evidence_ids=(evidence_ref,),
                                    user_confirmed=True,
                                    user_locked=True,
                                    corrected_at=completion_evidence.recorded_at,
                                ),
                            ),
                        },
                    ),
                ),
                domain_events=(
                    DomainEvent(
                        event_type="memory.commitment_resolved",
                        aggregate_ref=commitment_object_id,
                        payload={
                            "reason": "user_confirmed_todo_completion",
                            "completion_evidence_ref": evidence_ref,
                        },
                    ),
                    DomainEvent(
                        event_type="memory.dependencies_invalidated",
                        aggregate_ref=commitment_object_id,
                        payload={
                            "reason": "canonical_object_changed",
                            "trigger": "commitment_resolved",
                        },
                    ),
                ),
            )
        )

    def retract(
        self,
        *,
        changeset_id: str,
        user_id: str,
        workspace_id: str,
        base_state_version: int,
        target_type: RetractionTargetType,
        target_id: str,
        reason: str,
        evidence_ids: tuple[str, ...] = (),
        user_initiated: bool = False,
    ) -> ValidatedMemoryChangeSet:
        return self._planner.plan(
            MemoryChangeDraft(
                changeset_id=changeset_id,
                user_id=user_id,
                workspace_id=workspace_id,
                source_ref=SourceRef(
                    source_type=(
                        "user_correction" if user_initiated else "internal_correction"
                    ),
                    source_id=changeset_id,
                ),
                base_state_version=base_state_version,
                retractions=(
                    RetractionMutation(
                        logical_ref=f"retract:{target_type.value}:{target_id}",
                        target_type=target_type,
                        target_id=target_id,
                        reason=reason,
                        evidence_ids=evidence_ids,
                    ),
                ),
                domain_events=(
                    DomainEvent(
                        event_type="memory.dependencies_invalidated",
                        aggregate_ref=target_id,
                        payload={"reason": "controlled_retraction"},
                    ),
                ),
            )
        )

    def retract_source_evidence(
        self,
        *,
        changeset_id: str,
        user_id: str,
        workspace_id: str,
        base_state_version: int,
        evidence_ids: tuple[str, ...],
        source_id: str,
        reason: str,
    ) -> ValidatedMemoryChangeSet:
        """Plan privacy/source revocation for all evidence from one source."""

        evidence_ids = tuple(dict.fromkeys(evidence_ids))
        if not evidence_ids:
            raise ValueError("source evidence retraction requires evidence")
        return self._planner.plan(
            MemoryChangeDraft(
                changeset_id=changeset_id,
                user_id=user_id,
                workspace_id=workspace_id,
                source_ref=SourceRef(
                    source_type="privacy_deletion",
                    source_id=source_id,
                ),
                base_state_version=base_state_version,
                retractions=tuple(
                    RetractionMutation(
                        logical_ref=f"source:retract:{index}",
                        target_type=RetractionTargetType.EVIDENCE,
                        target_id=evidence_id,
                        reason=reason,
                    )
                    for index, evidence_id in enumerate(evidence_ids)
                ),
                domain_events=tuple(
                    DomainEvent(
                        event_type="memory.dependencies_invalidated",
                        aggregate_ref=evidence_id,
                        payload={"reason": "source_evidence_retracted"},
                    )
                    for evidence_id in evidence_ids
                ),
            )
        )

    def recompute_object_support(
        self,
        *,
        changeset_id: str,
        user_id: str,
        workspace_id: str,
        base_state_version: int,
        object_type: MemoryObjectType,
        object_id: str,
        expected_version: int,
        evidence_ids: tuple[str, ...],
        field_provenance: tuple[FieldProvenance, ...],
        canonical_key: str | None = None,
        external_memory_id: str | None = None,
    ) -> ValidatedMemoryChangeSet:
        """Re-version an object after its usable evidence set changed."""

        if not evidence_ids:
            raise ValueError("object support recomputation requires evidence")
        payload = ObjectMutationPayload(
            field_provenance=field_provenance,
            **(
                {
                    "canonical_key": canonical_key,
                    "external_memory_id": external_memory_id,
                }
                if object_type is MemoryObjectType.MEETING
                else {}
            ),
        )
        return self._planner.plan(
            MemoryChangeDraft(
                changeset_id=changeset_id,
                user_id=user_id,
                workspace_id=workspace_id,
                source_ref=SourceRef(
                    source_type="internal_correction",
                    source_id=changeset_id,
                ),
                base_state_version=base_state_version,
                expected_object_versions={object_id: expected_version},
                object_mutations=(
                    ObjectMutation(
                        logical_ref="dependency:object_support",
                        operation=LifecycleOperation.UPDATE,
                        object_type=object_type,
                        object_id=object_id,
                        expected_version=expected_version,
                        evidence_ids=evidence_ids,
                        payload=payload,
                    ),
                ),
                domain_events=(
                    DomainEvent(
                        event_type="memory.dependencies_invalidated",
                        aggregate_ref=object_id,
                        payload={"reason": "evidence_support_recomputed"},
                    ),
                ),
            )
        )

    def replace_assertion_support(
        self,
        *,
        changeset_id: str,
        user_id: str,
        workspace_id: str,
        base_state_version: int,
        assertion_id: str,
        evidence_ids: tuple[str, ...],
        payload: AssertionCreatePayload,
    ) -> ValidatedMemoryChangeSet:
        """Replace an immutable assertion with its still-supported form."""

        if not evidence_ids:
            raise ValueError("assertion support replacement requires evidence")
        return self._planner.plan(
            MemoryChangeDraft(
                changeset_id=changeset_id,
                user_id=user_id,
                workspace_id=workspace_id,
                source_ref=SourceRef(
                    source_type="internal_correction",
                    source_id=changeset_id,
                ),
                base_state_version=base_state_version,
                assertion_mutations=(
                    AssertionMutation(
                        logical_ref="dependency:assertion_replacement",
                        operation=LifecycleOperation.CREATE,
                        evidence_ids=evidence_ids,
                        payload=payload,
                    ),
                ),
                retractions=(
                    RetractionMutation(
                        logical_ref="dependency:assertion_retraction",
                        target_type=RetractionTargetType.ASSERTION,
                        target_id=assertion_id,
                        reason="evidence_support_changed",
                        evidence_ids=evidence_ids,
                    ),
                ),
                domain_events=(
                    DomainEvent(
                        event_type="memory.dependencies_invalidated",
                        aggregate_ref=assertion_id,
                        payload={"reason": "evidence_support_recomputed"},
                    ),
                ),
            )
        )

    def replace_relation_support(
        self,
        *,
        changeset_id: str,
        user_id: str,
        workspace_id: str,
        base_state_version: int,
        relation_id: str,
        source_object_id: str,
        target_object_id: str,
        relation_type: RelationType,
        evidence_ids: tuple[str, ...],
        payload: RelationCreatePayload,
    ) -> ValidatedMemoryChangeSet:
        """Replace an immutable relation with its still-supported form."""

        if not evidence_ids:
            raise ValueError("relation support replacement requires evidence")
        return self._planner.plan(
            MemoryChangeDraft(
                changeset_id=changeset_id,
                user_id=user_id,
                workspace_id=workspace_id,
                source_ref=SourceRef(
                    source_type="internal_correction",
                    source_id=changeset_id,
                ),
                base_state_version=base_state_version,
                relation_mutations=(
                    RelationMutation(
                        logical_ref="dependency:relation_replacement",
                        operation=LifecycleOperation.CREATE,
                        source_object_ref=source_object_id,
                        target_object_ref=target_object_id,
                        relation_type=relation_type,
                        evidence_ids=evidence_ids,
                        payload=payload,
                    ),
                ),
                retractions=(
                    RetractionMutation(
                        logical_ref="dependency:relation_retraction",
                        target_type=RetractionTargetType.RELATION,
                        target_id=relation_id,
                        reason="evidence_support_changed",
                        evidence_ids=evidence_ids,
                    ),
                ),
                domain_events=(
                    DomainEvent(
                        event_type="memory.dependencies_invalidated",
                        aggregate_ref=relation_id,
                        payload={"reason": "evidence_support_recomputed"},
                    ),
                ),
            )
        )

    def bind_entity_alias(
        self,
        *,
        changeset_id: str,
        user_id: str,
        workspace_id: str,
        base_state_version: int,
        entity_object_id: str,
        entity_expected_version: int,
        alias: str,
        evidence_ids: tuple[str, ...],
        speaker_ref: str | None = None,
    ) -> ValidatedMemoryChangeSet:
        alias = alias.strip()
        if not alias:
            raise ValueError("entity alias must not be empty")
        attributes = {
            "identity_aliases": (
                {
                    "value": alias,
                    "confidence": 1,
                    "user_confirmed": True,
                    "evidence_ids": evidence_ids,
                },
            )
        }
        if speaker_ref is not None:
            speaker_ref = speaker_ref.strip()
            if not speaker_ref.startswith("speaker:"):
                raise ValueError("speaker_ref must use speaker:{id}")
            attributes["speaker_ref"] = speaker_ref
        provenance = [
            FieldProvenance(
                field_name="identity_aliases",
                evidence_ids=evidence_ids,
                user_confirmed=True,
                user_locked=True,
            )
        ]
        if speaker_ref is not None:
            provenance.append(
                FieldProvenance(
                    field_name="speaker_ref",
                    evidence_ids=evidence_ids,
                    user_confirmed=True,
                    user_locked=True,
                )
            )
        return self._planner.plan(
            MemoryChangeDraft(
                changeset_id=changeset_id,
                user_id=user_id,
                workspace_id=workspace_id,
                source_ref=SourceRef(
                    source_type="user_correction", source_id=changeset_id
                ),
                base_state_version=base_state_version,
                expected_object_versions={
                    entity_object_id: entity_expected_version
                },
                object_mutations=(
                    ObjectMutation(
                        logical_ref="entity:bind_alias",
                        operation=LifecycleOperation.ENRICH,
                        object_type=MemoryObjectType.ENTITY,
                        object_id=entity_object_id,
                        expected_version=entity_expected_version,
                        evidence_ids=evidence_ids,
                        payload={
                            "attributes": attributes,
                            "field_provenance": tuple(provenance),
                        },
                    ),
                ),
                domain_events=(
                    DomainEvent(
                        event_type="memory.dependencies_invalidated",
                        aggregate_ref=entity_object_id,
                        payload={"reason": "user_confirmed_entity_alias"},
                    ),
                ),
            )
        )

    def unbind_entity_speaker(
        self,
        *,
        changeset_id: str,
        user_id: str,
        workspace_id: str,
        base_state_version: int,
        entity_object_id: str,
        entity_expected_version: int,
        evidence_ids: tuple[str, ...],
    ) -> ValidatedMemoryChangeSet:
        if not evidence_ids:
            raise ValueError("speaker unbinding requires evidence")
        return self._planner.plan(
            MemoryChangeDraft(
                changeset_id=changeset_id,
                user_id=user_id,
                workspace_id=workspace_id,
                source_ref=SourceRef(
                    source_type="user_correction",
                    source_id=changeset_id,
                ),
                base_state_version=base_state_version,
                expected_object_versions={
                    entity_object_id: entity_expected_version
                },
                object_mutations=(
                    ObjectMutation(
                        logical_ref="entity:unbind_speaker",
                        operation=LifecycleOperation.UPDATE,
                        object_type=MemoryObjectType.ENTITY,
                        object_id=entity_object_id,
                        expected_version=entity_expected_version,
                        evidence_ids=evidence_ids,
                        payload={
                            "attributes": {"speaker_ref": ""},
                            "field_provenance": (
                                FieldProvenance(
                                    field_name="speaker_ref",
                                    evidence_ids=evidence_ids,
                                    user_confirmed=True,
                                    user_locked=True,
                                ),
                            ),
                        },
                    ),
                ),
                domain_events=(
                    DomainEvent(
                        event_type="memory.dependencies_invalidated",
                        aggregate_ref=entity_object_id,
                        payload={"reason": "user_removed_speaker_binding"},
                    ),
                ),
            )
        )

    def merge_objects(
        self,
        *,
        changeset_id: str,
        user_id: str,
        workspace_id: str,
        base_state_version: int,
        object_type: MemoryObjectType,
        target_object_id: str,
        target_expected_version: int,
        source_object_ids: tuple[str, ...],
        evidence_ids: tuple[str, ...],
        valid_from: datetime,
        reason: str,
    ) -> ValidatedMemoryChangeSet:
        if not source_object_ids:
            raise ValueError("merge requires at least one source object")
        if target_object_id in source_object_ids:
            raise ValueError("merge target cannot also be a source object")
        return self._planner.plan(
            MemoryChangeDraft(
                changeset_id=changeset_id,
                user_id=user_id,
                workspace_id=workspace_id,
                source_ref=SourceRef(
                    source_type="internal_correction", source_id=changeset_id
                ),
                base_state_version=base_state_version,
                expected_object_versions={
                    target_object_id: target_expected_version
                },
                object_mutations=(
                    ObjectMutation(
                        logical_ref="merge:target",
                        operation=LifecycleOperation.MERGE,
                        object_type=object_type,
                        object_id=target_object_id,
                        expected_version=target_expected_version,
                        evidence_ids=evidence_ids,
                        payload={
                            "attributes": {
                                "merged_source_object_ids": source_object_ids,
                                "merge_reason": reason,
                            }
                        },
                    ),
                ),
                relation_mutations=tuple(
                    RelationMutation(
                        logical_ref=f"merge:relation:{source_id}",
                        operation=LifecycleOperation.CREATE,
                        source_object_ref=target_object_id,
                        target_object_ref=source_id,
                        relation_type="merged_from",
                        evidence_ids=evidence_ids,
                        payload={
                            "confidence": 1,
                            "epistemic_type": EpistemicType.USER_CONFIRMED,
                            "valid_from": valid_from,
                        },
                    )
                    for source_id in source_object_ids
                ),
                retractions=tuple(
                    RetractionMutation(
                        logical_ref=f"merge:retract:{source_id}",
                        target_type=RetractionTargetType.OBJECT,
                        target_id=source_id,
                        reason=reason,
                        evidence_ids=evidence_ids,
                    )
                    for source_id in source_object_ids
                ),
                domain_events=(
                    DomainEvent(
                        event_type="memory.dependencies_invalidated",
                        aggregate_ref=target_object_id,
                        payload={"reason": "controlled_merge"},
                    ),
                ),
            )
        )

    def split_object(
        self,
        *,
        changeset_id: str,
        user_id: str,
        workspace_id: str,
        base_state_version: int,
        object_type: MemoryObjectType,
        source_object_id: str,
        source_expected_version: int,
        child_mutations: tuple[ObjectMutation, ...],
        evidence_ids: tuple[str, ...],
        reason: str,
    ) -> ValidatedMemoryChangeSet:
        if len(child_mutations) < 2:
            raise ValueError("split requires at least two child objects")
        if any(
            child.operation is not LifecycleOperation.CREATE
            or child.object_type is not object_type
            for child in child_mutations
        ):
            raise ValueError("split children must be new objects of the source type")
        return self._planner.plan(
            MemoryChangeDraft(
                changeset_id=changeset_id,
                user_id=user_id,
                workspace_id=workspace_id,
                source_ref=SourceRef(
                    source_type="internal_correction", source_id=changeset_id
                ),
                base_state_version=base_state_version,
                expected_object_versions={
                    source_object_id: source_expected_version
                },
                object_mutations=(
                    ObjectMutation(
                        logical_ref="split:source",
                        operation=LifecycleOperation.SPLIT,
                        object_type=object_type,
                        object_id=source_object_id,
                        expected_version=source_expected_version,
                        evidence_ids=evidence_ids,
                        payload={
                            "attributes": {
                                "split_child_refs": tuple(
                                    child.logical_ref for child in child_mutations
                                ),
                                "split_reason": reason,
                            }
                        },
                    ),
                    *child_mutations,
                ),
                domain_events=(
                    DomainEvent(
                        event_type="memory.dependencies_invalidated",
                        aggregate_ref=source_object_id,
                        payload={"reason": "controlled_split"},
                    ),
                ),
            )
        )

    def rebind_relation(
        self,
        *,
        changeset_id: str,
        user_id: str,
        workspace_id: str,
        base_state_version: int,
        old_relation_id: str,
        source_object_id: str,
        target_object_id: str,
        relation_type: str,
        evidence_ids: tuple[str, ...],
        valid_from,
        reason: str,
    ) -> ValidatedMemoryChangeSet:
        return self._planner.plan(
            MemoryChangeDraft(
                changeset_id=changeset_id,
                user_id=user_id,
                workspace_id=workspace_id,
                source_ref=SourceRef(
                    source_type="internal_correction", source_id=changeset_id
                ),
                base_state_version=base_state_version,
                relation_mutations=(
                    RelationMutation(
                        logical_ref="rebind:new_relation",
                        operation=LifecycleOperation.CREATE,
                        source_object_ref=source_object_id,
                        target_object_ref=target_object_id,
                        relation_type=relation_type,
                        evidence_ids=evidence_ids,
                        payload={
                            "confidence": 1,
                            "epistemic_type": EpistemicType.USER_CONFIRMED,
                            "valid_from": valid_from,
                        },
                    ),
                ),
                retractions=(
                    RetractionMutation(
                        logical_ref=f"rebind:retract:{old_relation_id}",
                        target_type=RetractionTargetType.RELATION,
                        target_id=old_relation_id,
                        reason=reason,
                        evidence_ids=evidence_ids,
                    ),
                ),
                domain_events=(
                    DomainEvent(
                        event_type="memory.dependencies_invalidated",
                        aggregate_ref=source_object_id,
                        payload={"reason": "controlled_rebind"},
                    ),
                ),
            )
        )
