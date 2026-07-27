"""Pure, deterministic validation rules for semantic memory changes."""

from mem0.v3.contracts import ValidatedMemoryChangeSet
from mem0.v3.domain import LifecycleOperation, MemoryObjectType


_EVIDENCE_OPTIONAL_OBJECT_OPERATIONS = {
    LifecycleOperation.ARCHIVE,
    LifecycleOperation.DELETE,
}


class EvidenceRequiredRule:
    def validate(self, changeset: ValidatedMemoryChangeSet) -> None:
        for mutation in changeset.object_mutations:
            if mutation.operation not in _EVIDENCE_OPTIONAL_OBJECT_OPERATIONS and not mutation.evidence_ids:
                raise ValueError(f"object mutation {mutation.logical_ref!r} requires evidence")
        for mutation in changeset.assertion_mutations:
            if not mutation.evidence_ids:
                raise ValueError(f"assertion mutation {mutation.logical_ref!r} requires evidence")
        for mutation in changeset.relation_mutations:
            if not mutation.evidence_ids:
                raise ValueError(f"relation mutation {mutation.logical_ref!r} requires evidence")


class ExplicitRetractionRule:
    def validate(self, changeset: ValidatedMemoryChangeSet) -> None:
        mutation_groups = (
            changeset.object_mutations,
            changeset.assertion_mutations,
            changeset.relation_mutations,
        )
        for group in mutation_groups:
            for mutation in group:
                if mutation.operation is LifecycleOperation.RETRACT:
                    raise ValueError("retractions must use the explicit retractions collection")


class ImmutableStatementRule:
    def validate(self, changeset: ValidatedMemoryChangeSet) -> None:
        for mutation in changeset.assertion_mutations:
            if mutation.operation is not LifecycleOperation.CREATE:
                raise ValueError("assertions are immutable; create a replacement and retract the old assertion")
        for mutation in changeset.relation_mutations:
            if mutation.operation is not LifecycleOperation.CREATE:
                raise ValueError("relations are immutable; create a replacement and retract the old relation")


class ExpectedObjectVersionRule:
    def validate(self, changeset: ValidatedMemoryChangeSet) -> None:
        for mutation in changeset.object_mutations:
            if mutation.operation is LifecycleOperation.CREATE:
                continue
            if mutation.object_id is None:
                raise ValueError(f"object mutation {mutation.logical_ref!r} requires object_id")
            expected_version = mutation.expected_version
            if expected_version is None:
                expected_version = changeset.expected_object_versions.get(mutation.object_id)
            if expected_version is None:
                raise ValueError(f"object mutation {mutation.logical_ref!r} requires an expected object version")


class MeetingIdentityRule:
    def validate(self, changeset: ValidatedMemoryChangeSet) -> None:
        for mutation in changeset.object_mutations:
            if mutation.object_type is not MemoryObjectType.MEETING:
                continue
            external_memory_id = str(mutation.payload.external_memory_id or "").strip()
            canonical_key = str(mutation.payload.canonical_key or "").strip()
            if not external_memory_id:
                raise ValueError("meeting mutations require external_memory_id")
            if canonical_key != f"meeting:{external_memory_id}":
                raise ValueError("meeting canonical_key must be meeting:{external_memory_id}")


class ControlledLifecycleRule:
    _CONTROLLED_SOURCE_TYPES = {"internal_correction", "user_correction"}

    def validate(self, changeset: ValidatedMemoryChangeSet) -> None:
        for mutation in changeset.object_mutations:
            if mutation.operation is LifecycleOperation.CREATE:
                if mutation.object_id is not None or mutation.expected_version is not None:
                    raise ValueError("create object mutation cannot target an existing version")
                continue
            if mutation.operation in {
                LifecycleOperation.MERGE,
                LifecycleOperation.SPLIT,
            }:
                if changeset.source_ref.source_type not in self._CONTROLLED_SOURCE_TYPES:
                    raise ValueError("merge/split require a controlled correction source")
                if mutation.object_type is MemoryObjectType.MEETING:
                    raise ValueError("deterministic Meeting objects cannot be merged or split")
            if (
                mutation.operation is LifecycleOperation.DELETE
                and changeset.source_ref.source_type
                not in {*self._CONTROLLED_SOURCE_TYPES, "privacy_deletion"}
            ):
                raise ValueError("delete requires a correction or privacy-deletion source")


DEFAULT_CHANGESET_RULES = (
    EvidenceRequiredRule(),
    ExplicitRetractionRule(),
    ImmutableStatementRule(),
    ExpectedObjectVersionRule(),
    MeetingIdentityRule(),
    ControlledLifecycleRule(),
)
