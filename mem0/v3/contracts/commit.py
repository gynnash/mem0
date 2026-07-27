"""Receipt returned by the Summora-owned physical commit boundary."""

from pydantic import Field, model_validator

from mem0.v3.contracts.base import FrozenContract, NonEmptyStr


class MemoryCommitReceipt(FrozenContract):
    operation_key: NonEmptyStr
    changeset_id: NonEmptyStr
    committed_state_version: int = Field(ge=0)
    persisted_ids: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    audit_event_ids: tuple[NonEmptyStr, ...] = ()
    outbox_event_ids: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def validate_event_ids(self) -> "MemoryCommitReceipt":
        if len(self.audit_event_ids) != len(set(self.audit_event_ids)):
            raise ValueError("audit event ids must be unique")
        if len(self.outbox_event_ids) != len(set(self.outbox_event_ids)):
            raise ValueError("outbox event ids must be unique")
        return self
