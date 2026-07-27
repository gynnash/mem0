"""Consistent read watermark shared by all tools in one agent run."""

from datetime import datetime

from pydantic import Field, field_validator, model_validator

from mem0.v3.contracts.base import FrozenContract, NonEmptyStr


class MemoryReadSnapshot(FrozenContract):
    user_id: NonEmptyStr
    workspace_id: NonEmptyStr
    as_of_event_id: int = Field(ge=0)
    projection_checkpoint: int = Field(ge=0)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_checkpoint(self) -> "MemoryReadSnapshot":
        if self.projection_checkpoint > self.as_of_event_id:
            raise ValueError("projection checkpoint cannot exceed the source-of-truth event id")
        return self

    @property
    def projection_is_current(self) -> bool:
        return self.projection_checkpoint == self.as_of_event_id
