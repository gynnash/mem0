"""Stable envelope for agent-facing memory tool results."""

from enum import Enum
from typing import Generic, Optional, TypeVar

from pydantic import Field

from mem0.v3.contracts.base import FrozenContract, NonEmptyStr
from mem0.v3.contracts.versions import TOOL_CONTRACT_VERSION


class ToolStatus(str, Enum):
    SUCCESS = "success"
    NO_RESULT = "no_result"
    FAILED = "failed"


class ToolDiagnostics(FrozenContract):
    tool: NonEmptyStr
    version: int = Field(default=TOOL_CONTRACT_VERSION, ge=1)
    source_of_truth: NonEmptyStr = "mysql"
    projection_lag_ms: int = Field(default=0, ge=0)
    as_of_event_id: int = Field(ge=0)


ToolData = TypeVar("ToolData")


class ToolEnvelope(FrozenContract, Generic[ToolData]):
    status: ToolStatus
    data: Optional[ToolData] = None
    evidence_refs: tuple[NonEmptyStr, ...] = ()
    warnings: tuple[NonEmptyStr, ...] = ()
    diagnostics: ToolDiagnostics
