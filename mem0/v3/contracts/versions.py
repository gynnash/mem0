"""Version identifiers shared by mem0/v3 and Summora adapters."""

from pydantic import Field

from mem0.v3.contracts.base import FrozenContract


KERNEL_API_VERSION = 1
CHANGESET_SCHEMA_VERSION = 1
TOOL_CONTRACT_VERSION = 1
MINIMUM_STORAGE_SCHEMA_VERSION = 2


class ContractVersions(FrozenContract):
    kernel_api_version: int = Field(ge=1)
    changeset_schema_version: int = Field(ge=1)
    tool_contract_version: int = Field(ge=1)
    minimum_storage_schema_version: int = Field(ge=1)


CURRENT_CONTRACT_VERSIONS = ContractVersions(
    kernel_api_version=KERNEL_API_VERSION,
    changeset_schema_version=CHANGESET_SCHEMA_VERSION,
    tool_contract_version=TOOL_CONTRACT_VERSION,
    minimum_storage_schema_version=MINIMUM_STORAGE_SCHEMA_VERSION,
)
