import importlib.metadata
from typing import TYPE_CHECKING, Any

try:
    __version__ = importlib.metadata.version("mem0pin")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["AsyncMemory", "AsyncMemoryClient", "Memory", "MemoryClient"]

if TYPE_CHECKING:
    from mem0.client.main import AsyncMemoryClient, MemoryClient
    from mem0.memory.main import AsyncMemory, Memory


def __getattr__(name: str) -> Any:
    """Load legacy clients only when requested so mem0.v3 stays infrastructure-free."""

    if name in {"MemoryClient", "AsyncMemoryClient"}:
        from mem0.client.main import AsyncMemoryClient, MemoryClient

        return {"MemoryClient": MemoryClient, "AsyncMemoryClient": AsyncMemoryClient}[name]
    if name in {"Memory", "AsyncMemory"}:
        from mem0.memory.main import AsyncMemory, Memory

        return {"Memory": Memory, "AsyncMemory": AsyncMemory}[name]
    raise AttributeError(f"module 'mem0' has no attribute {name!r}")
