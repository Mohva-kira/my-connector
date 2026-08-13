"""Tool Architecture foundation (Personal Executive AI Assistant pivot,
Phase 8). Each tool wraps one existing read/query capability behind a
uniform, simple contract: a name, a description (for a future Agent/LLM
to pick from), and a synchronous ``run(...)`` method with typed
keyword arguments and a plain-dict/list-of-dicts return value.

No shared base class — composition, not inheritance
(context/ai-workflow-rules.md, coding style). ``Tool`` is a
``typing.Protocol`` used purely for static typing/documentation of the
shape every tool follows; nothing here is instantiated or dispatched via
reflection.

Phase 9 (Agent) will call these tools directly and sequentially — no
registry, no dynamic lookup by name, to keep execution deterministic and
easy to read in one file.
"""

from __future__ import annotations

from typing import Any, Protocol


class Tool(Protocol):
    name: str
    description: str

    def run(self, **kwargs: Any) -> Any: ...
