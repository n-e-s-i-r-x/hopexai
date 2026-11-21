from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator

import httpx


@dataclass(slots=True)
class SSEMessage:
    event: str | None
    data: str
    event_id: str | None = None


async def iter_sse(response: httpx.Response) -> AsyncIterator[SSEMessage]:
    """Read and dispatch one SSE event at a time without buffering the body."""
    data_lines: list[str] = []
    event: str | None = None
    event_id: str | None = None
    async for line in response.aiter_lines():
        if line == "":
            if data_lines or event is not None:
                yield SSEMessage(event=event, data="\n".join(data_lines), event_id=event_id)
            data_lines = []
            event = None
            event_id = None
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "data":
            data_lines.append(value)
        elif field == "event":
            event = value
        elif field == "id":
            event_id = value
    if data_lines or event is not None:
        yield SSEMessage(event=event, data="\n".join(data_lines), event_id=event_id)
