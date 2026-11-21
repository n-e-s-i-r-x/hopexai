from __future__ import annotations

import json
import secrets
import time
import uuid
from typing import Any, AsyncIterator

from .errors import NoEligibleCandidate, PublicAPIError, UpstreamFailure
from .requests import CanonicalRequest, PUBLIC_MODEL
from .router import GatewayRouter
from .types import Candidate, CanonicalDelta, CanonicalResponse


def new_chat_id() -> str:
    return "chatcmpl_" + secrets.token_urlsafe(18)


def new_response_id() -> str:
    return "resp_" + secrets.token_urlsafe(18)


def _chat_usage(usage: dict[str, int]) -> dict[str, int]:
    return dict(usage)


def _responses_usage(usage: dict[str, int]) -> dict[str, int]:
    if not usage:
        return {}
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)),
    }


def assistant_history_message(response: CanonicalResponse) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": response.content}
    if response.tool_calls:
        message["tool_calls"] = response.tool_calls
    if response.reasoning_summary:
        message["reasoning_summary"] = response.reasoning_summary
    if response.refusal:
        message["refusal"] = response.refusal
    return message


def chat_completion_body(response: CanonicalResponse, request_id: str) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": response.content}
    if response.tool_calls:
        message["tool_calls"] = response.tool_calls
    if response.reasoning_summary:
        message["reasoning_summary"] = response.reasoning_summary
    if response.refusal:
        message["refusal"] = response.refusal
    body: dict[str, Any] = {
        "id": new_chat_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": PUBLIC_MODEL,
        "choices": [{"index": 0, "message": message, "finish_reason": response.finish_reason}],
    }
    if response.usage:
        body["usage"] = _chat_usage(response.usage)
    return body


def responses_body(
    response: CanonicalResponse,
    response_id: str,
    *,
    stored: bool,
) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    if response.content is not None or response.refusal is not None:
        content_items = []
        if response.content is not None:
            content_items.append({"type": "output_text", "text": response.content, "annotations": []})
        if response.refusal is not None:
            content_items.append({"type": "refusal", "refusal": response.refusal})
        output.append({
            "id": "msg_" + secrets.token_urlsafe(10),
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": content_items,
        })
    for call in response.tool_calls:
        function = call.get("function", {})
        output.append({
            "id": "fc_" + secrets.token_urlsafe(10),
            "type": "function_call",
            "status": "completed",
            "call_id": call.get("id", ""),
            "name": function.get("name", ""),
            "arguments": function.get("arguments", "{}"),
        })
    if response.reasoning_summary:
        output.append({
            "id": "rs_" + secrets.token_urlsafe(10),
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": response.reasoning_summary}],
        })
    usage = _responses_usage(response.usage)
    body: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "completed_at": int(time.time()),
        "status": "incomplete" if response.finish_reason == "length" else "completed",
        "error": None,
        "incomplete_details": {"reason": "max_output_tokens"} if response.finish_reason == "length" else None,
        "model": PUBLIC_MODEL,
        "output": output,
        "output_text": response.content or "",
        "store": stored,
    }
    if usage:
        body["usage"] = usage
    return body


def _sse(data: dict[str, Any] | str, event: str | None = None) -> bytes:
    text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    prefix = f"event: {event}\n" if event else ""
    return (prefix + "data: " + text + "\n\n").encode("utf-8")


def _chat_chunk(
    completion_id: str,
    created: int,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": PUBLIC_MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage:
        chunk["usage"] = _chat_usage(usage)
    return chunk


async def chat_stream_body(
    router: GatewayRouter,
    config,
    request: CanonicalRequest,
    request_id: str,
    api_key_id: str,
    affinity_key: str | None,
    *,
    disconnected=None,
    on_candidate=None,
) -> AsyncIterator[bytes]:
    completion_id = new_chat_id()
    created = int(time.time())
    role_sent = False
    last_usage: dict[str, int] = {}
    last_usage_sent: dict[str, int] = {}
    stream_iterator = router.stream(config, request, request_id, api_key_id, affinity_key)
    try:
        async for candidate, delta in stream_iterator:
            if disconnected is not None and await disconnected():
                return
            if on_candidate is not None:
                await on_candidate(candidate)
                on_candidate = None
            if delta.usage:
                last_usage = delta.usage
            public_delta: dict[str, Any] = {}
            if delta.content is not None:
                public_delta["content"] = delta.content
            if delta.reasoning_summary is not None:
                public_delta["reasoning_summary"] = delta.reasoning_summary
            if delta.tool_calls:
                public_delta["tool_calls"] = delta.tool_calls
            if delta.refusal:
                public_delta["refusal"] = delta.refusal
            if delta.finish_reason is not None:
                if not role_sent:
                    yield _sse(_chat_chunk(completion_id, created, {"role": "assistant"}))
                    role_sent = True
                yield _sse(_chat_chunk(completion_id, created, public_delta, delta.finish_reason, delta.usage or None))
                if delta.usage:
                    last_usage_sent = dict(delta.usage)
            elif public_delta:
                if not role_sent:
                    yield _sse(_chat_chunk(completion_id, created, {"role": "assistant"}))
                    role_sent = True
                yield _sse(_chat_chunk(completion_id, created, public_delta, usage=delta.usage or None))
                if delta.usage:
                    last_usage_sent = dict(delta.usage)
        if not role_sent:
            yield _sse(_chat_chunk(completion_id, created, {"role": "assistant"}))
        if last_usage and last_usage_sent != last_usage:
            yield _sse({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": PUBLIC_MODEL,
                "choices": [],
                "usage": _chat_usage(last_usage),
            })
        yield _sse("[DONE]")
    except PublicAPIError as exc:
        yield _sse(exc.as_payload())
        yield _sse("[DONE]")
    except NoEligibleCandidate as exc:
        error = PublicAPIError(exc.status_code, exc.message, "service_unavailable_error", exc.code)
        yield _sse(error.as_payload())
        yield _sse("[DONE]")
    except Exception:
        error = PublicAPIError(502, "The HopeXAI stream could not be completed.", "server_error", "stream_failed")
        yield _sse(error.as_payload())
        yield _sse("[DONE]")
    finally:
        await stream_iterator.aclose()


async def responses_stream_body(
    router: GatewayRouter,
    config,
    request: CanonicalRequest,
    response_id: str,
    request_id: str,
    api_key_id: str,
    affinity_key: str | None,
    *,
    disconnected=None,
    on_candidate=None,
    on_complete=None,
) -> AsyncIterator[bytes]:
    sequence = 0
    created = int(time.time())
    output: list[dict[str, Any]] = []
    text_item_id = "msg_" + secrets.token_urlsafe(10)
    text_output_index = 0
    text_item: dict[str, Any] | None = None
    text_started = False
    text_content_started = False
    text_parts: list[str] = []
    tool_items: dict[int, dict[str, Any]] = {}
    reasoning_parts: list[str] = []
    reasoning_started = False
    reasoning_output_index: int | None = None
    usage: dict[str, int] = {}
    finish_reason = "stop"

    def event(kind: str, payload: dict[str, Any]) -> bytes:
        nonlocal sequence
        sequence += 1
        value = {"type": kind, "sequence_number": sequence, **payload}
        return _sse(value, kind)

    created_response = {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": "in_progress",
        "model": PUBLIC_MODEL,
        "output": [],
    }
    yield event("response.created", {"response": created_response})
    yield event("response.in_progress", {"response": created_response})
    stream_iterator = router.stream(config, request, request_id, api_key_id, affinity_key)
    try:
        async for candidate, delta in stream_iterator:
            if disconnected is not None and await disconnected():
                return
            if on_candidate is not None:
                await on_candidate(candidate)
                on_candidate = None
            if delta.usage:
                usage = delta.usage
            if delta.content is not None:
                text_parts.append(delta.content)
                if not text_started:
                    text_started = True
                    text_output_index = len(output)
                    text_item = {"id": text_item_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []}
                    output.append(text_item)
                    yield event("response.output_item.added", {"output_index": text_output_index, "item": text_item})
                if not text_content_started:
                    text_content_started = True
                    content_part = {"type": "output_text", "text": "", "annotations": []}
                    assert text_item is not None
                    text_item["content"].append(content_part)
                    yield event("response.content_part.added", {"item_id": text_item_id, "output_index": text_output_index, "content_index": 0, "part": content_part})
                yield event("response.output_text.delta", {"item_id": text_item_id, "output_index": text_output_index, "content_index": 0, "delta": delta.content})
            if delta.reasoning_summary is not None:
                reasoning_parts.append(delta.reasoning_summary)
                if not reasoning_started:
                    reasoning_started = True
                    index = len(output)
                    reasoning_output_index = index
                    reasoning_item = {"id": "rs_" + secrets.token_urlsafe(10), "type": "reasoning", "summary": []}
                    output.append(reasoning_item)
                    yield event("response.output_item.added", {"output_index": index, "item": reasoning_item})
                    yield event("response.reasoning_summary_part.added", {"item_id": reasoning_item["id"], "output_index": index, "summary_index": 0, "part": {"type": "summary_text", "text": ""}})
                yield event("response.reasoning_summary_text.delta", {"output_index": reasoning_output_index, "summary_index": 0, "delta": delta.reasoning_summary})
            for tool in delta.tool_calls:
                index = int(tool.get("index", 0))
                item = tool_items.get(index)
                function = tool.get("function", {}) if isinstance(tool.get("function"), dict) else {}
                if item is None:
                    item = {
                        "id": "fc_" + secrets.token_urlsafe(10),
                        "type": "function_call",
                        "status": "in_progress",
                        "call_id": tool.get("id") or "call_" + secrets.token_urlsafe(8),
                        "name": function.get("name", ""),
                        "arguments": "",
                    }
                    tool_items[index] = item
                    output_index = len(output)
                    item["_output_index"] = output_index
                    output.append(item)
                    public_item = {key: value for key, value in item.items() if not key.startswith("_")}
                    yield event("response.output_item.added", {"output_index": output_index, "item": public_item})
                if function.get("name") and not item.get("name"):
                    item["name"] = function["name"]
                arguments = function.get("arguments")
                if isinstance(arguments, str) and arguments:
                    item["arguments"] += arguments
                    yield event("response.function_call_arguments.delta", {"item_id": item["id"], "output_index": item["_output_index"], "delta": arguments})
            if delta.finish_reason:
                finish_reason = delta.finish_reason

        if text_started and text_item is not None:
            text_item["status"] = "completed"
            if text_content_started:
                text_item["content"][0]["text"] = "".join(text_parts)
                yield event("response.output_text.done", {"item_id": text_item_id, "output_index": text_output_index, "content_index": 0, "text": "".join(text_parts)})
                yield event("response.content_part.done", {"item_id": text_item_id, "output_index": text_output_index, "content_index": 0, "part": text_item["content"][0]})
            yield event("response.output_item.done", {"output_index": text_output_index, "item": text_item})
        for item in output:
            if item.get("type") == "function_call":
                item["status"] = "completed"
                output_index = item.pop("_output_index")
                yield event("response.function_call_arguments.done", {"item_id": item["id"], "output_index": output_index, "arguments": item["arguments"]})
                yield event("response.output_item.done", {"output_index": output_index, "item": item})
            elif item.get("type") == "reasoning":
                item["summary"] = [{"type": "summary_text", "text": "".join(reasoning_parts)}]
                yield event("response.reasoning_summary_text.done", {"output_index": output.index(item), "summary_index": 0, "text": "".join(reasoning_parts)})
                yield event("response.output_item.done", {"output_index": output.index(item), "item": item})
        canonical = CanonicalResponse(
            content="".join(text_parts) if text_started else None,
            tool_calls=[{
                "id": item["call_id"],
                "type": "function",
                "function": {"name": item.get("name", ""), "arguments": item.get("arguments", "{}")},
            } for item in tool_items.values()],
            reasoning_summary="".join(reasoning_parts) if reasoning_parts else None,
            finish_reason="length" if finish_reason == "length" else ("tool_calls" if tool_items else finish_reason),
            usage=usage,
        )
        completed_response = {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "completed_at": int(time.time()),
            "status": "incomplete" if canonical.finish_reason == "length" else "completed",
            "incomplete_details": {"reason": "max_output_tokens"} if canonical.finish_reason == "length" else None,
            "model": PUBLIC_MODEL,
            "output": output,
            "output_text": canonical.content or "",
            "usage": _responses_usage(usage),
        }
        if on_complete is not None:
            await on_complete(canonical, completed_response)
        terminal_event = "response.incomplete" if canonical.finish_reason == "length" else "response.completed"
        yield event(terminal_event, {"response": completed_response})
    except PublicAPIError as exc:
        yield event("error", exc.as_payload())
    except NoEligibleCandidate as exc:
        error = PublicAPIError(exc.status_code, exc.message, "service_unavailable_error", exc.code)
        yield event("error", error.as_payload())
    except Exception:
        error = PublicAPIError(502, "The HopeXAI response stream could not be completed.", "server_error", "stream_failed")
        yield event("error", error.as_payload())
    finally:
        await stream_iterator.aclose()
