from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

from .requests import CanonicalRequest
from .sse import iter_sse
from .types import Candidate, CanonicalDelta, CanonicalResponse


@dataclass(slots=True)
class RequestSpec:
    endpoint_kind: str
    body: dict[str, Any]
    headers: dict[str, str]


class AdapterError(ValueError):
    pass


def _is_string(value: Any) -> bool:
    return isinstance(value, str)


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
        return result if result >= 0 else None
    except (TypeError, ValueError):
        return None


def normalize_usage(value: Any, *, input_name: str = "prompt_tokens", output_name: str = "completion_tokens") -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    input_tokens = _safe_int(value.get("prompt_tokens", value.get("input_tokens", value.get("input_tokens_details", {}).get("total_tokens") if isinstance(value.get("input_tokens_details"), dict) else None)))
    output_tokens = _safe_int(value.get("completion_tokens", value.get("output_tokens", value.get("output_tokens_details", {}).get("total_tokens") if isinstance(value.get("output_tokens_details"), dict) else None)))
    if input_tokens is None and input_name in value:
        input_tokens = _safe_int(value[input_name])
    if output_tokens is None and output_name in value:
        output_tokens = _safe_int(value[output_name])
    result: dict[str, int] = {}
    if input_tokens is not None:
        result["prompt_tokens"] = input_tokens
    if output_tokens is not None:
        result["completion_tokens"] = output_tokens
    if input_tokens is not None or output_tokens is not None:
        result["total_tokens"] = (input_tokens or 0) + (output_tokens or 0)
    return result


def _join_text_parts(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if content is None:
        return None
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in {"text", "output_text", "input_text"} and isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "".join(parts) if parts else None
    return None


def _openai_messages(request: CanonicalRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": request.system_text()}]
    for original in request.provider_messages():
        item = dict(original)
        # Only the text/content and OpenAI-compatible tool history are forwarded.
        # Public summaries remain available in encrypted HopeXAI conversation state;
        # the system_text method includes them for provider-switch continuity.
        item.pop("reasoning_summary", None)
        messages.append(item)
    return messages


def _responses_input(request: CanonicalRequest) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in request.provider_messages():
        role = message.get("role")
        if role == "tool":
            result.append({
                "type": "function_call_output",
                "call_id": message.get("tool_call_id", ""),
                "output": message.get("content", ""),
            })
            continue
        for call in message.get("tool_calls", []) if role == "assistant" else []:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            result.append({
                "type": "function_call",
                "call_id": call.get("id", "") if isinstance(call, dict) else "",
                "name": function.get("name", "") if isinstance(function, dict) else "",
                "arguments": function.get("arguments", "{}") if isinstance(function, dict) else "{}",
            })
        content = message.get("content")
        if content is None:
            continue
        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue
        parts: list[dict[str, Any]] = []
        for part in content if isinstance(content, list) else []:
            if isinstance(part, str):
                parts.append({"type": "input_text", "text": part})
            elif not isinstance(part, dict):
                continue
            elif part.get("type") == "image_url":
                image = part.get("image_url")
                image_url = image.get("url") if isinstance(image, dict) else image
                if image_url:
                    parts.append({"type": "input_image", "image_url": image_url})
            elif part.get("type") in {"text", "input_text"}:
                parts.append({"type": "input_text", "text": part.get("text", "")})
        result.append({"role": role, "content": parts})
    return result


def _tool_schema_for_responses(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    converted = []
    for tool in tools:
        function = tool.get("function", {})
        converted.append({
            "type": "function",
            "name": function.get("name"),
            "description": function.get("description", ""),
            "parameters": function.get("parameters", {"type": "object", "properties": {}}),
            "strict": bool(function.get("strict", False)),
        })
    return converted


def _response_format_to_text(response_format: Any) -> dict[str, Any] | None:
    if not isinstance(response_format, dict):
        return None
    fmt_type = response_format.get("type")
    if fmt_type == "json_object":
        return {"format": {"type": "json_object"}}
    if fmt_type == "json_schema":
        schema = response_format.get("json_schema", response_format)
        if isinstance(schema, dict) and "schema" in schema:
            return {"format": {
                "type": "json_schema",
                "name": schema.get("name", "hopexai_schema"),
                "strict": schema.get("strict", True),
                "schema": schema["schema"],
            }}
        return {"format": {"type": "json_schema", "name": "hopexai_schema", "strict": True, "schema": schema}}
    return None


def _tool_choice_for_chat(choice: Any) -> Any:
    if isinstance(choice, dict) and choice.get("type") == "function" and isinstance(choice.get("name"), str):
        return {"type": "function", "function": {"name": choice["name"]}}
    return choice


def _response_format_to_chat(response_format: Any) -> dict[str, Any] | None:
    if not isinstance(response_format, dict):
        return None
    fmt_type = response_format.get("type")
    if fmt_type == "json_object":
        return {"type": "json_object"}
    if fmt_type != "json_schema":
        return None
    schema_info = response_format.get("json_schema", response_format)
    if not isinstance(schema_info, dict):
        schema_info = {"schema": schema_info}
    schema = schema_info.get("schema", schema_info)
    return {"type": "json_schema", "json_schema": {
        "name": schema_info.get("name", "hopexai_schema"),
        "strict": schema_info.get("strict", True),
        "schema": schema,
    }}


def _reasoning_effort(candidate: Candidate, request: CanonicalRequest) -> str | None:
    # A positive reasoning request without an explicit level still needs an
    # upstream setting. Use the gateway's neutral medium default rather than
    # silently accepting an enable flag that has no effect.
    effort = request.reasoning_effort or ("medium" if request.reasoning_enabled is True else None)
    if not effort:
        return None
    mapping = candidate.provider.reasoning.effort_map
    return mapping.get(effort, effort)


class BaseAdapter:
    format_name = ""

    def endpoint_kind(self, candidate: Candidate, public_kind: str) -> str:
        raise NotImplementedError

    def build(self, candidate: Candidate, request: CanonicalRequest, public_kind: str, stream: bool) -> RequestSpec:
        raise NotImplementedError

    def parse_response(self, data: Any, request: CanonicalRequest) -> CanonicalResponse:
        raise NotImplementedError

    async def stream_deltas(self, response: httpx.Response, request: CanonicalRequest) -> AsyncIterator[CanonicalDelta]:
        raise NotImplementedError


class OpenAIChatAdapter(BaseAdapter):
    format_name = "openai_chat"

    def endpoint_kind(self, candidate: Candidate, public_kind: str) -> str:
        if candidate.endpoints.get("chat_completions"):
            return "chat_completions"
        if candidate.endpoints.get("responses"):
            return "responses"
        return "chat_completions"

    def build(self, candidate: Candidate, request: CanonicalRequest, public_kind: str, stream: bool) -> RequestSpec:
        body: dict[str, Any] = {
            "model": candidate.model_id,
            "messages": _openai_messages(request),
            "stream": stream,
        }
        if request.max_output_tokens is not None:
            body[candidate.max_output_parameter or "max_tokens"] = request.max_output_tokens
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.stop is not None:
            body["stop"] = request.stop
        if request.tools:
            body["tools"] = request.tools
        if request.tool_choice is not None:
            body["tool_choice"] = _tool_choice_for_chat(request.tool_choice)
        if request.response_format is not None:
            body["response_format"] = (
                _response_format_to_chat(request.response_format)
                if request.kind == "response"
                else request.response_format
            )
        if request.extra_params:
            body.update(request.extra_params)
        if candidate.capabilities.get("reasoning") is True:
            reasoning_config = candidate.provider.reasoning
            if request.reasoning_effort or request.reasoning_enabled is True:
                mapped = _reasoning_effort(candidate, request)
                if reasoning_config.effort_parameter and mapped:
                    body[reasoning_config.effort_parameter] = mapped
            elif request.reasoning_enabled is False and reasoning_config.effort_parameter and reasoning_config.disabled_effort_value:
                body[reasoning_config.effort_parameter] = reasoning_config.disabled_effort_value
            if request.reasoning_enabled is True and reasoning_config.enabled_parameter:
                body[reasoning_config.enabled_parameter] = True
        if request.include_reasoning and candidate.capabilities.get("reasoning_summary") is True:
            parameter = candidate.provider.reasoning.summary_parameter
            if parameter:
                body[parameter] = True
        return RequestSpec(self.endpoint_kind(candidate, public_kind), body, {})

    def parse_response(self, data: Any, request: CanonicalRequest) -> CanonicalResponse:
        if not isinstance(data, dict):
            raise AdapterError("upstream response was not a JSON object")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise AdapterError("upstream response did not contain a completion choice")
        choice = choices[0]
        message = choice.get("message", choice.get("delta", {}))
        if not isinstance(message, dict):
            message = {}
        content = _join_text_parts(message.get("content"))
        calls = message.get("tool_calls", [])
        normalized_calls: list[dict[str, Any]] = []
        if isinstance(calls, list):
            for index, call in enumerate(calls):
                if not isinstance(call, dict):
                    continue
                function = call.get("function", {})
                if not isinstance(function, dict):
                    function = {}
                normalized_calls.append({
                    "id": call.get("id") or f"call_{index}",
                    "type": "function",
                    "function": {
                        "name": function.get("name", ""),
                        "arguments": function.get("arguments", "{}"),
                    },
                })
        summary = None
        if request.include_reasoning and candidate_summary_allowed(request):
            summary = _extract_reasoning_summary(message, data)
        finish = choice.get("finish_reason") or ("tool_calls" if normalized_calls else "stop")
        return CanonicalResponse(
            content=content,
            tool_calls=normalized_calls,
            reasoning_summary=summary,
            finish_reason=_normalize_finish(finish),
            usage=normalize_usage(data.get("usage")),
            refusal=message.get("refusal") if isinstance(message.get("refusal"), str) else None,
        )

    async def stream_deltas(self, response: httpx.Response, request: CanonicalRequest) -> AsyncIterator[CanonicalDelta]:
        finish_seen = False
        async for event in iter_sse(response):
            if event.data == "[DONE]":
                if not finish_seen:
                    yield CanonicalDelta(finish_reason="stop")
                return
            try:
                data = json.loads(event.data)
            except (ValueError, TypeError):
                continue
            if not isinstance(data, dict):
                continue
            if event.event == "error" or "error" in data or data.get("type") == "error":
                raise AdapterError("upstream Chat Completions stream reported an error")
            choices = data.get("choices", [])
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                choice = choices[0]
                delta = choice.get("delta", {})
                if not isinstance(delta, dict):
                    delta = {}
                tool_calls = []
                if isinstance(delta.get("tool_calls"), list):
                    for call in delta["tool_calls"]:
                        if not isinstance(call, dict):
                            continue
                        function = call.get("function", {})
                        tool_calls.append({
                            "index": call.get("index", 0),
                            "id": call.get("id"),
                            "type": call.get("type", "function"),
                            "function": function if isinstance(function, dict) else {},
                        })
                summary = None
                if request.include_reasoning and isinstance(delta.get("reasoning_summary"), str):
                    summary = delta["reasoning_summary"]
                finish = choice.get("finish_reason")
                if finish is not None:
                    finish_seen = True
                yield CanonicalDelta(
                    content=_join_text_parts(delta.get("content")),
                    tool_calls=tool_calls,
                    reasoning_summary=summary,
                    finish_reason=_normalize_finish(finish) if finish else None,
                    usage=normalize_usage(data.get("usage")),
                    refusal=delta.get("refusal") if isinstance(delta.get("refusal"), str) else None,
                )
            elif data.get("usage"):
                yield CanonicalDelta(usage=normalize_usage(data.get("usage")))


class OpenAIResponsesAdapter(BaseAdapter):
    format_name = "openai_responses"

    def endpoint_kind(self, candidate: Candidate, public_kind: str) -> str:
        if candidate.endpoints.get("responses"):
            return "responses"
        if candidate.endpoints.get("chat_completions"):
            return "chat_completions"
        return "responses"

    def build(self, candidate: Candidate, request: CanonicalRequest, public_kind: str, stream: bool) -> RequestSpec:
        endpoint = self.endpoint_kind(candidate, public_kind)
        if endpoint == "chat_completions":
            return OpenAIChatAdapter().build(candidate, request, public_kind, stream)
        body: dict[str, Any] = {
            "model": candidate.model_id,
            "instructions": request.system_text(),
            "input": _responses_input(request),
            "stream": stream,
        }
        if request.max_output_tokens is not None:
            body[candidate.max_output_parameter or "max_output_tokens"] = request.max_output_tokens
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.tools:
            body["tools"] = _tool_schema_for_responses(request.tools)
        if request.tool_choice is not None:
            body["tool_choice"] = request.tool_choice
        text_format = _response_format_to_text(request.response_format)
        if text_format:
            body["text"] = text_format
        if candidate.capabilities.get("reasoning") is True:
            if request.reasoning_effort or request.reasoning_enabled is True:
                effort = _reasoning_effort(candidate, request)
                if effort:
                    body["reasoning"] = {"effort": effort}
            elif request.reasoning_enabled is False and candidate.provider.reasoning.disabled_effort_value:
                body["reasoning"] = {"effort": candidate.provider.reasoning.disabled_effort_value}
            if request.reasoning_enabled is True and candidate.provider.reasoning.enabled_parameter:
                body[candidate.provider.reasoning.enabled_parameter] = True
            if request.include_reasoning and candidate.capabilities.get("reasoning_summary") is True:
                summary_parameter = candidate.provider.reasoning.summary_parameter
                if summary_parameter:
                    body[summary_parameter] = True
                else:
                    body.setdefault("reasoning", {})["summary"] = "auto"
        return RequestSpec("responses", body, {})

    def parse_response(self, data: Any, request: CanonicalRequest) -> CanonicalResponse:
        if not isinstance(data, dict):
            raise AdapterError("upstream Responses body was not a JSON object")
        if isinstance(data.get("choices"), list):
            return OpenAIChatAdapter().parse_response(data, request)
        output = data.get("output")
        if not isinstance(output, list):
            raise AdapterError("upstream Responses body did not contain output items")
        text_parts: list[str] = []
        calls: list[dict[str, Any]] = []
        summaries: list[str] = []
        refusal = None
        for item in output:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "message":
                for part in item.get("content", []) if isinstance(item.get("content"), list) else []:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                        text_parts.append(part["text"])
                    elif part.get("type") == "refusal" and isinstance(part.get("refusal"), str):
                        refusal = part["refusal"]
            elif kind in {"function_call", "tool_call"}:
                calls.append({
                    "id": item.get("call_id", item.get("id", f"call_{len(calls)}")),
                    "type": "function",
                    "function": {"name": item.get("name", ""), "arguments": item.get("arguments", "{}")},
                })
            elif kind == "reasoning" and request.include_reasoning:
                summary = item.get("summary")
                if isinstance(summary, list):
                    summaries.extend(part.get("text", "") for part in summary if isinstance(part, dict) and isinstance(part.get("text"), str))
                elif isinstance(summary, str):
                    summaries.append(summary)
        if request.include_reasoning:
            explicit = data.get("reasoning_summary")
            if isinstance(explicit, str):
                summaries.append(explicit)
        status = data.get("status")
        finish = "length" if status == "incomplete" else ("tool_calls" if calls else "stop")
        return CanonicalResponse(
            content="".join(text_parts) if text_parts else None,
            tool_calls=calls,
            reasoning_summary="\n".join(summaries) if summaries else None,
            finish_reason=finish,
            usage=normalize_usage(data.get("usage")),
            refusal=refusal,
        )

    async def stream_deltas(self, response: httpx.Response, request: CanonicalRequest) -> AsyncIterator[CanonicalDelta]:
        if self._response_is_chat_compat(response):
            async for delta in OpenAIChatAdapter().stream_deltas(response, request):
                yield delta
            return
        tool_ids: dict[int, str] = {}
        tool_names: dict[int, str] = {}
        async for event in iter_sse(response):
            if not event.data:
                continue
            try:
                data = json.loads(event.data)
            except (ValueError, TypeError):
                continue
            if not isinstance(data, dict):
                continue
            kind = data.get("type", event.event)
            if event.event == "error" or "error" in data or kind == "error":
                raise AdapterError("upstream Responses stream reported an error")
            if kind == "response.output_item.added":
                item = data.get("item", {})
                index = data.get("output_index", 0)
                if isinstance(item, dict) and item.get("type") == "function_call":
                    tool_ids[index] = item.get("call_id", item.get("id", f"call_{index}"))
                    tool_names[index] = item.get("name", "")
                    yield CanonicalDelta(tool_calls=[{"index": index, "id": tool_ids[index], "type": "function", "function": {"name": tool_names[index], "arguments": ""}}])
            elif kind == "response.output_text.delta" and isinstance(data.get("delta"), str):
                yield CanonicalDelta(content=data["delta"])
            elif kind in {"response.reasoning_summary_text.delta", "response.reasoning_summary.delta"} and request.include_reasoning:
                if isinstance(data.get("delta"), str):
                    yield CanonicalDelta(reasoning_summary=data["delta"])
            elif kind == "response.function_call_arguments.delta" and isinstance(data.get("delta"), str):
                index = data.get("output_index", 0)
                yield CanonicalDelta(tool_calls=[{"index": index, "function": {"arguments": data["delta"]}}])
            elif kind == "response.completed":
                response_obj = data.get("response", {})
                usage = normalize_usage(response_obj.get("usage") if isinstance(response_obj, dict) else None)
                yield CanonicalDelta(finish_reason="tool_calls" if tool_ids else "stop", usage=usage)
            elif kind in {"response.failed", "error"}:
                raise AdapterError("upstream Responses stream failed")

    @staticmethod
    def _response_is_chat_compat(response: httpx.Response) -> bool:
        # Discovery determines this adapter's endpoint. A chat-compatible SSE
        # payload is still accepted if an upstream aliases /responses to chat.
        return False


class AnthropicAdapter(BaseAdapter):
    format_name = "anthropic"

    def endpoint_kind(self, candidate: Candidate, public_kind: str) -> str:
        return "messages"

    def build(self, candidate: Candidate, request: CanonicalRequest, public_kind: str, stream: bool) -> RequestSpec:
        messages: list[dict[str, Any]] = []
        for message in request.provider_messages():
            role = message.get("role")
            if role == "system" or role == "developer":
                continue
            if role == "tool":
                messages.append({"role": "user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id", ""),
                    "content": _join_text_parts(message.get("content")) or "",
                }]})
                continue
            content = message.get("content")
            blocks: list[dict[str, Any]] = []
            if isinstance(content, str):
                blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, str):
                        blocks.append({"type": "text", "text": part})
                    elif isinstance(part, dict) and part.get("type") in {"text", "input_text"}:
                        blocks.append({"type": "text", "text": part.get("text", "")})
                    elif isinstance(part, dict) and part.get("type") == "image_url":
                        image = part.get("image_url")
                        image_url = image.get("url") if isinstance(image, dict) else image
                        if isinstance(image_url, str):
                            if image_url.startswith("data:image/"):
                                try:
                                    header, encoded = image_url.split(",", 1)
                                    media_type = header[5:].split(";", 1)[0]
                                    blocks.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": encoded}})
                                except ValueError:
                                    raise AdapterError("invalid data image")
                            else:
                                blocks.append({"type": "image", "source": {"type": "url", "url": image_url}})
            for call in message.get("tool_calls", []) if role == "assistant" else []:
                function = call.get("function", {}) if isinstance(call, dict) else {}
                args = function.get("arguments", "{}") if isinstance(function, dict) else "{}"
                try:
                    parsed_args = json.loads(args) if isinstance(args, str) else {}
                except ValueError:
                    parsed_args = {"_raw_arguments": str(args)}
                blocks.append({
                    "type": "tool_use",
                    "id": call.get("id", "") if isinstance(call, dict) else "",
                    "name": function.get("name", "") if isinstance(function, dict) else "",
                    "input": parsed_args if isinstance(parsed_args, dict) else {"value": parsed_args},
                })
            messages.append({"role": "assistant" if role == "assistant" else "user", "content": blocks or [{"type": "text", "text": ""}]})

        max_output_parameter = candidate.max_output_parameter or "max_tokens"
        body: dict[str, Any] = {
            "model": candidate.model_id,
            "system": request.system_text(),
            "messages": messages,
            "stream": stream,
        }
        body[max_output_parameter] = request.max_output_tokens or 1024
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if isinstance(request.stop, list):
            body["stop_sequences"] = request.stop
        elif isinstance(request.stop, str):
            body["stop_sequences"] = [request.stop]
        if request.tools:
            body["tools"] = []
            for item in request.tools:
                function = item["function"]
                body["tools"].append({
                    "name": function["name"],
                    "description": function.get("description", ""),
                    "input_schema": function.get("parameters", {"type": "object", "properties": {}}),
                })
        if request.tools and request.tool_choice is not None:
            choice = request.tool_choice
            if choice == "auto":
                body["tool_choice"] = {"type": "auto"}
            elif choice == "required":
                body["tool_choice"] = {"type": "any"}
            elif choice == "none":
                body.pop("tools", None)
            elif isinstance(choice, dict) and choice.get("type") == "function" and isinstance(choice.get("name"), str):
                body["tool_choice"] = {"type": "tool", "name": choice["name"]}
        if request.response_format is not None:
            fmt = request.response_format
            if fmt.get("type") == "json_object":
                body["output_config"] = {"format": {"type": "json"}}
            elif fmt.get("type") == "json_schema":
                schema = fmt.get("json_schema", fmt)
                body["output_config"] = {"format": {
                    "type": "json_schema",
                    "schema": schema.get("schema", schema),
                }}
        if (request.reasoning_enabled is True or request.reasoning_effort) and candidate.capabilities.get("reasoning") is True:
            budget_map = {"minimal": 1024, "low": 2048, "medium": 4096, "high": 8192, "extra_high": 12000, "maximum": 16000}
            max_tokens = int(body[max_output_parameter])
            if max_tokens > 1:
                budget = budget_map.get(request.reasoning_effort or "medium", 4096)
                body["thinking"] = {"type": "enabled", "budget_tokens": min(budget, max_tokens - 1)}
                body.pop("temperature", None)
        headers = {"anthropic-version": "2023-06-01"}
        return RequestSpec("messages", body, headers)

    def parse_response(self, data: Any, request: CanonicalRequest) -> CanonicalResponse:
        if not isinstance(data, dict):
            raise AdapterError("Anthropic response was not a JSON object")
        text: list[str] = []
        summaries: list[str] = []
        calls: list[dict[str, Any]] = []
        for index, block in enumerate(data.get("content", []) if isinstance(data.get("content"), list) else []):
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text" and isinstance(block.get("text"), str):
                text.append(block["text"])
            elif kind == "summary" and isinstance(block.get("text"), str) and request.include_reasoning:
                summaries.append(block["text"])
            elif kind == "tool_use":
                calls.append({
                    "id": block.get("id", f"call_{index}"),
                    "type": "function",
                    "function": {"name": block.get("name", ""), "arguments": json.dumps(block.get("input", {}), ensure_ascii=False, separators=(",", ":"))},
                })
        stop = data.get("stop_reason")
        finish = "tool_calls" if calls or stop == "tool_use" else ("length" if stop == "max_tokens" else "stop")
        return CanonicalResponse(
            content="".join(text) if text else None,
            tool_calls=calls,
            reasoning_summary="\n".join(summaries) if summaries else None,
            finish_reason=finish,
            usage=normalize_usage(data.get("usage"), input_name="input_tokens", output_name="output_tokens"),
        )

    async def stream_deltas(self, response: httpx.Response, request: CanonicalRequest) -> AsyncIterator[CanonicalDelta]:
        active_tools: dict[int, dict[str, Any]] = {}
        async for event in iter_sse(response):
            if not event.data:
                continue
            try:
                data = json.loads(event.data)
            except (ValueError, TypeError):
                continue
            if not isinstance(data, dict):
                continue
            kind = data.get("type", event.event)
            if event.event == "error" or "error" in data or kind == "error":
                raise AdapterError("upstream Anthropic stream reported an error")
            if kind == "content_block_start":
                index = data.get("index", 0)
                block = data.get("content_block", {})
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    active_tools[index] = {"id": block.get("id"), "name": block.get("name", "")}
                    yield CanonicalDelta(tool_calls=[{"index": index, "id": block.get("id"), "type": "function", "function": {"name": block.get("name", ""), "arguments": ""}}])
                elif isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str) and block.get("text"):
                    yield CanonicalDelta(content=block["text"])
                elif isinstance(block, dict) and block.get("type") == "summary" and request.include_reasoning:
                    if isinstance(block.get("text"), str):
                        yield CanonicalDelta(reasoning_summary=block["text"])
            elif kind == "content_block_delta":
                index = data.get("index", 0)
                delta = data.get("delta", {})
                if not isinstance(delta, dict):
                    continue
                delta_type = delta.get("type")
                if delta_type == "text_delta" and isinstance(delta.get("text"), str):
                    yield CanonicalDelta(content=delta["text"])
                elif delta_type == "input_json_delta" and isinstance(delta.get("partial_json"), str):
                    yield CanonicalDelta(tool_calls=[{"index": index, "function": {"arguments": delta["partial_json"]}}])
                elif delta_type == "summary_delta" and request.include_reasoning and isinstance(delta.get("text"), str):
                    yield CanonicalDelta(reasoning_summary=delta["text"])
            elif kind == "message_start":
                message = data.get("message", {})
                usage = normalize_usage(message.get("usage") if isinstance(message, dict) else None, input_name="input_tokens", output_name="output_tokens")
                if usage:
                    yield CanonicalDelta(usage=usage)
            elif kind == "message_delta":
                delta = data.get("delta", {})
                stop = delta.get("stop_reason") if isinstance(delta, dict) else None
                usage = normalize_usage(data.get("usage"), input_name="input_tokens", output_name="output_tokens")
                if stop or usage:
                    finish = "tool_calls" if stop == "tool_use" else ("length" if stop == "max_tokens" else "stop" if stop else None)
                    yield CanonicalDelta(finish_reason=finish, usage=usage)
            elif kind == "error":
                raise AdapterError("Anthropic stream reported an error")


class AdapterRegistry:
    def for_candidate(self, candidate: Candidate) -> BaseAdapter:
        format_name = candidate.api_format
        if format_name == "anthropic":
            return AnthropicAdapter()
        if format_name == "openai_responses":
            if candidate.endpoints.get("responses"):
                return OpenAIResponsesAdapter()
            return OpenAIChatAdapter()
        if format_name == "openai_chat":
            if not candidate.endpoints.get("chat_completions") and candidate.endpoints.get("responses"):
                return OpenAIResponsesAdapter()
            return OpenAIChatAdapter()
        raise AdapterError("provider API format has not been discovered or configured")


def _normalize_finish(value: Any) -> str:
    if not isinstance(value, str):
        return "stop"
    if value in {"stop", "end_turn", "completed", "complete"}:
        return "stop"
    if value in {"length", "max_tokens", "incomplete"}:
        return "length"
    if value in {"tool_calls", "tool_use"}:
        return "tool_calls"
    if value == "content_filter":
        return "content_filter"
    return "stop"


def _extract_reasoning_summary(message: dict[str, Any], data: dict[str, Any]) -> str | None:
    # Only explicitly labeled summary fields are exposed. Fields such as
    # reasoning_content, analysis, or opaque reasoning_details are intentionally
    # never returned as private chain-of-thought.
    value = message.get("reasoning_summary")
    if isinstance(value, str):
        return value
    value = data.get("reasoning_summary")
    if isinstance(value, str):
        return value
    reasoning = message.get("reasoning")
    if isinstance(reasoning, dict):
        summary = reasoning.get("summary")
        if isinstance(summary, str):
            return summary
        if isinstance(summary, list):
            return "\n".join(part.get("text", "") for part in summary if isinstance(part, dict) and isinstance(part.get("text"), str)) or None
    return None


def candidate_summary_allowed(request: CanonicalRequest) -> bool:
    return request.include_reasoning
