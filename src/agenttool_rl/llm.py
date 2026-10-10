"""Small OpenAI-compatible chat client used by the Base LLM baseline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ChatBackendError(RuntimeError):
    """Raised when a chat backend cannot produce a usable response."""


@dataclass(frozen=True)
class ChatMessage:
    """Normalized first-choice response from a chat completion endpoint."""

    content: str | None = None
    tool_name: str | None = None
    tool_arguments: str | dict[str, Any] | None = None
    tool_call_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChatBackend(Protocol):
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ChatMessage: ...


@dataclass
class OpenAICompatibleBackend:
    """Call vLLM, SGLang, Ollama, or another OpenAI-compatible endpoint."""

    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 120.0
    temperature: float = 0.0
    require_tool_call: bool = True

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ChatMessage:
        endpoint = self._endpoint()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": [{"type": "function", "function": tool} for tool in tools],
            "temperature": self.temperature,
        }
        if self.require_tool_call:
            payload["tool_choice"] = "required"
        request = Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers=self._headers(),
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ChatBackendError(
                f"chat endpoint returned HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ChatBackendError(f"chat endpoint request failed: {exc}") from exc

        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ChatBackendError("chat endpoint response has no first-choice message") from exc
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            tool_call = tool_calls[0]
            function = tool_call["function"]
            return ChatMessage(
                content=message.get("content"),
                tool_name=function["name"],
                tool_arguments=function.get("arguments", "{}"),
                tool_call_id=tool_call.get("id"),
            )
        return ChatMessage(content=message.get("content"))

    def _endpoint(self) -> str:
        value = self.base_url.rstrip("/")
        if value.endswith("/chat/completions"):
            return value
        if value.endswith("/v1"):
            return value + "/chat/completions"
        return value + "/v1/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
