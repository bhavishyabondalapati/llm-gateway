"""OpenAI-compatible request/response models (the subset the gateway needs).

Unknown fields are allowed and passed through, so clients written for the
OpenAI SDK keep working.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool", "developer"]
    # Either a plain string or a list of content parts ({"type": "text", "text": ...}).
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    user: str | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = ""


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


def message_text(msg: ChatMessage) -> str:
    """Flatten a message's content to plain text."""
    if msg.content is None:
        return ""
    if isinstance(msg.content, str):
        return msg.content
    return "\n".join(p.get("text", "") for p in msg.content if p.get("type") == "text")


def estimate_tokens(text: str) -> int:
    """Cheap offline token estimate (~4 characters per token for English)."""
    return max(1, (len(text) + 3) // 4) if text else 0
