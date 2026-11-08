"""One place that talks to Claude. Lazy import, injected client, structured JSON output, token accounting."""

from __future__ import annotations

import json
from dataclasses import dataclass

DEFAULT_MODEL = "claude-opus-5-5"


class ModelError(RuntimeError):
    pass


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, response) -> None:
        u = getattr(response, "usage", None)
        self.calls += 1
        self.input_tokens += getattr(u, "input_tokens", 0) or 0
        self.output_tokens += getattr(u, "output_tokens", 0) or 0


def make_client():  # pragma: no cover - needs the optional extra and a key
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("Install the Claude extra first: uv sync --extra claude") from exc
    return anthropic.Anthropic()


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def call_json(client, system: str, prompt: str, schema: dict, usage: Usage | None = None,
              model: str = DEFAULT_MODEL, max_tokens: int = 4096, effort: str = "low") -> dict:
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
    )
    if usage is not None:
        usage.add(response)
    stop = getattr(response, "stop_reason", None)
    if stop == "refusal":
        raise ModelError("the model declined the request")
    if stop == "max_tokens":
        raise ModelError("the reply was cut off before the JSON was complete")
    body = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise ModelError(f"the reply was not valid JSON: {exc}") from exc
