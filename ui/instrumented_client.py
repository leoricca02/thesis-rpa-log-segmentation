"""Instrumentation layer between the thesis pipeline and the demo UI.

Nothing in ``src/`` is modified. The UI observes the pipeline from the outside:

* ``InstrumentedClient`` subclasses ``SmartLLMClient`` and reports every LLM
  call (prompt, cache hit/miss, token usage, latency, parsed answer). When
  ``include_thoughts`` is on, it also asks Gemini for its thought summaries
  and separates them from the JSON answer. Because the cache key of the base
  class does not cover ``includeThoughts``, cached responses (including the
  ones committed under ``results/case_study``) are still reused; thoughts are
  kept in a side store so a cached answer can still show the thoughts that
  produced it.
* ``ThreadLocalStdout`` lets each pipeline run capture the lines the phases
  ``print`` without touching the server's own output. Those lines drive the
  phase timeline and the console panel.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any, Callable

from smart_llm_client import SmartLLMClient

EventSink = Callable[[dict[str, Any]], None]


class InstrumentedClient(SmartLLMClient):
    """A ``SmartLLMClient`` that reports each call to an event sink."""

    def __init__(
        self,
        *,
        on_event: EventSink,
        phase_of: Callable[[], str],
        include_thoughts: bool = False,
        thoughts_file: str | None = None,
        seed_cache_files: tuple[str, ...] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._on_event = on_event
        self._phase_of = phase_of
        self.include_thoughts = include_thoughts
        self._thoughts_file = thoughts_file
        self._thoughts: dict[str, str] = self._load_json(thoughts_file)
        # Seed entries are merged underneath the run's own cache, so a fresh
        # UI cache still replays the committed thesis responses.
        for path in seed_cache_files:
            for key, value in self._load_json(path).items():
                self.cache.setdefault(key, value)
        self._last_thoughts = ""
        self._last_usage: dict[str, int] = {}

    @staticmethod
    def _load_json(path: str | None) -> dict[str, Any]:
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_thoughts(self) -> None:
        if not self._thoughts_file:
            return
        tmp = f"{self._thoughts_file}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(self._thoughts, handle, indent=2, ensure_ascii=False)
            os.replace(tmp, self._thoughts_file)
        except OSError:
            pass

    # ------------------------------------------------------ base-class hooks
    def _thinking_config(self, model: str, thinking_budget: int) -> dict[str, Any]:
        config = SmartLLMClient._thinking_config(model, thinking_budget)
        # Thought summaries only exist when the model is allowed to think.
        if self.include_thoughts and thinking_budget > 0:
            config["includeThoughts"] = True
        return config

    def _extract_json(self, result: dict[str, Any]) -> Any:
        """Split thought-summary parts from the answer, then parse the answer."""
        self._last_thoughts = ""
        try:
            parts = result["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError):
            return SmartLLMClient._extract_json(result)
        thoughts = [p.get("text", "") for p in parts if p.get("thought")]
        answer = [p for p in parts if not p.get("thought")]
        self._last_thoughts = "\n\n".join(t.strip() for t in thoughts if t.strip())
        if not answer:
            return SmartLLMClient._extract_json(result)
        answer_text = "".join(p.get("text", "") for p in answer)
        stripped = {
            **result,
            "candidates": [{"content": {"parts": [{"text": answer_text}]}}],
        }
        return SmartLLMClient._extract_json(stripped)

    def _record_usage(self, result: dict[str, Any], model: str) -> None:
        usage = result.get("usageMetadata", {}) if isinstance(result, dict) else {}
        self._last_usage = {
            "input": int(usage.get("promptTokenCount", 0) or 0),
            "output": int(usage.get("candidatesTokenCount", 0) or 0),
            "thoughts": int(usage.get("thoughtsTokenCount", 0) or 0),
            "total": int(usage.get("totalTokenCount", 0) or 0),
        }
        super()._record_usage(result, model)

    # ------------------------------------------------------------ main entry
    def generate_content(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        thinking_budget: int = 0,
    ) -> Any:
        key = self._get_cache_key(
            model, system_prompt, user_prompt, schema, thinking_budget
        )
        cached = key in self.cache
        phase = self._phase_of()
        thinking = SmartLLMClient._thinking_config(model, thinking_budget)
        self._last_thoughts = ""
        self._last_usage = {}
        self._on_event(
            {
                "type": "llm_call",
                "phase": phase,
                "model": model,
                "cached": cached,
                "thinking": thinking,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        started = time.monotonic()
        try:
            answer = super().generate_content(
                model, system_prompt, user_prompt, schema, thinking_budget
            )
        except Exception as exc:
            self._on_event(
                {"type": "llm_error", "phase": phase, "message": str(exc)[:500]}
            )
            raise

        if cached:
            thoughts = self._thoughts.get(key, "")
        else:
            thoughts = self._last_thoughts
            if thoughts:
                self._thoughts[key] = thoughts
                self._save_thoughts()
        self._on_event(
            {
                "type": "llm_result",
                "phase": phase,
                "cached": cached,
                "seconds": round(time.monotonic() - started, 2),
                "tokens": self._last_usage,
                "thoughts": thoughts,
                "answer": answer,
            }
        )
        return answer


class ThreadLocalStdout:
    """A ``sys.stdout`` replacement that routes writes per thread.

    A thread that registers a line callback has its output split into lines
    and handed to that callback (and still echoed to the real terminal);
    every other thread writes straight through.
    """

    def __init__(self, real: Any) -> None:
        self._real = real
        self._local = threading.local()

    def register(self, on_line: Callable[[str], None]) -> None:
        self._local.on_line = on_line
        self._local.buffer = ""

    def unregister(self) -> None:
        self.flush_line()
        self._local.on_line = None

    def write(self, text: str) -> int:
        on_line = getattr(self._local, "on_line", None)
        self._real.write(text)
        if on_line is not None:
            self._local.buffer += text
            *lines, self._local.buffer = self._local.buffer.split("\n")
            for line in lines:
                on_line(line)
        return len(text)

    def flush_line(self) -> None:
        on_line = getattr(self._local, "on_line", None)
        if on_line is not None and self._local.buffer:
            on_line(self._local.buffer)
            self._local.buffer = ""

    def flush(self) -> None:
        self._real.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def install_stdout_router() -> ThreadLocalStdout:
    """Install (once) and return the process-wide stdout router."""
    if not isinstance(sys.stdout, ThreadLocalStdout):
        sys.stdout = ThreadLocalStdout(sys.stdout)
    return sys.stdout
