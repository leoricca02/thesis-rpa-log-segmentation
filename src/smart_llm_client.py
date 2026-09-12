"""Lightweight, dependency-minimal REST client for Google's Gemini models.

This client is engineered for academic research and high-volume data processing.
It bypasses bulky SDKs in favour of direct HTTP requests and implements four
features that matter for a reproducible, low-cost thesis pipeline:

1. Deterministic JSON-schema-constrained output.
2. Local MD5 caching (keyed on model + prompts + schema) to avoid paying for
   identical calls during repeated test runs.
3. Token telemetry written to CSV via the ``csv`` module (properly quoted).
4. Bounded retry with exponential back-off on transient HTTP failures
   (429 rate-limit / 5xx), which became essential after Google's December 2025
   free-tier quota reductions.

The network call is isolated behind ``_transport`` so the client can be unit
tested deterministically without hitting the network.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Callable, Protocol

import requests
from dotenv import load_dotenv

# Load hidden environment variables (e.g. GEMINI_API_KEY) from a local .env file
# so secrets are never hard-coded or committed to version control.
load_dotenv()

# Markdown code-fence stripper, compiled once at import time.
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```")

# HTTP status codes worth retrying: 429 (rate limit) and the 5xx server errors.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class TransportError(RuntimeError):
    """Raised when the underlying HTTP transport fails irrecoverably."""


class _Transport(Protocol):
    """Minimal callable contract for the HTTP layer (enables test injection)."""

    def __call__(
        self, url: str, payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        ...


def _default_transport(
    url: str, payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    """Perform a single POST request to the Gemini endpoint.

    Args:
        url: Fully-qualified endpoint URL including the API key.
        payload: JSON-serialisable request body.
        timeout: Per-request timeout in seconds.

    Returns:
        The parsed JSON response body as a dict.

    Raises:
        requests.HTTPError: If the server returns a 4xx/5xx status.
        requests.RequestException: For connection/timeout failures.
    """
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


class SmartLLMClient:
    """A cached, observable, retry-aware REST wrapper for Gemini models."""

    _ENDPOINT_TEMPLATE = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "{model}:generateContent?key={api_key}"
    )

    def __init__(
        self,
        api_key: str | None = None,
        cache_file: str = "gemini_cache.json",
        token_log_file: str = "token_telemetry.csv",
        *,
        timeout: float = 60.0,
        max_retries: int = 4,
        base_backoff: float = 2.0,
        transport: _Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initialise the client.

        Args:
            api_key: Gemini API key. Falls back to the GEMINI_API_KEY env var.
            cache_file: Path to the on-disk JSON response cache.
            token_log_file: Path to the CSV telemetry log.
            timeout: Per-request HTTP timeout in seconds.
            max_retries: Maximum retry attempts on transient failures.
            base_backoff: Base seconds for exponential back-off (2 -> 2,4,8,...).
            transport: Injectable HTTP callable; defaults to a real POST. Tests
                pass a fake to avoid network access.
            sleep: Injectable sleep function (tests pass a no-op).

        Raises:
            ValueError: If no API key is available and the default transport is
                used (a real network call would be impossible without a key).
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._transport = transport or _default_transport
        self._sleep = sleep

        # Only hard-fail on a missing key when we would actually hit the network.
        # With an injected transport (tests), the key is irrelevant.
        if not self.api_key and transport is None:
            raise ValueError(
                "CRITICAL: API Key not found! Add GEMINI_API_KEY to your .env file."
            )

        self.cache_file = cache_file
        self.token_log_file = token_log_file
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.base_backoff = base_backoff
        self.cache = self._load_cache()

    # ------------------------------------------------------------------ cache
    def _load_cache(self) -> dict[str, Any]:
        """Load the on-disk cache, tolerating a missing or corrupt file."""
        if not os.path.exists(self.cache_file):
            return {}
        try:
            with open(self.cache_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            # A corrupt or unreadable cache must never block startup.
            return {}

    def _save_cache(self) -> None:
        """Persist the in-memory cache atomically (write-temp-then-rename)."""
        tmp_path = f"{self.cache_file}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(self.cache, handle, indent=4, ensure_ascii=False)
            os.replace(tmp_path, self.cache_file)
        except OSError as exc:
            print(f"[!] Warning: could not persist cache to disk: {exc}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def _get_cache_key(
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        thinking_budget: int = 0,
    ) -> str:
        """Return a stable MD5 fingerprint of the full request.

        ``sort_keys=True`` makes the schema serialisation order-independent so
        semantically identical schemas always hit the same cache entry.
        ``thinking_budget`` is included because a different budget changes the
        response distribution and must not hit a zero-thinking cached result.
        """
        data_string = (
            f"{model}\x1f{system_prompt}\x1f{user_prompt}\x1f"
            f"{json.dumps(schema, sort_keys=True)}\x1f{thinking_budget}"
        )
        return hashlib.md5(data_string.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------- telemetry
    def _log_tokens(
        self, model: str, in_tokens: int, out_tokens: int, total_tokens: int
    ) -> None:
        """Append one telemetry row using the csv module (safely quoted)."""
        file_exists = os.path.exists(self.token_log_file)
        try:
            with open(
                self.token_log_file, "a", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.writer(handle)
                if not file_exists:
                    writer.writerow(
                        [
                            "timestamp",
                            "model",
                            "input_tokens",
                            "output_tokens",
                            "total_tokens",
                        ]
                    )
                writer.writerow(
                    [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        model,
                        in_tokens,
                        out_tokens,
                        total_tokens,
                    ]
                )
        except OSError as exc:
            print(f"[!] Warning: could not write telemetry: {exc}")

    # ------------------------------------------------------------------ core
    def generate_content(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        thinking_budget: int = 0,
    ) -> Any:
        """Return parsed JSON output for a schema-constrained completion.

        Checks the cache first; on a miss, calls the API with bounded retry,
        logs token usage, strips markdown fences, parses JSON, and caches the
        result.

        Args:
            model: Gemini model identifier (e.g. 'gemini-2.5-flash').
            system_prompt: System instruction text.
            user_prompt: User message text.
            schema: Gemini ``responseSchema`` constraining the output shape.
            thinking_budget: Maximum internal thinking tokens Gemini may spend.
                0 disables thinking entirely (fastest, cheapest — use for
                Phase 0A/0B which are simple classification tasks).
                512–1024 is enough for Phase 1/2 reasoning on small logs.
                Default is 0 (off) so callers must opt in to thinking.

        Returns:
            The parsed JSON value (dict or list, depending on the schema).

        Raises:
            TransportError: If every retry attempt fails.
            ValueError: If the response cannot be parsed into JSON or has an
                unexpected shape.
        """
        cache_key = self._get_cache_key(model, system_prompt, user_prompt, schema, thinking_budget)
        if cache_key in self.cache:
            print("\n[CACHE HIT] Reusing locally stored response (no API call).")
            return self.cache[cache_key]

        print(f"\n[CACHE MISS] Calling {model} API...")
        url = self._ENDPOINT_TEMPLATE.format(model=model, api_key=self.api_key or "")
        generation_config: dict[str, Any] = {
            "responseMimeType": "application/json",
            "responseSchema": schema,
            "thinkingConfig": {"thinkingBudget": max(0, thinking_budget)},
        }
        payload = {
            "contents": [{"parts": [{"text": user_prompt}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": generation_config,
        }

        result = self._call_with_retry(url, payload)
        self._record_usage(result, model)
        parsed = self._extract_json(result)

        self.cache[cache_key] = parsed
        self._save_cache()
        return parsed

    def _call_with_retry(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Call the transport with exponential back-off on transient errors."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._transport(url, payload, self.timeout)
            except requests.HTTPError as exc:
                status = (
                    exc.response.status_code if exc.response is not None else None
                )
                last_exc = exc
                if status not in _RETRYABLE_STATUS or attempt == self.max_retries:
                    raise TransportError(
                        f"HTTP {status} after {attempt + 1} attempt(s): {exc}"
                    ) from exc
                self._backoff(attempt, f"HTTP {status}")
            except requests.RequestException as exc:
                # Connection errors / timeouts are transient by nature.
                last_exc = exc
                if attempt == self.max_retries:
                    raise TransportError(
                        f"Network failure after {attempt + 1} attempt(s): {exc}"
                    ) from exc
                self._backoff(attempt, "network error")
        # Unreachable, but keeps type-checkers satisfied.
        raise TransportError(f"Exhausted retries: {last_exc}")

    def _backoff(self, attempt: int, reason: str) -> None:
        """Sleep for an exponentially increasing interval and log the reason."""
        delay = self.base_backoff * (2 ** attempt)
        print(
            f"[RETRY] {reason}; backing off {delay:.1f}s "
            f"(attempt {attempt + 1}/{self.max_retries})."
        )
        self._sleep(delay)

    def _record_usage(self, result: dict[str, Any], model: str) -> None:
        """Extract usage metadata and append it to the telemetry log."""
        try:
            usage = result.get("usageMetadata", {})
            in_t = int(usage.get("promptTokenCount", 0))
            out_t = int(usage.get("candidatesTokenCount", 0))
            tot_t = int(usage.get("totalTokenCount", 0))
            self._log_tokens(model, in_t, out_t, tot_t)
            print(f"[TOKENS] {in_t} in | {out_t} out | {tot_t} total")
        except (TypeError, ValueError) as exc:
            print(f"[!] Warning: could not log token telemetry: {exc}")

    @staticmethod
    def _extract_json(result: dict[str, Any]) -> Any:
        """Pull the model text out of the response envelope and parse it.

        Args:
            result: Raw Gemini response body.

        Returns:
            The parsed JSON value.

        Raises:
            ValueError: If the envelope is malformed or the text is not JSON.
        """
        try:
            text_response = result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            # A blocked prompt or empty candidate list lands here.
            block_reason = ""
            if isinstance(result, dict):
                feedback = result.get("promptFeedback", {})
                block_reason = feedback.get("blockReason", "")
            raise ValueError(
                f"Malformed/empty Gemini response (blockReason={block_reason!r}): "
                f"{str(result)[:300]}"
            ) from exc

        clean_text = _CODE_FENCE_RE.sub("", text_response).strip()
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Model returned non-JSON content: {clean_text[:300]!r}"
            ) from exc


if __name__ == "__main__":
    pass
