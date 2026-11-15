"""Small Responses API client with deterministic offline mocking.

Live mode reads ``OPENAI_API_KEY`` only from the process environment. The
caller may load a local ``.env`` file into that environment before starting the
process; this module never opens, prints, or logs the file or key itself.
"""

from __future__ import annotations

import copy
import json
import os
import random
import socket
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Iterable, Mapping

from epistemics_memory import MODEL
from epistemics_memory.model_client import MockCall, MockResponse, ResponseResult


RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses"
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})
NON_RETRYABLE_RATE_LIMIT_CODES = frozenset(
    {
        "billing_hard_limit_reached",
        "credit_balance_exhausted",
        "insufficient_quota",
        "organization_spend_limit_exceeded",
        "organization_usage_limit_exceeded",
        "project_spend_limit_exceeded",
    }
)


class ResponsesClientError(RuntimeError):
    """Base class for client errors that never exposes authorization data."""


class ConfigurationError(ResponsesClientError):
    """The client configuration is missing or invalid."""


class APIConnectionError(ResponsesClientError):
    """All attempts failed before a valid HTTP response was received."""

    def __init__(self, attempts: int) -> None:
        self.attempts = attempts
        super().__init__(
            f"OpenAI Responses API connection failed after {attempts} attempt(s)"
        )


class APIError(ResponsesClientError):
    """A non-success HTTP response, represented without its potentially raw body."""

    def __init__(
        self,
        *,
        status_code: int,
        error_code: str | None,
        request_id: str | None,
        retryable: bool,
    ) -> None:
        self.status_code = status_code
        self.error_code = error_code
        self.request_id = request_id
        self.retryable = retryable
        details = [f"HTTP {status_code}"]
        if error_code:
            details.append(f"code={error_code}")
        if request_id:
            details.append(f"request_id={request_id}")
        super().__init__("OpenAI Responses API error (" + ", ".join(details) + ")")


class ResponseFormatError(ResponsesClientError):
    """The service or mock returned a malformed success response."""


@dataclass(frozen=True)
class _TransportReply:
    status_code: int
    body: dict[str, Any]
    headers: dict[str, str]


class _RetryableTransportFailure(Exception):
    pass


class ResponsesClient:
    """Synchronous client for non-streaming ``POST /v1/responses`` calls."""

    def __init__(
        self,
        *,
        model: str = MODEL,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        backoff_base_seconds: float = 0.5,
        max_retry_delay_seconds: float = 60.0,
        mock: bool = False,
        mock_responses: Iterable[MockResponse] = (),
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        if not model.strip():
            raise ConfigurationError("model must be non-empty")
        if timeout_seconds <= 0:
            raise ConfigurationError("timeout_seconds must be positive")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise ConfigurationError("max_retries must be a non-negative integer")
        if max_retries < 0:
            raise ConfigurationError("max_retries must be a non-negative integer")
        if backoff_base_seconds < 0:
            raise ConfigurationError("backoff_base_seconds cannot be negative")
        if max_retry_delay_seconds <= 0:
            raise ConfigurationError("max_retry_delay_seconds must be positive")

        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max_retries
        self.backoff_base_seconds = float(backoff_base_seconds)
        self.max_retry_delay_seconds = float(max_retry_delay_seconds)
        self.mock = mock
        self._sleep = sleep
        self._random_value = random_value
        self._mock_responses = deque(mock_responses)
        self._mock_calls: list[MockCall] = []

        if not mock and self._mock_responses:
            raise ConfigurationError("mock_responses require mock=True")
        if mock:
            self._api_key: str | None = None
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise ConfigurationError(
                    "OPENAI_API_KEY is not set in the process environment"
                )
            self._api_key = api_key

    def __repr__(self) -> str:
        return (
            f"ResponsesClient(model={self.model!r}, "
            f"timeout_seconds={self.timeout_seconds!r}, "
            f"max_retries={self.max_retries!r}, mock={self.mock!r})"
        )

    @property
    def mock_calls(self) -> tuple[MockCall, ...]:
        """Return safe payload-only call records for offline assertions."""

        return tuple(self._mock_calls)

    def create_response(
        self,
        *,
        instructions: str,
        input_text: str,
        text_format: Mapping[str, Any] | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> ResponseResult:
        """Create one independent, non-streaming response.

        The Responses API has no seed field. ``seed`` is accepted so the
        provider-neutral runner can use one interface, but is intentionally
        not transmitted.
        """

        if not instructions.strip():
            raise ValueError("instructions must be non-empty")
        if not input_text.strip():
            raise ValueError("input_text must be non-empty")
        if temperature is not None and (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0 <= temperature <= 2
        ):
            raise ValueError("temperature must be between 0 and 2")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer")
        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": input_text,
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if text_format is not None:
            payload["text"] = {"format": copy.deepcopy(dict(text_format))}

        attempts = self.max_retries + 1
        for attempt_index in range(attempts):
            try:
                reply = self._send(payload)
            except _RetryableTransportFailure:
                if attempt_index == attempts - 1:
                    raise APIConnectionError(attempts) from None
                self._sleep(self._fallback_delay(attempt_index))
                continue

            if 200 <= reply.status_code < 300:
                return self._parse_success(reply.body)

            api_error = self._api_error(reply)
            if not api_error.retryable or attempt_index == attempts - 1:
                raise api_error

            retry_after = self._retry_after_seconds(reply.headers)
            if (
                retry_after is not None
                and retry_after > self.max_retry_delay_seconds
            ):
                raise api_error
            delay = (
                retry_after + self._jitter()
                if retry_after is not None
                else self._fallback_delay(attempt_index)
            )
            self._sleep(delay)

        raise AssertionError("retry loop exhausted without returning or raising")

    def _send(self, payload: dict[str, Any]) -> _TransportReply:
        if self.mock:
            return self._send_mock(payload)
        return self._send_live(payload)

    def _send_mock(self, payload: dict[str, Any]) -> _TransportReply:
        self._mock_calls.append(
            MockCall(
                payload=copy.deepcopy(payload),
                timeout_seconds=self.timeout_seconds,
            )
        )
        if not self._mock_responses:
            raise ConfigurationError("mock response queue is empty")
        outcome = self._mock_responses.popleft()
        if outcome.status_code is None:
            raise _RetryableTransportFailure
        return _TransportReply(
            status_code=outcome.status_code,
            body=copy.deepcopy(dict(outcome.body)),
            headers=self._normalize_headers(outcome.headers),
        )

    def _send_live(self, payload: dict[str, Any]) -> _TransportReply:
        assert self._api_key is not None
        request = urllib.request.Request(
            RESPONSES_ENDPOINT,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "epistemics-memory/0.1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                body = self._decode_success_body(response.read())
                return _TransportReply(
                    status_code=response.status,
                    body=body,
                    headers=self._normalize_headers(response.headers),
                )
        except urllib.error.HTTPError as error:
            return _TransportReply(
                status_code=error.code,
                body=self._decode_error_body(error.read()),
                headers=self._normalize_headers(error.headers),
            )
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError):
            raise _RetryableTransportFailure from None

    @staticmethod
    def _decode_success_body(raw: bytes) -> dict[str, Any]:
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ResponseFormatError("response body was not valid UTF-8 JSON") from error
        if not isinstance(value, dict):
            raise ResponseFormatError("response body must be a JSON object")
        return value

    @staticmethod
    def _decode_error_body(raw: bytes) -> dict[str, Any]:
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _parse_success(body: dict[str, Any]) -> ResponseResult:
        response_id = body.get("id")
        model = body.get("model")
        status = body.get("status")
        output = body.get("output")
        if not isinstance(response_id, str) or not response_id:
            raise ResponseFormatError("response is missing a valid id")
        if not isinstance(model, str) or not model:
            raise ResponseFormatError("response is missing a valid model")
        if not isinstance(status, str) or not status:
            raise ResponseFormatError("response is missing a valid status")
        if not isinstance(output, list):
            raise ResponseFormatError("response is missing a valid output list")

        text_parts: list[str] = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text = part.get("text")
                    if not isinstance(text, str):
                        raise ResponseFormatError("output_text part has invalid text")
                    text_parts.append(text)

        return ResponseResult(
            response_id=response_id,
            model=model,
            status=status,
            output_text="".join(text_parts),
            raw=copy.deepcopy(body),
        )

    @staticmethod
    def _normalize_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
        if headers is None:
            return {}
        return {str(key).lower(): str(value) for key, value in headers.items()}

    @staticmethod
    def _error_code(body: Mapping[str, Any]) -> str | None:
        error = body.get("error")
        if not isinstance(error, dict):
            return None
        code = error.get("code")
        return code if isinstance(code, str) and code else None

    def _api_error(self, reply: _TransportReply) -> APIError:
        error_code = self._error_code(reply.body)
        retryable = (
            reply.status_code in RETRYABLE_STATUS_CODES
            or reply.status_code >= 500
        )
        if reply.status_code == 429 and error_code in NON_RETRYABLE_RATE_LIMIT_CODES:
            retryable = False
        return APIError(
            status_code=reply.status_code,
            error_code=error_code,
            request_id=reply.headers.get("x-request-id"),
            retryable=retryable,
        )

    def _fallback_delay(self, attempt_index: int) -> float:
        exponential = self.backoff_base_seconds * (2**attempt_index)
        return min(exponential + self._jitter(), self.max_retry_delay_seconds)

    def _jitter(self) -> float:
        value = self._random_value()
        if not 0 <= value <= 1:
            raise ConfigurationError("random_value must return a number from 0 to 1")
        return value * self.backoff_base_seconds

    @staticmethod
    def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
        raw = headers.get("retry-after")
        if raw is None:
            return None
        try:
            return max(0.0, float(raw.strip()))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(raw)
            except (TypeError, ValueError, OverflowError):
                return None
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
