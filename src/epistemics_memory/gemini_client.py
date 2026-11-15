"""Google Gemini client with structured output and deterministic offline mocks.

Live mode reads ``GEMINI_API_KEY`` only from the process environment. A caller
may source a local ``.env`` file before starting the process; this module never
opens, prints, or logs that file or the key. The key is passed explicitly to
the Google Gen AI SDK, preventing its optional ``GOOGLE_API_KEY`` fallback.
"""

from __future__ import annotations

import copy
import hashlib
import os
import random
import time
from collections import deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Iterable, Mapping

from epistemics_memory.model_client import MockCall, MockResponse, ResponseResult


DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, *range(500, 600)})
NON_RETRYABLE_RATE_LIMIT_CODES = frozenset(
    {
        "billing_hard_limit_reached",
        "credit_balance_exhausted",
        "insufficient_quota",
    }
)


class GeminiClientError(RuntimeError):
    """Base class for Gemini errors that never exposes authorization data."""


class GeminiConfigurationError(GeminiClientError):
    """The Gemini client configuration is missing or invalid."""


class GeminiConnectionError(GeminiClientError):
    """All configured SDK attempts failed without a usable response."""

    def __init__(self, attempts: int) -> None:
        self.attempts = attempts
        super().__init__(
            f"Gemini API connection failed after {attempts} attempt(s)"
        )


class GeminiAPIError(GeminiClientError):
    """A safe representation of a non-success Gemini response."""

    def __init__(
        self,
        *,
        status_code: int,
        error_code: str | None,
        retryable: bool,
    ) -> None:
        self.status_code = status_code
        self.error_code = error_code
        self.retryable = retryable
        details = [f"HTTP {status_code}"]
        if error_code:
            details.append(f"code={error_code}")
        super().__init__("Gemini API error (" + ", ".join(details) + ")")


class GeminiResponseFormatError(GeminiClientError):
    """The SDK or mock returned a malformed successful response."""


class GeminiClient:
    """Synchronous Gemini client using the official ``google-genai`` SDK."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        backoff_base_seconds: float = 0.5,
        max_retry_delay_seconds: float = 60.0,
        minimum_request_interval_seconds: float = 0.0,
        mock: bool = False,
        mock_responses: Iterable[MockResponse] = (),
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not model.strip():
            raise GeminiConfigurationError("model must be non-empty")
        if timeout_seconds <= 0:
            raise GeminiConfigurationError("timeout_seconds must be positive")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise GeminiConfigurationError(
                "max_retries must be a non-negative integer"
            )
        if max_retries < 0:
            raise GeminiConfigurationError(
                "max_retries must be a non-negative integer"
            )
        if backoff_base_seconds < 0:
            raise GeminiConfigurationError(
                "backoff_base_seconds cannot be negative"
            )
        if max_retry_delay_seconds <= 0:
            raise GeminiConfigurationError(
                "max_retry_delay_seconds must be positive"
            )
        if minimum_request_interval_seconds < 0:
            raise GeminiConfigurationError(
                "minimum_request_interval_seconds cannot be negative"
            )

        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max_retries
        self.backoff_base_seconds = float(backoff_base_seconds)
        self.max_retry_delay_seconds = float(max_retry_delay_seconds)
        self.minimum_request_interval_seconds = float(
            minimum_request_interval_seconds
        )
        self.mock = mock
        self._sleep = sleep
        self._random_value = random_value
        self._monotonic = monotonic
        self._last_request_started: float | None = None
        self._mock_responses = deque(mock_responses)
        self._mock_calls: list[MockCall] = []
        self._sdk_client: Any | None = None
        self._sdk_types: Any | None = None
        self._sdk_errors: Any | None = None

        if not mock and self._mock_responses:
            raise GeminiConfigurationError("mock_responses require mock=True")
        if mock:
            self._api_key: str | None = None
        else:
            api_key = os.environ.get("GEMINI_API_KEY", "").strip()
            if not api_key:
                raise GeminiConfigurationError(
                    "GEMINI_API_KEY is not set in the process environment"
                )
            self._api_key = api_key

    def __repr__(self) -> str:
        return (
            f"GeminiClient(model={self.model!r}, "
            f"timeout_seconds={self.timeout_seconds!r}, "
            f"max_retries={self.max_retries!r}, mock={self.mock!r})"
        )

    @property
    def mock_calls(self) -> tuple[MockCall, ...]:
        """Return safe payload-only request records for offline assertions."""

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
        """Create one independent, non-streaming Gemini response."""

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
        if seed is not None:
            payload["seed"] = seed
        if text_format is not None:
            schema = text_format.get("schema")
            if not isinstance(schema, Mapping):
                raise ValueError("text_format must contain an object schema")
            payload["response_mime_type"] = "application/json"
            payload["response_json_schema"] = copy.deepcopy(dict(schema))

        if self.mock:
            return self._create_mock_response(payload)
        return self._create_live_response(payload)

    def _create_mock_response(self, payload: dict[str, Any]) -> ResponseResult:
        attempts = self.max_retries + 1
        for attempt_index in range(attempts):
            self._mock_calls.append(
                MockCall(
                    payload=copy.deepcopy(payload),
                    timeout_seconds=self.timeout_seconds,
                )
            )
            if not self._mock_responses:
                raise GeminiConfigurationError("mock response queue is empty")
            outcome = self._mock_responses.popleft()
            if outcome.status_code is None:
                if attempt_index == attempts - 1:
                    raise GeminiConnectionError(attempts)
                self._sleep(self._fallback_delay(attempt_index))
                continue
            if 200 <= outcome.status_code < 300:
                return self._parse_mock_success(outcome.body)

            api_error = self._mock_api_error(outcome)
            if not api_error.retryable or attempt_index == attempts - 1:
                raise api_error
            retry_after = self._retry_after_seconds(outcome.headers)
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

    def _create_live_response(self, payload: dict[str, Any]) -> ResponseResult:
        client, types, errors = self._get_sdk()
        config_values: dict[str, Any] = {
            "system_instruction": payload["instructions"],
        }
        for field in ("temperature", "seed", "response_mime_type"):
            if field in payload:
                config_values[field] = payload[field]
        if "response_json_schema" in payload:
            config_values["response_json_schema"] = payload[
                "response_json_schema"
            ]
        config = types.GenerateContentConfig(**config_values)
        self._wait_for_request_slot()
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=payload["input"],
                config=config,
            )
        except errors.APIError as error:
            status_code = getattr(error, "code", 0)
            if isinstance(status_code, bool) or not isinstance(status_code, int):
                status_code = 0
            error_code = getattr(error, "status", None)
            if not isinstance(error_code, str):
                error_code = None
            retryable = status_code in RETRYABLE_STATUS_CODES
            if status_code == 429 and error_code in NON_RETRYABLE_RATE_LIMIT_CODES:
                retryable = False
            raise GeminiAPIError(
                status_code=status_code,
                error_code=error_code,
                retryable=retryable,
            ) from None
        except Exception:
            raise GeminiConnectionError(self.max_retries + 1) from None

        try:
            output_text = response.text
        except (AttributeError, TypeError, ValueError):
            raise GeminiResponseFormatError(
                "Gemini response is missing usable text"
            ) from None
        if not isinstance(output_text, str):
            raise GeminiResponseFormatError("Gemini response text must be a string")

        raw = self._response_dict(response, output_text)
        response_id = getattr(response, "response_id", None)
        if not isinstance(response_id, str) or not response_id:
            digest = hashlib.sha256(output_text.encode("utf-8")).hexdigest()[:16]
            response_id = f"gemini_{digest}"
        model = getattr(response, "model_version", None)
        if not isinstance(model, str) or not model:
            model = self.model
        return ResponseResult(
            response_id=response_id,
            model=model,
            status="completed",
            output_text=output_text,
            raw=raw,
        )

    def _wait_for_request_slot(self) -> None:
        """Space live requests without exposing request content or credentials."""

        now = self._monotonic()
        if self._last_request_started is None:
            self._last_request_started = now
            return
        wait_seconds = max(
            0.0,
            self.minimum_request_interval_seconds
            - (now - self._last_request_started),
        )
        if wait_seconds:
            self._sleep(wait_seconds)
        self._last_request_started = now + wait_seconds

    def _get_sdk(self) -> tuple[Any, Any, Any]:
        if self._sdk_client is not None:
            return self._sdk_client, self._sdk_types, self._sdk_errors
        try:
            from google import genai
            from google.genai import errors, types
        except ImportError:
            raise GeminiConfigurationError(
                "google-genai is not installed; install the project dependencies"
            ) from None

        assert self._api_key is not None
        retry_options = types.HttpRetryOptions(
            attempts=self.max_retries + 1,
            initial_delay=self.backoff_base_seconds,
            max_delay=self.max_retry_delay_seconds,
            exp_base=2,
            jitter=1.0,
            http_status_codes=sorted(RETRYABLE_STATUS_CODES),
        )
        http_options = types.HttpOptions(
            timeout=int(self.timeout_seconds * 1000),
            retry_options=retry_options,
        )
        self._sdk_client = genai.Client(
            api_key=self._api_key,
            http_options=http_options,
        )
        self._sdk_types = types
        self._sdk_errors = errors
        return self._sdk_client, self._sdk_types, self._sdk_errors

    @staticmethod
    def _response_dict(response: Any, output_text: str) -> dict[str, Any]:
        model_dump = getattr(response, "model_dump", None)
        if callable(model_dump):
            value = model_dump(mode="json", exclude_none=True)
            if isinstance(value, dict):
                return value
        return {"text": output_text}

    @staticmethod
    def _parse_mock_success(body: Mapping[str, Any]) -> ResponseResult:
        response_id = body.get("id")
        model = body.get("model")
        status = body.get("status")
        output_text = body.get("text")
        if not isinstance(response_id, str) or not response_id:
            raise GeminiResponseFormatError("mock response is missing a valid id")
        if not isinstance(model, str) or not model:
            raise GeminiResponseFormatError("mock response is missing a valid model")
        if not isinstance(status, str) or not status:
            raise GeminiResponseFormatError("mock response is missing a valid status")
        if not isinstance(output_text, str):
            raise GeminiResponseFormatError("mock response is missing valid text")
        return ResponseResult(
            response_id=response_id,
            model=model,
            status=status,
            output_text=output_text,
            raw=copy.deepcopy(dict(body)),
        )

    def _mock_api_error(self, response: MockResponse) -> GeminiAPIError:
        error = response.body.get("error")
        error_code: str | None = None
        if isinstance(error, Mapping):
            raw_code = error.get("code")
            if isinstance(raw_code, str) and raw_code:
                error_code = raw_code
        status_code = response.status_code
        assert status_code is not None
        retryable = status_code in RETRYABLE_STATUS_CODES
        if status_code == 429 and error_code in NON_RETRYABLE_RATE_LIMIT_CODES:
            retryable = False
        return GeminiAPIError(
            status_code=status_code,
            error_code=error_code,
            retryable=retryable,
        )

    def _fallback_delay(self, attempt_index: int) -> float:
        exponential = self.backoff_base_seconds * (2**attempt_index)
        return min(exponential + self._jitter(), self.max_retry_delay_seconds)

    def _jitter(self) -> float:
        value = self._random_value()
        if not 0 <= value <= 1:
            raise GeminiConfigurationError(
                "random_value must return a number from 0 to 1"
            )
        return value * self.backoff_base_seconds

    @staticmethod
    def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
        raw = next(
            (value for key, value in headers.items() if key.lower() == "retry-after"),
            None,
        )
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
