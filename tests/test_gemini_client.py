"""Offline tests for the Gemini client and its retry contract."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from epistemics_memory.gemini_client import (
    DEFAULT_GEMINI_MODEL,
    GeminiAPIError,
    GeminiClient,
    GeminiConfigurationError,
    GeminiConnectionError,
)
from epistemics_memory.model_client import MockResponse


DUMMY_KEY = "TEST_DUMMY_GEMINI_KEY_NOT_A_CREDENTIAL"


def success_body(text: str = "mock Gemini answer") -> dict[str, object]:
    return {
        "id": "gemini_TEST_FIXTURE",
        "model": DEFAULT_GEMINI_MODEL,
        "status": "completed",
        "text": text,
    }


def mock_client(
    *responses: MockResponse,
    max_retries: int = 2,
    sleep=lambda _seconds: None,
) -> GeminiClient:
    return GeminiClient(
        mock=True,
        mock_responses=responses,
        max_retries=max_retries,
        sleep=sleep,
        random_value=lambda: 0.0,
    )


class GeminiMockModeTests(unittest.TestCase):
    def test_mock_mode_needs_no_key_and_never_constructs_sdk(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch.object(
            GeminiClient,
            "_get_sdk",
            side_effect=AssertionError("mock mode must not construct the SDK"),
        ):
            client = mock_client(MockResponse(200, success_body()))
            result = client.create_response(
                instructions="Answer from synthetic memory.",
                input_text="What is the synthetic test fact?",
            )

        self.assertEqual("mock Gemini answer", result.output_text)
        self.assertEqual("gemini_TEST_FIXTURE", result.response_id)
        self.assertEqual(DEFAULT_GEMINI_MODEL, result.model)
        self.assertEqual(1, len(client.mock_calls))

    def test_mock_call_records_structured_output_seed_and_temperature(self) -> None:
        client = mock_client(MockResponse(200, success_body("{}")))
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }

        client.create_response(
            instructions="Return synthetic JSON.",
            input_text="Use the schema.",
            text_format={
                "type": "json_schema",
                "name": "synthetic_test",
                "strict": True,
                "schema": schema,
            },
            temperature=0.0,
            seed=1729,
        )

        call = client.mock_calls[0]
        self.assertEqual(DEFAULT_GEMINI_MODEL, call.payload["model"])
        self.assertEqual(0.0, call.payload["temperature"])
        self.assertEqual(1729, call.payload["seed"])
        self.assertEqual("application/json", call.payload["response_mime_type"])
        self.assertEqual(schema, call.payload["response_json_schema"])
        self.assertFalse(hasattr(call, "headers"))

    def test_transient_429_honors_retry_after_then_succeeds(self) -> None:
        sleeps: list[float] = []
        client = mock_client(
            MockResponse(
                429,
                {"error": {"code": "rate_limit_exceeded"}},
                {"Retry-After": "2"},
            ),
            MockResponse(200, success_body()),
            sleep=sleeps.append,
        )

        result = client.create_response(instructions="Test.", input_text="Test.")

        self.assertEqual("mock Gemini answer", result.output_text)
        self.assertEqual([2.0], sleeps)
        self.assertEqual(2, len(client.mock_calls))

    def test_quota_429_is_not_retried(self) -> None:
        client = mock_client(
            MockResponse(429, {"error": {"code": "insufficient_quota"}})
        )

        with self.assertRaises(GeminiAPIError) as raised:
            client.create_response(instructions="Test.", input_text="Test.")

        self.assertFalse(raised.exception.retryable)
        self.assertEqual(1, len(client.mock_calls))

    def test_server_error_uses_the_same_retry_budget(self) -> None:
        sleeps: list[float] = []
        client = mock_client(
            MockResponse(503, {"error": {"code": "unavailable"}}),
            MockResponse(200, success_body()),
            sleep=sleeps.append,
        )

        result = client.create_response(instructions="Test.", input_text="Test.")

        self.assertEqual("mock Gemini answer", result.output_text)
        self.assertEqual([0.5], sleeps)
        self.assertEqual(2, len(client.mock_calls))

    def test_connection_failures_stop_after_configured_attempts(self) -> None:
        client = mock_client(
            MockResponse.connection_failure(),
            MockResponse.connection_failure(),
            max_retries=1,
        )

        with self.assertRaises(GeminiConnectionError) as raised:
            client.create_response(instructions="Test.", input_text="Test.")

        self.assertEqual(2, raised.exception.attempts)
        self.assertEqual(2, len(client.mock_calls))

    def test_error_does_not_expose_raw_message(self) -> None:
        client = mock_client(
            MockResponse(
                400,
                {"error": {"code": "bad_request", "message": DUMMY_KEY}},
            )
        )

        with self.assertRaises(GeminiAPIError) as raised:
            client.create_response(instructions="Test.", input_text="Test.")

        self.assertNotIn(DUMMY_KEY, str(raised.exception))


class GeminiEnvironmentSafetyTests(unittest.TestCase):
    def test_live_mode_requires_gemini_key_and_ignores_google_key(self) -> None:
        with patch.dict(
            os.environ,
            {"GOOGLE_API_KEY": DUMMY_KEY},
            clear=True,
        ):
            with self.assertRaisesRegex(
                GeminiConfigurationError, "GEMINI_API_KEY"
            ):
                GeminiClient()

    def test_key_never_appears_in_client_representation(self) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": DUMMY_KEY}, clear=True):
            client = GeminiClient()

        self.assertNotIn(DUMMY_KEY, repr(client))
        self.assertNotIn("api_key", repr(client).lower())

    def test_live_request_spacing_uses_configured_minimum_interval(self) -> None:
        times = iter((100.0, 102.0, 111.0))
        sleeps: list[float] = []
        with patch.dict(os.environ, {"GEMINI_API_KEY": DUMMY_KEY}, clear=True):
            client = GeminiClient(
                minimum_request_interval_seconds=8.0,
                sleep=sleeps.append,
                monotonic=lambda: next(times),
            )

        client._wait_for_request_slot()
        client._wait_for_request_slot()
        client._wait_for_request_slot()

        self.assertEqual([6.0, 5.0], sleeps)


if __name__ == "__main__":
    unittest.main()
