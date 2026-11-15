"""Offline tests for the Responses API client and retry contract."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from epistemics_memory import MODEL
from epistemics_memory.openai_client import (
    APIConnectionError,
    APIError,
    ConfigurationError,
    MockResponse,
    ResponsesClient,
)


DUMMY_KEY = "TEST_DUMMY_KEY_NOT_A_CREDENTIAL"


def success_body(text: str = "mock answer") -> dict[str, object]:
    return {
        "id": "resp_TEST_FIXTURE",
        "model": MODEL,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }


def mock_client(
    *responses: MockResponse,
    max_retries: int = 2,
    sleep=lambda _seconds: None,
) -> ResponsesClient:
    return ResponsesClient(
        mock=True,
        mock_responses=responses,
        max_retries=max_retries,
        sleep=sleep,
        random_value=lambda: 0.0,
    )


class MockModeTests(unittest.TestCase):
    def test_mock_mode_needs_no_key_and_never_opens_network(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch(
            "epistemics_memory.openai_client.urllib.request.urlopen",
            side_effect=AssertionError("network must not be called in mock mode"),
        ):
            client = mock_client(MockResponse(200, success_body()))
            result = client.create_response(
                instructions="Answer from memory.",
                input_text="What is the test fact?",
            )

        self.assertEqual("mock answer", result.output_text)
        self.assertEqual("resp_TEST_FIXTURE", result.response_id)
        self.assertEqual(MODEL, result.model)
        self.assertEqual(1, len(client.mock_calls))

    def test_mock_call_records_payload_and_timeout_but_no_headers(self) -> None:
        client = ResponsesClient(
            mock=True,
            mock_responses=(MockResponse(200, success_body("{}")),),
            timeout_seconds=12.5,
        )
        text_format = {
            "type": "json_schema",
            "name": "test_fixture",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }

        client.create_response(
            instructions="Return JSON.",
            input_text="Use the test schema.",
            text_format=text_format,
        )

        call = client.mock_calls[0]
        self.assertEqual(12.5, call.timeout_seconds)
        self.assertEqual(MODEL, call.payload["model"])
        self.assertEqual("Return JSON.", call.payload["instructions"])
        self.assertEqual("Use the test schema.", call.payload["input"])
        self.assertEqual(text_format, call.payload["text"]["format"])
        self.assertFalse(hasattr(call, "headers"))

    def test_output_text_joins_multiple_message_parts(self) -> None:
        body = success_body()
        body["output"] = [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "first"},
                    {"type": "output_text", "text": " second"},
                ],
            }
        ]
        client = mock_client(MockResponse(200, body))

        result = client.create_response(instructions="Test.", input_text="Test.")

        self.assertEqual("first second", result.output_text)

    def test_transient_429_honors_retry_after_then_succeeds(self) -> None:
        sleeps: list[float] = []
        client = mock_client(
            MockResponse(
                429,
                {"error": {"code": "rate_limit_exceeded"}},
                {"Retry-After": "2", "X-Request-ID": "req_TEST_RATE_LIMIT"},
            ),
            MockResponse(200, success_body()),
            sleep=sleeps.append,
        )

        result = client.create_response(instructions="Test.", input_text="Test.")

        self.assertEqual("mock answer", result.output_text)
        self.assertEqual([2.0], sleeps)
        self.assertEqual(2, len(client.mock_calls))

    def test_quota_429_is_not_retried(self) -> None:
        client = mock_client(
            MockResponse(
                429,
                {"error": {"code": "credit_balance_exhausted"}},
                {"X-Request-ID": "req_TEST_QUOTA"},
            )
        )

        with self.assertRaises(APIError) as raised:
            client.create_response(instructions="Test.", input_text="Test.")

        self.assertFalse(raised.exception.retryable)
        self.assertEqual("credit_balance_exhausted", raised.exception.error_code)
        self.assertEqual(1, len(client.mock_calls))

    def test_non_transient_error_does_not_expose_raw_error_message(self) -> None:
        client = mock_client(
            MockResponse(
                400,
                {
                    "error": {
                        "code": "bad_request",
                        "message": DUMMY_KEY,
                    }
                },
                {"X-Request-ID": "req_TEST_BAD_REQUEST"},
            )
        )

        with self.assertRaises(APIError) as raised:
            client.create_response(instructions="Test.", input_text="Test.")

        self.assertFalse(raised.exception.retryable)
        self.assertNotIn(DUMMY_KEY, str(raised.exception))
        self.assertIn("req_TEST_BAD_REQUEST", str(raised.exception))
        self.assertEqual(1, len(client.mock_calls))

    def test_server_errors_stop_after_configured_retries(self) -> None:
        sleeps: list[float] = []
        client = ResponsesClient(
            mock=True,
            mock_responses=(
                MockResponse(500, {"error": {"code": "server_error"}}),
                MockResponse(500, {"error": {"code": "server_error"}}),
                MockResponse(503, {"error": {"code": "server_is_overloaded"}}),
            ),
            max_retries=2,
            backoff_base_seconds=0.25,
            sleep=sleeps.append,
            random_value=lambda: 0.0,
        )

        with self.assertRaises(APIError) as raised:
            client.create_response(instructions="Test.", input_text="Test.")

        self.assertEqual(503, raised.exception.status_code)
        self.assertEqual([0.25, 0.5], sleeps)
        self.assertEqual(3, len(client.mock_calls))

    def test_connection_failures_are_retried(self) -> None:
        client = mock_client(
            MockResponse.connection_failure(),
            MockResponse.connection_failure(),
            max_retries=1,
        )

        with self.assertRaises(APIConnectionError) as raised:
            client.create_response(instructions="Test.", input_text="Test.")

        self.assertEqual(2, raised.exception.attempts)
        self.assertEqual(2, len(client.mock_calls))

    def test_retry_after_above_configured_limit_is_not_shortened(self) -> None:
        sleeps: list[float] = []
        client = ResponsesClient(
            mock=True,
            mock_responses=(
                MockResponse(
                    503,
                    {"error": {"code": "server_is_overloaded"}},
                    {"Retry-After": "120"},
                ),
            ),
            max_retry_delay_seconds=60,
            sleep=sleeps.append,
        )

        with self.assertRaises(APIError):
            client.create_response(instructions="Test.", input_text="Test.")

        self.assertEqual([], sleeps)
        self.assertEqual(1, len(client.mock_calls))


class EnvironmentSafetyTests(unittest.TestCase):
    def test_live_mode_requires_key_in_process_environment(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ConfigurationError, "OPENAI_API_KEY"):
                ResponsesClient()

    def test_key_never_appears_in_client_representation(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": DUMMY_KEY}, clear=True):
            client = ResponsesClient()

        self.assertNotIn(DUMMY_KEY, repr(client))
        self.assertNotIn("api_key", repr(client).lower())

    def test_mock_responses_cannot_be_used_accidentally_in_live_mode(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": DUMMY_KEY}, clear=True):
            with self.assertRaisesRegex(ConfigurationError, "mock=True"):
                ResponsesClient(
                    mock=False,
                    mock_responses=(MockResponse(200, success_body()),),
                )


if __name__ == "__main__":
    unittest.main()
