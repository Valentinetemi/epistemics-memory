"""Provider-neutral response types used by the pilot runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class ResponseResult:
    """The response fields needed by the experiment runner."""

    response_id: str
    model: str
    status: str
    output_text: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class MockResponse:
    """One queued offline transport outcome shared by provider mocks.

    ``status_code=None`` simulates a connection failure. Mock outcomes cannot
    carry request headers, so authorization values cannot enter mock traces.
    """

    status_code: int | None
    body: Mapping[str, Any] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status_code is not None and not 100 <= self.status_code <= 599:
            raise ValueError("mock status_code must be between 100 and 599")

    @classmethod
    def connection_failure(cls) -> MockResponse:
        return cls(status_code=None)


@dataclass(frozen=True)
class MockCall:
    """A safe mock request record containing no headers or API key."""

    payload: dict[str, Any]
    timeout_seconds: float


class ModelClient(Protocol):
    """Small interface implemented by each model provider."""

    model: str
    mock: bool

    def create_response(
        self,
        *,
        instructions: str,
        input_text: str,
        text_format: Mapping[str, Any] | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> ResponseResult: ...
