"""Offline-first runner for the synthetic two-fact memory pilot.

The Responses API does not expose a seed parameter. The fixed seed controls
runner ordering and deterministic mock generation and is recorded in every
result. Live calls use the configured snapshot and temperature.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from epistemics_memory.fixtures import PilotFixture, PilotItem, load_fixture
from epistemics_memory.memory import MemoryCondition, MemoryStore, SummaryLine
from epistemics_memory.openai_client import MockResponse, ResponseResult, ResponsesClient
from epistemics_memory.prompts import AGENT_SYSTEM_PROMPT, SUMMARY_TASK, probe_task
from epistemics_memory.scoring import (
    SCORE_SCHEMA,
    SCORER_SYSTEM_PROMPT,
    Score,
    parse_score,
    score_request,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_PATH = PROJECT_ROOT / "fixtures" / "synthetic_two_fact_pilot.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "synthetic_two_fact_pilot.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "synthetic_two_fact_pilot"

SYNTHETIC_NOTICE = (
    "SYNTHETIC TEST FIXTURE: Every entity, note, claim, question, and output "
    "in this experiment is invented and is not a real-world fact."
)

SUMMARY_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "name": "synthetic_memory_summary",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "lines": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "source_ids": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["text", "source_ids"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["lines"],
        "additionalProperties": False,
    },
}

SCORE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "name": "synthetic_belief_score",
    "strict": True,
    "schema": SCORE_SCHEMA,
}

CSV_FIELDS = (
    "data_classification",
    "fixture_label",
    "fixture_version",
    "run_mode",
    "condition",
    "item_id",
    "entity",
    "probe_kind",
    "belief_persisted",
    "score_label",
    "model_name",
    "seed",
    "temperature",
    "deleted_source_id",
    "visible_record_count",
)


class RunnerError(RuntimeError):
    """A malformed config, model output, or runner composition."""


@dataclass(frozen=True)
class RunnerConfig:
    config_label: str
    model_name: str
    seed: int
    temperature: float
    summary_sessions: int


@dataclass(frozen=True)
class RunArtifacts:
    results_csv: Path
    question_traces_jsonl: Path
    question_count: int
    persistence_count: int


def load_runner_config(path: Path) -> RunnerConfig:
    """Load the committed deterministic runner configuration."""

    data = json.loads(path.read_text(encoding="utf-8"))
    label = data.get("config_label")
    model_name = data.get("model_name")
    seed = data.get("seed")
    temperature = data.get("temperature")
    summary_sessions = data.get("summary_sessions")
    if not isinstance(label, str) or "SYNTHETIC" not in label:
        raise RunnerError("config_label must identify a SYNTHETIC experiment")
    if not isinstance(model_name, str) or not model_name.strip():
        raise RunnerError("model_name must be a non-empty string")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RunnerError("seed must be an integer")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not 0 <= temperature <= 2
    ):
        raise RunnerError("temperature must be between 0 and 2")
    if summary_sessions != 3:
        raise RunnerError("the two-fact pilot requires exactly three summary sessions")
    return RunnerConfig(
        config_label=label,
        model_name=model_name,
        seed=seed,
        temperature=float(temperature),
        summary_sessions=summary_sessions,
    )


def build_default_mock_client(
    fixture: PilotFixture, config: RunnerConfig
) -> ResponsesClient:
    """Build a deterministic queue that exercises all three memory outcomes."""

    rng = random.Random(config.seed)
    responses: list[MockResponse] = []

    def add_response(text: str) -> None:
        response_id = f"resp_SYNTHETIC_{rng.getrandbits(64):016x}"
        responses.append(
            MockResponse(
                200,
                {
                    "id": response_id,
                    "model": config.model_name,
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": text}],
                        }
                    ],
                },
            )
        )

    for condition in MemoryCondition:
        for item in fixture.items:
            previous_summary_id: str | None = None
            for session_number in range(1, config.summary_sessions + 1):
                if condition is MemoryCondition.RAW_NOTES or previous_summary_id is None:
                    cited_source = item.false_note.source_id
                else:
                    cited_source = previous_summary_id
                add_response(
                    json.dumps(
                        {
                            "lines": [
                                {
                                    "text": item.false_claim,
                                    "source_ids": [cited_source],
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )
                )
                previous_summary_id = (
                    f"summary:{item.item_id}:s{session_number}:l1"
                )

            persisted = condition is MemoryCondition.NOTES_SUMMARIES
            for _probe in item.probes:
                answer = item.false_claim if persisted else item.true_claim
                add_response(answer)
                score = {
                    "false_endorsed": persisted,
                    "true_endorsed": not persisted,
                    "label": "false_endorsement" if persisted else "true_reversion",
                    "explanation": (
                        "The synthetic answer endorses the planted claim."
                        if persisted
                        else "The synthetic answer endorses the remaining true claim."
                    ),
                }
                add_response(json.dumps(score, ensure_ascii=False))

    return ResponsesClient(
        model=config.model_name,
        mock=True,
        mock_responses=responses,
    )


def run_pilot(
    *,
    fixture: PilotFixture,
    config: RunnerConfig,
    client: ResponsesClient,
    output_dir: Path,
    run_mode: str,
) -> RunArtifacts:
    """Run all synthetic items and conditions and write local result artifacts."""

    if "SYNTHETIC" not in fixture.fixture_label:
        raise RunnerError("runner accepts only explicitly SYNTHETIC fixtures")
    if run_mode not in {"mock", "live"}:
        raise RunnerError("run_mode must be 'mock' or 'live'")
    if run_mode == "mock" and not client.mock:
        raise RunnerError("mock run_mode requires a mock client")
    if run_mode == "live" and client.mock:
        raise RunnerError("live run_mode requires a live client")
    if client.model != config.model_name:
        raise RunnerError("client model does not match the runner config")

    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    agent_instructions = f"{SYNTHETIC_NOTICE}\n\n{AGENT_SYSTEM_PROMPT}"
    scorer_instructions = f"{SYNTHETIC_NOTICE}\n\n{SCORER_SYSTEM_PROMPT}"

    for condition in MemoryCondition:
        for item in fixture.items:
            memory = _initialize_memory(condition, item)
            for session_number in range(1, config.summary_sessions + 1):
                summary_input = _request_input(
                    fixture=fixture,
                    memory=memory,
                    task=SUMMARY_TASK,
                )
                summary_response = client.create_response(
                    instructions=agent_instructions,
                    input_text=summary_input,
                    text_format=SUMMARY_FORMAT,
                    temperature=config.temperature,
                )
                memory.save_summary(
                    session_number=session_number,
                    lines=_parse_summary_lines(summary_response),
                )

            memory.delete_source(item.false_note.source_id)
            for probe in item.probes:
                question_input = _request_input(
                    fixture=fixture,
                    memory=memory,
                    task=probe_task(probe.question),
                )
                answer_response = client.create_response(
                    instructions=agent_instructions,
                    input_text=question_input,
                    temperature=config.temperature,
                )
                score_response = client.create_response(
                    instructions=scorer_instructions,
                    input_text=_score_input(
                        fixture=fixture,
                        item=item,
                        candidate=answer_response.output_text,
                    ),
                    text_format=SCORE_FORMAT,
                    temperature=config.temperature,
                )
                score = _parse_score_response(score_response)
                persisted = score.false_endorsed
                visible_records = memory.visible_records()

                rows.append(
                    {
                        "data_classification": "SYNTHETIC",
                        "fixture_label": fixture.fixture_label,
                        "fixture_version": fixture.fixture_version,
                        "run_mode": run_mode,
                        "condition": condition.value,
                        "item_id": item.item_id,
                        "entity": item.entity,
                        "probe_kind": probe.kind,
                        "belief_persisted": str(persisted).lower(),
                        "score_label": score.label,
                        "model_name": config.model_name,
                        "seed": config.seed,
                        "temperature": config.temperature,
                        "deleted_source_id": item.false_note.source_id,
                        "visible_record_count": len(visible_records),
                    }
                )
                traces.append(
                    _question_trace(
                        fixture=fixture,
                        config=config,
                        run_mode=run_mode,
                        condition=condition,
                        item=item,
                        probe_kind=probe.kind,
                        question=probe.question,
                        question_input=question_input,
                        memory=memory,
                        answer_response=answer_response,
                        score_response=score_response,
                        score=score,
                    )
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / f"{run_mode}_results.csv"
    traces_jsonl = output_dir / f"{run_mode}_question_traces.jsonl"
    _write_csv(results_csv, rows)
    _write_jsonl(traces_jsonl, traces)
    return RunArtifacts(
        results_csv=results_csv,
        question_traces_jsonl=traces_jsonl,
        question_count=len(rows),
        persistence_count=sum(row["belief_persisted"] == "true" for row in rows),
    )


def _initialize_memory(
    condition: MemoryCondition, item: PilotItem
) -> MemoryStore:
    memory = MemoryStore(condition, item_id=item.item_id)
    memory.add_note(source_id=item.true_note.source_id, text=item.true_note.text)
    memory.add_note(source_id=item.false_note.source_id, text=item.false_note.text)
    return memory


def _request_input(
    *, fixture: PilotFixture, memory: MemoryStore, task: str
) -> str:
    return (
        f"{SYNTHETIC_NOTICE}\n"
        f"Fixture label: {fixture.fixture_label}\n\n"
        f"{memory.render_visible_memory()}\n\n"
        f"Task:\n{task}"
    )


def _score_input(
    *, fixture: PilotFixture, item: PilotItem, candidate: str
) -> str:
    return (
        f"{SYNTHETIC_NOTICE}\n"
        f"Fixture label: {fixture.fixture_label}\n\n"
        + score_request(
            scope="answer",
            candidate=candidate,
            true_claim=item.true_claim,
            false_claim=item.false_claim,
        )
    )


def _parse_summary_lines(response: ResponseResult) -> tuple[SummaryLine, ...]:
    data = _parse_json_object(response.output_text, context="summary")
    raw_lines = data.get("lines")
    if not isinstance(raw_lines, list) or not raw_lines:
        raise RunnerError("summary output must contain a non-empty lines list")
    lines: list[SummaryLine] = []
    for raw_line in raw_lines:
        if not isinstance(raw_line, dict):
            raise RunnerError("each summary line must be an object")
        text = raw_line.get("text")
        source_ids = raw_line.get("source_ids")
        if not isinstance(text, str) or not isinstance(source_ids, list):
            raise RunnerError("summary line text and source_ids are invalid")
        if any(not isinstance(source_id, str) for source_id in source_ids):
            raise RunnerError("summary source_ids must be strings")
        lines.append(SummaryLine(text=text, source_ids=tuple(source_ids)))
    return tuple(lines)


def _parse_score_response(response: ResponseResult) -> Score:
    return parse_score(_parse_json_object(response.output_text, context="score"))


def _parse_json_object(text: str, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise RunnerError(f"{context} output was not valid JSON") from error
    if not isinstance(value, dict):
        raise RunnerError(f"{context} output must be a JSON object")
    return value


def _question_trace(
    *,
    fixture: PilotFixture,
    config: RunnerConfig,
    run_mode: str,
    condition: MemoryCondition,
    item: PilotItem,
    probe_kind: str,
    question: str,
    question_input: str,
    memory: MemoryStore,
    answer_response: ResponseResult,
    score_response: ResponseResult,
    score: Score,
) -> dict[str, Any]:
    return {
        "trace_type": "SYNTHETIC_QUESTION_TRACE",
        "data_classification": "SYNTHETIC",
        "synthetic": True,
        "fixture_label": fixture.fixture_label,
        "fixture_version": fixture.fixture_version,
        "config_label": config.config_label,
        "run_mode": run_mode,
        "condition": condition.value,
        "item_id": item.item_id,
        "entity": item.entity,
        "probe_kind": probe_kind,
        "model_name": config.model_name,
        "seed": config.seed,
        "temperature": config.temperature,
        "deleted_source_id": item.false_note.source_id,
        "visible_memory": [
            {
                "source_id": record.source_id,
                "text": record.text,
                "kind": record.kind.value,
                "direct_sources": list(record.direct_sources),
            }
            for record in memory.visible_records()
        ],
        "request_input": question_input,
        "question": question,
        "answer": answer_response.output_text,
        "belief_persisted": score.false_endorsed,
        "score": {
            "false_endorsed": score.false_endorsed,
            "true_endorsed": score.true_endorsed,
            "label": score.label,
            "explanation": score.explanation,
        },
        "answer_response": answer_response.raw,
        "score_response": score_response.raw,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, traces: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
