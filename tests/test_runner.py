"""Offline integration tests for the synthetic pilot runner."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from epistemics_memory.cli import build_parser, main
from epistemics_memory.fixtures import load_fixture
from epistemics_memory.memory import MemoryCondition
from epistemics_memory.runner import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_FIXTURE_PATH,
    build_default_mock_client,
    load_runner_config,
    run_pilot,
)


class SyntheticRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = load_fixture(DEFAULT_FIXTURE_PATH)
        cls.config = load_runner_config(DEFAULT_CONFIG_PATH)

    def run_mock(self, output_dir: Path):
        client = build_default_mock_client(self.fixture, self.config)
        artifacts = run_pilot(
            fixture=self.fixture,
            config=self.config,
            client=client,
            output_dir=output_dir,
            run_mode="mock",
        )
        return client, artifacts

    def test_config_fixes_model_seed_temperature_and_sessions(self) -> None:
        self.assertIn("SYNTHETIC", self.config.config_label)
        self.assertEqual("gemini", self.config.provider)
        self.assertEqual("gemini-2.5-flash", self.config.model_name)
        self.assertEqual(1729, self.config.seed)
        self.assertEqual(0.0, self.config.temperature)
        self.assertEqual(3, self.config.summary_sessions)

    def test_mock_run_writes_eighteen_question_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client, artifacts = self.run_mock(Path(directory))
            with artifacts.results_csv.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            traces = [
                json.loads(line)
                for line in artifacts.question_traces_jsonl.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

        self.assertEqual(18, artifacts.question_count)
        self.assertEqual(6, artifacts.persistence_count)
        self.assertEqual(18, len(rows))
        self.assertEqual(18, len(traces))
        self.assertEqual(54, len(client.mock_calls))
        self.assertEqual({"gemini"}, {row["provider"] for row in rows})
        self.assertTrue(all(trace["provider"] == "gemini" for trace in traces))
        self.assertEqual(
            {"raw_notes", "notes_summaries", "provenance"},
            {row["condition"] for row in rows},
        )
        self.assertEqual(
            {"direct", "related", "open"},
            {row["probe_kind"] for row in rows},
        )

    def test_mock_persistence_exercises_the_three_memory_designs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _client, artifacts = self.run_mock(Path(directory))
            with artifacts.results_csv.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        persistence_by_condition = {
            condition.value: sum(
                row["belief_persisted"] == "true"
                for row in rows
                if row["condition"] == condition.value
            )
            for condition in MemoryCondition
        }
        self.assertEqual(
            {
                "raw_notes": 0,
                "notes_summaries": 6,
                "provenance": 0,
            },
            persistence_by_condition,
        )

    def test_every_artifact_and_request_is_explicitly_synthetic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client, artifacts = self.run_mock(Path(directory))
            with artifacts.results_csv.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            traces = [
                json.loads(line)
                for line in artifacts.question_traces_jsonl.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

        self.assertTrue(all(row["data_classification"] == "SYNTHETIC" for row in rows))
        self.assertTrue(all("SYNTHETIC" in row["fixture_label"] for row in rows))
        self.assertTrue(all(trace["synthetic"] is True for trace in traces))
        self.assertTrue(
            all(trace["trace_type"] == "SYNTHETIC_QUESTION_TRACE" for trace in traces)
        )
        for call in client.mock_calls:
            self.assertIn("SYNTHETIC", call.payload["instructions"])
            self.assertIn("SYNTHETIC", call.payload["input"])
            self.assertEqual(self.config.model_name, call.payload["model"])
            self.assertEqual(self.config.temperature, call.payload["temperature"])
            self.assertEqual(self.config.seed, call.payload["seed"])

    def test_traces_contain_post_deletion_memory_and_no_deleted_note(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _client, artifacts = self.run_mock(Path(directory))
            traces = [
                json.loads(line)
                for line in artifacts.question_traces_jsonl.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

        for trace in traces:
            visible_ids = {
                record["source_id"] for record in trace["visible_memory"]
            }
            self.assertNotIn(trace["deleted_source_id"], visible_ids)
            self.assertNotIn(
                f"[{trace['deleted_source_id']}]",
                trace["request_input"],
            )
            self.assertIn("answer_response", trace)
            self.assertIn("score_response", trace)

        notes_summary_traces = [
            trace for trace in traces if trace["condition"] == "notes_summaries"
        ]
        other_traces = [
            trace for trace in traces if trace["condition"] != "notes_summaries"
        ]
        self.assertTrue(
            all(
                any(
                    record["kind"] == "summary"
                    for record in trace["visible_memory"]
                )
                for trace in notes_summary_traces
            )
        )
        self.assertTrue(
            all(
                not any(
                    record["kind"] == "summary"
                    for record in trace["visible_memory"]
                )
                for trace in other_traces
            )
        )

    def test_each_probe_is_a_separate_stateless_mock_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client, _artifacts = self.run_mock(Path(directory))

        question_calls = [
            call
            for call in client.mock_calls
            if "Task:\nQuestion:" in call.payload["input"]
        ]
        self.assertEqual(18, len(question_calls))
        for call in question_calls:
            self.assertNotIn("previous_response_id", call.payload)

    def test_fixed_seed_makes_mock_artifacts_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _client_one, first = self.run_mock(root / "first")
            _client_two, second = self.run_mock(root / "second")

            first_csv = first.results_csv.read_text(encoding="utf-8")
            second_csv = second.results_csv.read_text(encoding="utf-8")
            first_jsonl = first.question_traces_jsonl.read_text(encoding="utf-8")
            second_jsonl = second.question_traces_jsonl.read_text(encoding="utf-8")

        self.assertEqual(first_csv, second_csv)
        self.assertEqual(first_jsonl, second_jsonl)

    def test_cli_defaults_to_mock_without_key_or_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {}, clear=True
        ), patch(
            "epistemics_memory.gemini_client.GeminiClient._get_sdk",
            side_effect=AssertionError("default CLI must not construct the SDK"),
        ):
            exit_code = main(["run", "--output-dir", directory])

            self.assertEqual(0, exit_code)
            self.assertTrue((Path(directory) / "mock_results.csv").is_file())
            self.assertTrue(
                (Path(directory) / "mock_question_traces.jsonl").is_file()
            )

    def test_live_mode_requires_an_explicit_flag(self) -> None:
        default_args = build_parser().parse_args(["run"])
        live_args = build_parser().parse_args(["run", "--live"])

        self.assertFalse(default_args.live)
        self.assertTrue(live_args.live)


if __name__ == "__main__":
    unittest.main()
