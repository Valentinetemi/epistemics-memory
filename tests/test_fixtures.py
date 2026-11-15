"""Tests for the expanded synthetic deleted-source software fixture."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from epistemics_memory.fixtures import load_fixture


FIXTURE_PATH = (
    Path(__file__).parents[1] / "fixtures" / "synthetic_two_fact_pilot.json"
)


class FixtureTests(unittest.TestCase):
    def test_fixture_has_at_least_twenty_explicitly_synthetic_cases(self) -> None:
        fixture = load_fixture(FIXTURE_PATH)

        self.assertIn("SYNTHETIC", fixture.fixture_label)
        self.assertIn("TEST FIXTURE", fixture.fixture_label)
        self.assertIn("INVENTED", fixture.fixture_label)
        self.assertGreaterEqual(len(fixture.items), 20)
        for item in fixture.items:
            self.assertEqual("SYNTHETIC", item.data_classification)
            self.assertIn("SYNTHETIC", item.case_label)
            self.assertEqual(("direct", "related", "open"), tuple(p.kind for p in item.probes))
            self.assertEqual(
                (False, True, False),
                tuple(probe.requires_inference for probe in item.probes),
            )
            self.assertNotEqual(item.true_claim, item.false_claim)

    def test_case_identifiers_entities_and_sources_are_unique(self) -> None:
        fixture = load_fixture(FIXTURE_PATH)

        self.assertEqual(len(fixture.items), len({item.item_id for item in fixture.items}))
        self.assertEqual(len(fixture.items), len({item.entity for item in fixture.items}))
        source_ids = [
            source_id
            for item in fixture.items
            for source_id in (item.true_note.source_id, item.false_note.source_id)
        ]
        self.assertEqual(len(source_ids), len(set(source_ids)))

    def test_fixture_rejects_an_unlabeled_copy(self) -> None:
        data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        data["fixture_label"] = "production facts"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_text(json.dumps(data), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "synthetic invented test data"):
                load_fixture(path)

    def test_fixture_rejects_an_unlabeled_case(self) -> None:
        data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        data["items"][0]["data_classification"] = "UNLABELED"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_text(json.dumps(data), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "explicitly labeled SYNTHETIC"):
                load_fixture(path)

    def test_fixture_rejects_related_probe_without_inference_label(self) -> None:
        data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        data["items"][0]["probes"][1]["requires_inference"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_text(json.dumps(data), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "related probe"):
                load_fixture(path)


if __name__ == "__main__":
    unittest.main()
