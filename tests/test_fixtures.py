"""Tests for the explicitly invented two-fact software fixture."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from epistemics_memory.fixtures import load_fixture


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "two_fact_pilot.json"


class FixtureTests(unittest.TestCase):
    def test_fixture_is_labeled_invented_and_has_fixed_shape(self) -> None:
        fixture = load_fixture(FIXTURE_PATH)

        self.assertIn("TEST FIXTURE", fixture.fixture_label)
        self.assertIn("INVENTED", fixture.fixture_label)
        self.assertEqual(2, len(fixture.items))
        for item in fixture.items:
            self.assertEqual(("direct", "related", "open"), tuple(p.kind for p in item.probes))
            self.assertNotEqual(item.true_claim, item.false_claim)

    def test_fixture_rejects_an_unlabeled_copy(self) -> None:
        data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        data["fixture_label"] = "production facts"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_text(json.dumps(data), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "invented test data"):
                load_fixture(path)


if __name__ == "__main__":
    unittest.main()
