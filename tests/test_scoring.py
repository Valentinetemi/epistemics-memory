"""Tests for the preregistered automatic-scoring contract."""

from __future__ import annotations

import json
import unittest

from epistemics_memory.scoring import parse_score, score_request


class ScoringTests(unittest.TestCase):
    def test_score_request_names_both_reference_claims(self) -> None:
        request = json.loads(
            score_request(
                scope="answer",
                candidate="It is amber.",
                true_claim="The lamp is amber.",
                false_claim="The lamp is violet.",
            )
        )

        self.assertEqual("answer", request["scope"])
        self.assertEqual("The lamp is amber.", request["true_claim"])
        self.assertEqual("The lamp is violet.", request["planted_false_claim"])

    def test_false_endorsement_takes_label_precedence(self) -> None:
        score = parse_score(
            {
                "false_endorsed": True,
                "true_endorsed": True,
                "label": "false_endorsement",
                "explanation": "The candidate asserts both claims.",
            }
        )

        self.assertTrue(score.false_endorsed)

    def test_inconsistent_false_flag_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "precedence label"):
            parse_score(
                {
                    "false_endorsed": True,
                    "true_endorsed": False,
                    "label": "other",
                    "explanation": "Inconsistent on purpose.",
                }
            )

    def test_unknown_cannot_endorse_a_claim(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot endorse"):
            parse_score(
                {
                    "false_endorsed": False,
                    "true_endorsed": True,
                    "label": "unknown_or_conflict",
                    "explanation": "Inconsistent on purpose.",
                }
            )


if __name__ == "__main__":
    unittest.main()
