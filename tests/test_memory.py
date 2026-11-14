"""Unit tests for the three external-memory condition implementations."""

from __future__ import annotations

import unittest

from epistemics_memory import CONDITIONS
from epistemics_memory.memory import (
    MemoryCondition,
    MemoryStore,
    RecordKind,
    SummaryLine,
)


TRUE_ID = "note:fixture:true"
FALSE_ID = "note:fixture:false"
TRUE_TEXT = "The test beacon is amber."
FALSE_TEXT = "The test beacon is violet."


def store_with_notes(condition: MemoryCondition) -> MemoryStore:
    store = MemoryStore(condition, item_id="fixture")
    store.add_note(source_id=TRUE_ID, text=TRUE_TEXT)
    store.add_note(source_id=FALSE_ID, text=FALSE_TEXT)
    return store


def visible_ids(store: MemoryStore) -> tuple[str, ...]:
    return tuple(record.source_id for record in store.visible_records())


class ConditionContractTests(unittest.TestCase):
    def test_exported_condition_names_match_memory_enum(self) -> None:
        self.assertEqual(CONDITIONS, tuple(condition.value for condition in MemoryCondition))

    def test_raw_notes_discards_generated_summaries(self) -> None:
        store = store_with_notes(MemoryCondition.RAW_NOTES)

        saved = store.save_summary(
            session_number=1,
            lines=(SummaryLine(FALSE_TEXT, (FALSE_ID,)),),
        )
        store.delete_source(FALSE_ID)

        self.assertEqual((), saved)
        self.assertEqual((TRUE_ID,), visible_ids(store))
        self.assertEqual(2, len(store.records))

    def test_notes_summaries_keeps_derived_text_after_source_deletion(self) -> None:
        store = store_with_notes(MemoryCondition.NOTES_SUMMARIES)
        saved = store.save_summary(
            session_number=1,
            lines=(SummaryLine(FALSE_TEXT, (FALSE_ID,)),),
        )

        store.delete_source(FALSE_ID)

        self.assertEqual((), saved[0].direct_sources)
        self.assertEqual(
            (TRUE_ID, "summary:fixture:s1:l1"),
            visible_ids(store),
        )
        self.assertEqual(FALSE_TEXT, store.visible_summary_records()[0].text)

    def test_provenance_hides_direct_false_derived_line_only(self) -> None:
        store = store_with_notes(MemoryCondition.PROVENANCE)
        saved = store.save_summary(
            session_number=1,
            lines=(
                SummaryLine(TRUE_TEXT, (TRUE_ID,)),
                SummaryLine(FALSE_TEXT, (FALSE_ID,)),
            ),
        )

        store.delete_source(FALSE_ID)

        self.assertEqual((TRUE_ID,), saved[0].direct_sources)
        self.assertEqual((FALSE_ID,), saved[1].direct_sources)
        self.assertEqual(
            (TRUE_ID, "summary:fixture:s1:l1"),
            visible_ids(store),
        )

    def test_provenance_invalidation_is_transitive_across_summaries(self) -> None:
        store = store_with_notes(MemoryCondition.PROVENANCE)
        first = store.save_summary(
            session_number=1,
            lines=(SummaryLine(FALSE_TEXT, (FALSE_ID,)),),
        )[0]
        second = store.save_summary(
            session_number=2,
            lines=(SummaryLine("The later profile repeats violet.", (first.source_id,)),),
        )[0]
        third = store.save_summary(
            session_number=3,
            lines=(
                SummaryLine(
                    "The combined profile still says violet.",
                    (TRUE_ID, second.source_id),
                ),
                SummaryLine(TRUE_TEXT, (TRUE_ID,)),
            ),
        )

        store.delete_source(FALSE_ID)

        self.assertEqual(
            (TRUE_ID, "summary:fixture:s3:l2"),
            visible_ids(store),
        )
        self.assertEqual(
            ("summary:fixture:s3:l2",),
            tuple(record.source_id for record in store.visible_summary_records()),
        )
        self.assertNotIn(first.source_id, visible_ids(store))
        self.assertNotIn(second.source_id, visible_ids(store))

    def test_provenance_requires_sources_visible_before_the_batch(self) -> None:
        store = store_with_notes(MemoryCondition.PROVENANCE)

        with self.assertRaisesRegex(ValueError, "unavailable source IDs"):
            store.save_summary(
                session_number=1,
                lines=(SummaryLine("Unsupported line.", ("missing:id",)),),
            )

        self.assertEqual(2, len(store.records))

    def test_provenance_rejects_source_deleted_before_summary(self) -> None:
        store = store_with_notes(MemoryCondition.PROVENANCE)
        store.delete_source(FALSE_ID)

        with self.assertRaisesRegex(ValueError, FALSE_ID):
            store.save_summary(
                session_number=1,
                lines=(SummaryLine(FALSE_TEXT, (FALSE_ID,)),),
            )

    def test_summary_batch_is_atomic_when_a_later_line_is_invalid(self) -> None:
        store = store_with_notes(MemoryCondition.PROVENANCE)

        with self.assertRaisesRegex(ValueError, "missing:id"):
            store.save_summary(
                session_number=1,
                lines=(
                    SummaryLine(TRUE_TEXT, (TRUE_ID,)),
                    SummaryLine("Unsupported line.", ("missing:id",)),
                ),
            )

        self.assertEqual(2, len(store.records))

    def test_render_visible_memory_preserves_order_and_omits_dependencies(self) -> None:
        store = store_with_notes(MemoryCondition.PROVENANCE)
        store.save_summary(
            session_number=1,
            lines=(SummaryLine(TRUE_TEXT, (TRUE_ID,)),),
        )
        store.delete_source(FALSE_ID)

        self.assertEqual(
            "Memory entries (oldest to newest):\n"
            f"[{TRUE_ID}] {TRUE_TEXT}\n"
            f"[summary:fixture:s1:l1] {TRUE_TEXT}",
            store.render_visible_memory(),
        )

    def test_deleted_summary_hides_its_provenance_descendants(self) -> None:
        store = store_with_notes(MemoryCondition.PROVENANCE)
        parent = store.save_summary(
            session_number=1,
            lines=(SummaryLine(TRUE_TEXT, (TRUE_ID,)),),
        )[0]
        child = store.save_summary(
            session_number=2,
            lines=(SummaryLine("A child line.", (parent.source_id,)),),
        )[0]

        store.delete_source(parent.source_id)

        self.assertNotIn(parent.source_id, visible_ids(store))
        self.assertNotIn(child.source_id, visible_ids(store))

    def test_summary_line_rejects_duplicate_dependencies(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be unique"):
            SummaryLine(TRUE_TEXT, (TRUE_ID, TRUE_ID))

    def test_unknown_deletion_fails_closed(self) -> None:
        store = store_with_notes(MemoryCondition.RAW_NOTES)

        with self.assertRaisesRegex(KeyError, "unknown source ID"):
            store.delete_source("missing:id")

    def test_summary_records_have_expected_kind_and_session(self) -> None:
        store = store_with_notes(MemoryCondition.NOTES_SUMMARIES)

        record = store.save_summary(
            session_number=3,
            lines=(SummaryLine(TRUE_TEXT, (TRUE_ID,)),),
        )[0]

        self.assertIs(RecordKind.SUMMARY, record.kind)
        self.assertEqual(3, record.session_number)

    def test_summary_session_number_must_be_a_positive_integer(self) -> None:
        store = store_with_notes(MemoryCondition.NOTES_SUMMARIES)

        for invalid in (0, -1, 1.5, True, "1"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    store.save_summary(
                        session_number=invalid,  # type: ignore[arg-type]
                        lines=(SummaryLine(TRUE_TEXT, (TRUE_ID,)),),
                    )


if __name__ == "__main__":
    unittest.main()
