"""In-memory implementations of the pilot's three memory conditions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class MemoryCondition(str, Enum):
    """The three preregistered external-memory designs."""

    RAW_NOTES = "raw_notes"
    NOTES_SUMMARIES = "notes_summaries"
    PROVENANCE = "provenance"


class RecordKind(str, Enum):
    NOTE = "note"
    SUMMARY = "summary"


@dataclass(frozen=True)
class SummaryLine:
    """One generated summary line and the visible records it cites."""

    text: str
    source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("summary line text must be non-empty")
        if not self.source_ids:
            raise ValueError("summary line must cite at least one source ID")
        if any(not source_id.strip() for source_id in self.source_ids):
            raise ValueError("summary source IDs must be non-empty")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("summary source IDs must be unique")


@dataclass(frozen=True)
class MemoryRecord:
    """An immutable note or stored derived summary line."""

    source_id: str
    text: str
    kind: RecordKind
    direct_sources: tuple[str, ...] = ()
    session_number: int | None = None


class MemoryStore:
    """Store memory records and apply one condition's visibility semantics.

    Provenance records form a directed acyclic graph because a summary line can
    cite only records that were visible before its batch was saved. A provenance
    summary remains visible only while every direct source is recursively visible.
    """

    _ITEM_ID = re.compile(r"^[A-Za-z0-9_-]+$")

    def __init__(self, condition: MemoryCondition | str, *, item_id: str) -> None:
        self.condition = MemoryCondition(condition)
        if not self._ITEM_ID.fullmatch(item_id):
            raise ValueError("item_id must contain only letters, numbers, '_' or '-'")
        self.item_id = item_id
        self._records: dict[str, MemoryRecord] = {}
        self._deleted: set[str] = set()
        self._saved_sessions: set[int] = set()

    @property
    def records(self) -> tuple[MemoryRecord, ...]:
        """Return all stored records, including records currently hidden."""

        return tuple(self._records.values())

    @property
    def deleted_source_ids(self) -> frozenset[str]:
        return frozenset(self._deleted)

    def add_note(self, *, source_id: str, text: str) -> MemoryRecord:
        """Append an original note to memory."""

        self._validate_new_record(source_id=source_id, text=text)
        record = MemoryRecord(
            source_id=source_id,
            text=text,
            kind=RecordKind.NOTE,
        )
        self._records[source_id] = record
        return record

    def save_summary(
        self, *, session_number: int, lines: Iterable[SummaryLine]
    ) -> tuple[MemoryRecord, ...]:
        """Save a summary batch when the condition persists summaries.

        Raw-notes mode validates the batch shape but returns no stored records.
        Notes-plus-summaries mode stores only derived text. Provenance mode also
        stores each line's direct source IDs for recursive invalidation.
        """

        if (
            not isinstance(session_number, int)
            or isinstance(session_number, bool)
            or session_number < 1
        ):
            raise ValueError("session_number must be a positive integer")
        batch = tuple(lines)
        if not batch:
            raise ValueError("a summary session must contain at least one line")
        if any(not isinstance(line, SummaryLine) for line in batch):
            raise TypeError("all summary lines must be SummaryLine instances")

        if self.condition is MemoryCondition.RAW_NOTES:
            return ()
        if session_number in self._saved_sessions:
            raise ValueError(f"summary session already saved: {session_number}")

        visible_before = frozenset(record.source_id for record in self.visible_records())
        pending: list[MemoryRecord] = []
        for line_number, line in enumerate(batch, start=1):
            source_id = self._summary_source_id(session_number, line_number)
            self._validate_new_record(source_id=source_id, text=line.text)
            if self.condition is MemoryCondition.PROVENANCE:
                unavailable = set(line.source_ids) - visible_before
                if unavailable:
                    joined = ", ".join(sorted(unavailable))
                    raise ValueError(f"summary cites unavailable source IDs: {joined}")
                direct_sources = line.source_ids
            else:
                direct_sources = ()
            pending.append(
                MemoryRecord(
                    source_id=source_id,
                    text=line.text,
                    kind=RecordKind.SUMMARY,
                    direct_sources=direct_sources,
                    session_number=session_number,
                )
            )

        for record in pending:
            self._records[record.source_id] = record
        self._saved_sessions.add(session_number)
        return tuple(pending)

    def delete_source(self, source_id: str) -> None:
        """Hide a record and, in provenance mode, all of its descendants."""

        if source_id not in self._records:
            raise KeyError(f"unknown source ID: {source_id}")
        self._deleted.add(source_id)

    def visible_records(self) -> tuple[MemoryRecord, ...]:
        """Return visible records in their original insertion order."""

        memo: dict[str, bool] = {}
        return tuple(
            record
            for record in self._records.values()
            if self._is_visible(record.source_id, memo=memo, visiting=set())
        )

    def visible_summary_records(self) -> tuple[MemoryRecord, ...]:
        return tuple(
            record
            for record in self.visible_records()
            if record.kind is RecordKind.SUMMARY
        )

    def render_visible_memory(self) -> str:
        """Render exactly the memory block that a later caller can supply."""

        records = self.visible_records()
        if not records:
            return "Memory entries (oldest to newest):\n(none)"
        lines = ["Memory entries (oldest to newest):"]
        lines.extend(f"[{record.source_id}] {record.text}" for record in records)
        return "\n".join(lines)

    def _is_visible(
        self, source_id: str, *, memo: dict[str, bool], visiting: set[str]
    ) -> bool:
        if source_id in memo:
            return memo[source_id]
        if source_id in self._deleted:
            memo[source_id] = False
            return False

        record = self._records.get(source_id)
        if record is None:
            memo[source_id] = False
            return False
        if (
            self.condition is not MemoryCondition.PROVENANCE
            or record.kind is RecordKind.NOTE
        ):
            memo[source_id] = True
            return True
        if source_id in visiting:
            raise RuntimeError(f"provenance cycle detected at {source_id}")

        visiting.add(source_id)
        visible = bool(record.direct_sources) and all(
            self._is_visible(dependency, memo=memo, visiting=visiting)
            for dependency in record.direct_sources
        )
        visiting.remove(source_id)
        memo[source_id] = visible
        return visible

    def _summary_source_id(self, session_number: int, line_number: int) -> str:
        return f"summary:{self.item_id}:s{session_number}:l{line_number}"

    def _validate_new_record(self, *, source_id: str, text: str) -> None:
        if not source_id.strip():
            raise ValueError("source_id must be non-empty")
        if not text.strip():
            raise ValueError("record text must be non-empty")
        if source_id in self._records:
            raise ValueError(f"duplicate source ID: {source_id}")
