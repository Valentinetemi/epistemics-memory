"""Load and validate explicitly synthetic, invented deleted-source cases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Note:
    source_id: str
    text: str


@dataclass(frozen=True)
class Probe:
    kind: str
    question: str
    requires_inference: bool


@dataclass(frozen=True)
class PilotItem:
    item_id: str
    data_classification: str
    case_label: str
    entity: str
    true_note: Note
    false_note: Note
    true_claim: str
    false_claim: str
    probes: tuple[Probe, ...]


@dataclass(frozen=True)
class PilotFixture:
    fixture_label: str
    fixture_version: str
    items: tuple[PilotItem, ...]


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _note(value: Any, field: str) -> Note:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return Note(
        source_id=_required_text(value.get("source_id"), f"{field}.source_id"),
        text=_required_text(value.get("text"), f"{field}.text"),
    )


def load_fixture(path: Path) -> PilotFixture:
    """Load the expanded synthetic fixture, failing closed on design drift."""

    data = json.loads(path.read_text(encoding="utf-8"))
    label = _required_text(data.get("fixture_label"), "fixture_label")
    if not all(marker in label for marker in ("SYNTHETIC", "TEST FIXTURE", "INVENTED")):
        raise ValueError("fixture_label must clearly identify synthetic invented test data")

    raw_items = data.get("items")
    if not isinstance(raw_items, list) or len(raw_items) < 20:
        raise ValueError("the expanded fixture must contain at least 20 items")

    items: list[PilotItem] = []
    source_ids: set[str] = set()
    item_ids: set[str] = set()
    entities: set[str] = set()
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise ValueError(f"items[{index}] must be an object")
        true_note = _note(raw.get("true_note"), f"items[{index}].true_note")
        false_note = _note(raw.get("false_note"), f"items[{index}].false_note")
        if true_note.text == false_note.text:
            raise ValueError(f"items[{index}] must contain conflicting note text")

        raw_probes = raw.get("probes")
        if not isinstance(raw_probes, list) or len(raw_probes) != 3:
            raise ValueError(f"items[{index}] must contain exactly three probes")
        parsed_probes: list[Probe] = []
        for probe_index, probe in enumerate(raw_probes):
            if not isinstance(probe, dict):
                raise ValueError(
                    f"items[{index}].probes[{probe_index}] must be an object"
                )
            requires_inference = probe.get("requires_inference")
            if not isinstance(requires_inference, bool):
                raise ValueError(
                    f"items[{index}].probes[{probe_index}].requires_inference "
                    "must be a boolean"
                )
            parsed_probes.append(
                Probe(
                    kind=_required_text(
                        probe.get("kind"),
                        f"items[{index}].probes[{probe_index}].kind",
                    ),
                    question=_required_text(
                        probe.get("question"),
                        f"items[{index}].probes[{probe_index}].question",
                    ),
                    requires_inference=requires_inference,
                )
            )
        probes = tuple(parsed_probes)
        if tuple(probe.kind for probe in probes) != ("direct", "related", "open"):
            raise ValueError("probe order must be direct, related, open")
        if tuple(probe.requires_inference for probe in probes) != (
            False,
            True,
            False,
        ):
            raise ValueError("only the related probe must require inference")

        data_classification = _required_text(
            raw.get("data_classification"),
            f"items[{index}].data_classification",
        )
        case_label = _required_text(
            raw.get("case_label"), f"items[{index}].case_label"
        )
        if data_classification != "SYNTHETIC" or "SYNTHETIC" not in case_label:
            raise ValueError(f"items[{index}] must be explicitly labeled SYNTHETIC")

        item = PilotItem(
            item_id=_required_text(raw.get("item_id"), f"items[{index}].item_id"),
            data_classification=data_classification,
            case_label=case_label,
            entity=_required_text(raw.get("entity"), f"items[{index}].entity"),
            true_note=true_note,
            false_note=false_note,
            true_claim=_required_text(raw.get("true_claim"), f"items[{index}].true_claim"),
            false_claim=_required_text(raw.get("false_claim"), f"items[{index}].false_claim"),
            probes=probes,
        )
        if item.true_claim != item.true_note.text or item.false_claim != item.false_note.text:
            raise ValueError("scoring claims must exactly match their source notes")
        if item.item_id in item_ids:
            raise ValueError(f"duplicate item_id: {item.item_id}")
        if item.entity in entities:
            raise ValueError(f"duplicate entity: {item.entity}")
        for source_id in (true_note.source_id, false_note.source_id):
            if source_id in source_ids:
                raise ValueError(f"duplicate source_id: {source_id}")
            source_ids.add(source_id)
        item_ids.add(item.item_id)
        entities.add(item.entity)
        items.append(item)

    return PilotFixture(
        fixture_label=label,
        fixture_version=_required_text(data.get("fixture_version"), "fixture_version"),
        items=tuple(items),
    )
