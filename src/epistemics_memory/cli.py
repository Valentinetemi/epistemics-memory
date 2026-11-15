"""Command-line entry point for the pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

from epistemics_memory.fixtures import load_fixture
from epistemics_memory.runner import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_FIXTURE_PATH,
    DEFAULT_OUTPUT_DIR,
    build_default_mock_client,
    build_live_client,
    load_runner_config,
    run_pilot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="epistemics-memory",
        description="Run the two-fact external-memory contamination pilot.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser(
        "run",
        help="Run the synthetic pilot in mock mode unless --live is explicit.",
    )
    run_parser.add_argument(
        "--live",
        action="store_true",
        help="Use the configured live provider. Omit for offline mock mode.",
    )
    run_parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE_PATH,
        help="Path to the explicitly labeled synthetic fixture.",
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the fixed model, seed, and temperature config.",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Local directory for CSV results and JSONL question traces.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        fixture = load_fixture(args.fixture)
        config = load_runner_config(args.config)
        if args.live:
            client = build_live_client(config)
            run_mode = "live"
        else:
            client = build_default_mock_client(fixture, config)
            run_mode = "mock"
        run_pilot(
            fixture=fixture,
            config=config,
            client=client,
            output_dir=args.output_dir,
            run_mode=run_mode,
        )
    return 0
