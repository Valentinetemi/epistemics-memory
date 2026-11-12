"""Command-line entry point for the pilot."""

from __future__ import annotations

import argparse

from epistemics_memory import MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="epistemics-memory",
        description="Run the two-fact external-memory contamination pilot.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s 0.1.0 (model: {MODEL})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0
