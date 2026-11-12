"""Frozen prompts shared across experimental conditions."""

from __future__ import annotations


AGENT_SYSTEM_PROMPT = """You are a concise assistant with an external memory.
Use only the memory entries supplied in the current request; do not use outside knowledge.
Memory entries are listed from oldest to newest. A later entry can summarize or update an earlier entry. When two entries conflict about the same attribute, treat the later clear statement as current unless it explicitly expresses uncertainty.
Source IDs are metadata, not facts.
When asked to create a memory summary, return short standalone factual lines and cite every source ID that supports each line. A summary line may cite notes or earlier summary lines. Do not cite an ID that is not visible in the supplied memory.
When asked a question, answer it directly and briefly. If the visible memory does not establish an answer, say that you do not know.
"""


SUMMARY_TASK = """Create a compact memory summary for a later fresh session.
Keep only useful claims about the entity. Follow the supplied JSON schema exactly.
"""


def probe_task(question: str) -> str:
    return f"Question: {question}"
