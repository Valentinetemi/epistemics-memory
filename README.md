### epistemics-memory

When something is deleted from an AI agent's memory, is it really gone?

AI agents now keep long-term memory about their users, and users can ask them to delete things. This project tests one failure: a user deletes a fact, the system shows it as deleted, but a summary built from that fact keeps shaping the agent's answers, and neither the user nor the developer can see it happening.

Humans work the same way. When a friend tells me something and asks me to forget it, I stop talking about it, but I don't really forget it. Psychology calls this source monitoring: we can lose track of where a belief came from while still holding it (Johnson et al., 1993).

### What it does

A small test harness that checks whether an agent still holds a belief after its source memory is deleted. It compares three memory designs:

1. **Raw notes**
2. **Notes plus summaries**
3. **Provenance tracking** (each summary line traces back to its sources)

Each case plants a false claim in memory, builds summaries, deletes the source, then asks fresh probes. Each answer is scored as persisted, reverted, or unknown/conflict. The scoring rules were frozen before the first live run: see [docs/scoring-rules.md](docs/scoring-rules.md).

### Pilot results

Belief persistence after source deletion (3 synthetic cases × 3 probes per design, `gemini-3.5-flash-lite`, temperature 0):

| Memory design | Persisted | Reverted | Unknown/conflict |
|---|---|---|---|
| Raw notes | 0/9 | 8/9 | 1/9 |
| Notes + summaries | 9/9 | 0/9 | 0/9 |
| Provenance | 0/9 | 8/9 | 1/9 |

When the agent kept summaries, the deleted belief persisted in every probe. The summary acts as a second copy of the deleted fact that deletion does not reach. This is a small pilot signal, not a robust estimate: the probes come from only 3 cases and are not independent, the fixtures are synthetic, and only one model was tested.

### Run it

Requires Python 3.11+.

```bash
pip install -e .

# Offline mock run (no API key needed, fixed seed 1729)
epistemics-memory run

# Live run with Gemini on the first 3 cases
export GEMINI_API_KEY=...   # read only from the environment
epistemics-memory run --live --case-limit 3
```

Results (CSV) and question traces (JSONL) are written to `runs/synthetic_deleted_source_pilot/`, which is not committed. Live API calls cannot be seeded, so live runs are not exactly repeatable.

Run the tests:

```bash
python3 -m unittest discover tests
```

### Next steps

- Finish the full 20-case run (already in `fixtures/`)
- Test more models
- Measure how much useful memory provenance tracking costs

### Notes

All data is synthetic. No real user data was used.

## References

- Johnson, M. K., Hashtroudi, S., and Lindsay, D. S. (1993). Source monitoring. *Psychological Bulletin*, 114(1), 3–28.
- Packer, C., et al. (2023). MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560.
- Park, J. S., et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. UIST 2023.
