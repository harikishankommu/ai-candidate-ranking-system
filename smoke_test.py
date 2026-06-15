#!/usr/bin/env python3
"""Dependency-free smoke test for the scoring pipeline."""

import json
from pathlib import Path

from src.scoring import score_candidate

sample = Path(__file__).parent / "data" / "sample_candidates.jsonl"
rows = []
with sample.open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            candidate = json.loads(line)
            result = score_candidate(candidate)
            rows.append((result.raw_score, candidate["candidate_id"]))

assert rows, "No sample candidates were loaded"
assert all(isinstance(score, float) for score, _ in rows)
assert len({cid for _, cid in rows}) == len(rows), "Duplicate candidate IDs in sample"
print(f"Smoke test passed for {len(rows)} sample candidates.")
