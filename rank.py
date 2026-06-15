#!/usr/bin/env python3
"""Rank Redrob candidates with a transparent hybrid scoring model."""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

from src.scoring import build_reasoning, coarse_score, score_candidate


def iter_candidates(path: Path) -> Iterator[Dict[str, Any]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc


def normalize_scores(rows: List[Tuple[float, str, Dict[str, Any], Any]]) -> List[float]:
    values = [r[0] for r in rows]
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return [1.0 - i * 1e-6 for i in range(len(rows))]
    return [0.35 + 0.64 * ((value - lo) / (hi - lo)) for value in values]


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank candidates for the Redrob Senior AI Engineer JD.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--review-out", default=None, help="Optional component-score CSV for review")
    parser.add_argument("--top-k", type=int, default=100, help="Number of candidates to output (default: 100)")
    args = parser.parse_args()

    source = Path(args.candidates)
    if not source.exists():
        print(f"Candidate file not found: {source}", file=sys.stderr)
        return 2
    if args.top_k < 1:
        print("--top-k must be positive", file=sys.stderr)
        return 2

    started = time.perf_counter()
    # Stage 1: cheap, broad shortlist over the full pool.
    shortlist_size = max(5000, args.top_k * 30)
    heap: List[Tuple[float, str, Dict[str, Any]]] = []
    processed = 0
    for candidate in iter_candidates(source):
        processed += 1
        cid = candidate.get("candidate_id", "")
        rough = coarse_score(candidate)
        item = (rough, cid, candidate)
        if len(heap) < shortlist_size:
            heapq.heappush(heap, item)
        elif (rough, cid) > (heap[0][0], heap[0][1]):
            heapq.heapreplace(heap, item)

    # Stage 2: detailed evidence, behavioral, logistics, and anomaly scoring.
    scored: List[Tuple[float, str, Dict[str, Any], Any]] = []
    for _, cid, candidate in heap:
        breakdown = score_candidate(candidate)
        scored.append((breakdown.raw_score, cid, candidate, breakdown))

    # Primary score descending, candidate ID ascending for deterministic tie-breaking.
    scored.sort(key=lambda row: (-row[0], row[1]))
    selected = scored[: args.top_k]
    normalized = normalize_scores(selected)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, ((_, cid, candidate, breakdown), score) in enumerate(zip(selected, normalized), 1):
            writer.writerow([cid, rank, f"{score:.6f}", build_reasoning(candidate, breakdown, rank)])

    if args.review_out:
        review_path = Path(args.review_out)
        review_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "candidate_id", "rank", "current_title", "current_company", "years_of_experience",
            "location", "raw_score", "evidence_score", "title_score", "experience_score",
            "product_score", "behavior_score", "logistics_score", "stability_score",
            "penalty_score", "anomaly_score", "evidence_labels", "concerns"
        ]
        with review_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for rank, (_, cid, candidate, breakdown) in enumerate(selected, 1):
                profile = candidate.get("profile", {})
                row = {
                    "candidate_id": cid,
                    "rank": rank,
                    "current_title": profile.get("current_title", ""),
                    "current_company": profile.get("current_company", ""),
                    "years_of_experience": profile.get("years_of_experience", ""),
                    "location": profile.get("location", ""),
                    **breakdown.to_dict(),
                }
                writer.writerow({key: row.get(key, "") for key in fields})

    elapsed = time.perf_counter() - started
    print(f"Processed {processed:,} candidates, detailed-scored {len(scored):,}, and wrote {len(selected)} rows to {output} in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
