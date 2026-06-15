"""Small-sample Streamlit demo required by the challenge sandbox specification."""

from __future__ import annotations

import csv
import io
import json
import tempfile
from pathlib import Path

import streamlit as st

from src.scoring import build_reasoning, score_candidate

st.set_page_config(page_title="Redrob AI Candidate Ranker", layout="wide")
st.title("Redrob AI Candidate Ranker")
st.caption("Hybrid evidence-based ranking: career proof > keyword matching")

uploaded = st.file_uploader("Upload a JSONL sample (up to 100 candidates)", type=["jsonl"])
if uploaded:
    candidates = []
    for line_number, raw in enumerate(uploaded.getvalue().decode("utf-8").splitlines(), 1):
        if raw.strip():
            candidates.append(json.loads(raw))
    if len(candidates) > 100:
        st.error("Please upload at most 100 candidates for the sandbox demo.")
    else:
        scored = []
        for candidate in candidates:
            breakdown = score_candidate(candidate)
            scored.append((breakdown.raw_score, candidate["candidate_id"], candidate, breakdown))
        scored.sort(key=lambda row: (-row[0], row[1]))
        rows = []
        for rank, (score, cid, candidate, breakdown) in enumerate(scored, 1):
            profile = candidate["profile"]
            rows.append({
                "rank": rank,
                "candidate_id": cid,
                "score": round(score, 3),
                "title": profile["current_title"],
                "company": profile["current_company"],
                "experience": profile["years_of_experience"],
                "reasoning": build_reasoning(candidate, breakdown, rank),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})
        st.download_button("Download ranked CSV", buffer.getvalue(), "ranked_sample.csv", "text/csv")
else:
    st.info("Upload a JSONL sample from the provided candidate pool to run the ranker end-to-end.")
