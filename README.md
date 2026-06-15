# AI Recruiter — Evidence-Based Candidate Ranking

A CPU-only candidate-ranking system for the **India Runs / Redrob Data & AI Challenge**. It ranks the top 100 candidates for the released Senior AI Engineer JD without hosted APIs, GPUs, or per-candidate LLM calls.

## Why this approach

The dataset intentionally contains keyword stuffers and inconsistent profiles. Therefore, the model trusts **career-history evidence** much more than a candidate's skills list. It combines:

1. Search, retrieval, ranking, recommendation, and evaluation evidence from work history.
2. Current role and seniority fit.
3. The 5–9 year experience preference.
4. Product-company and recent hands-on engineering evidence.
5. Redrob activity, response, interview, notice-period, and relocation signals.
6. Consistency checks for career dates, claimed experience, current-role mismatches, and implausible expert skills.

The system is transparent: `--review-out` writes every score component for inspection.

## Project structure

```text
.
├── rank.py                 # Full-pool ranking command
├── app.py                  # Streamlit small-sample demo
├── src/scoring.py          # Feature extraction, scoring, anomaly checks, reasoning
├── requirements.txt
├── submission_metadata.yaml
├── outputs/
│   ├── submission.csv
│   └── top100_review.csv
└── data/                   # Put candidates.jsonl here; not committed
```

## Reproduce the ranking

Python 3.10+ is sufficient. The ranker itself uses only the standard library.

```bash
python rank.py \
  --candidates ./data/candidates.jsonl \
  --out ./outputs/submission.csv \
  --review-out ./outputs/top100_review.csv
```

Validate it with the organizer's script:

```bash
python validate_submission.py ./outputs/submission.csv
```

## Run the demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a JSONL file containing at most 100 candidates. The demo ranks them and provides a downloadable CSV.

## Scoring outline

The ranker is a weighted hybrid model. Career descriptions receive high weight for demonstrated ranking/retrieval systems, vector infrastructure, production deployment, and evaluation. Skills-only matches receive almost no weight. A separate behavioral score modifies technical fit, and severe profile inconsistencies are strongly penalized to avoid honeypots.

## Resume-ready description

> Built a CPU-only AI candidate-ranking system over 100,000 profiles using evidence-based hybrid scoring, behavioral signals, and honeypot detection; produced explainable top-100 recommendations in under five minutes without external APIs.

## Important

Before submission, replace placeholders in `submission_metadata.yaml`, rename the final CSV to your registered participant ID, and create a real GitHub repository and Streamlit/Hugging Face sandbox link.
