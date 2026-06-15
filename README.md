# AI Candidate Ranking System

An evidence-based candidate ranking system built for the **India Runs / Redrob Data & AI Challenge**.

This project ranks the **top 100 candidates** for a Senior AI Engineer role from a large candidate dataset. The system is designed to go beyond simple keyword matching by checking real career evidence, work history, skills, activity signals, and profile consistency.

---

## Project Goal

Recruiters often miss strong candidates when they depend only on keyword search.

This project solves that problem by building a ranking system that:

* Understands candidate profiles beyond keywords
* Gives more importance to real work experience
* Penalizes keyword-stuffed or inconsistent profiles
* Produces an explainable top-100 candidate shortlist
* Runs fully on CPU without GPU or paid APIs

---

## Key Features

* Processes **100,000 candidate profiles**
* Selects the best **top 100 candidates**
* Uses a fast two-stage ranking pipeline
* Gives higher weight to career-history evidence
* Uses behavioral signals like:

  * Open-to-work status
  * Recruiter response rate
  * Recent platform activity
  * Notice period
  * Relocation preference
* Detects suspicious or inconsistent profiles
* Generates clear reasoning for each ranked candidate
* Produces a valid `submission.csv` file

---

## Why This Approach?

The dataset contains profiles that may mention AI keywords without real experience.

So, the system does not blindly trust skills like:

```text
AI, ML, LLM, RAG, FAISS, Vector DB
```

Instead, it checks whether the candidate has actually worked on:

```text
ranking systems
semantic search
recommendation systems
retrieval systems
production ML systems
evaluation metrics like NDCG / MRR
A/B testing
```

Career history is trusted more than skill keywords because real work experience is harder to fake.

---

## Tech Stack

```text
Python
Streamlit
Pandas
Standard Python libraries
```

The main ranking pipeline is CPU-only and does not require GPU.

---

## Project Structure

```text
ai-candidate-ranking-system/
│
├── rank.py
├── app.py
├── validate_submission.py
├── requirements.txt
├── submission_metadata.yaml
│
├── src/
│   └── scoring.py
│
├── data/
│   ├── sample_candidates.jsonl
│   └── candidates.jsonl
│
├── outputs/
│   ├── submission.csv
│   └── top100_review.csv
│
└── README.md
```

### File Explanation

| File                           | Purpose                                           |
| ------------------------------ | ------------------------------------------------- |
| `rank.py`                      | Main script to rank candidates                    |
| `src/scoring.py`               | Scoring logic, feature extraction, anomaly checks |
| `app.py`                       | Streamlit demo app                                |
| `validate_submission.py`       | Validates final submission file                   |
| `outputs/submission.csv`       | Final file to submit                              |
| `outputs/top100_review.csv`    | Detailed score breakdown for review               |
| `data/candidates.jsonl`        | Full dataset file, not uploaded to GitHub         |
| `data/sample_candidates.jsonl` | Small sample for demo/testing                     |

---

## How the Ranking Works

The system uses a two-stage ranking approach.

### Stage 1: Fast Shortlisting

The system first scans all candidates and selects a smaller shortlist.

```text
100,000 candidates
        ↓
coarse_score()
        ↓
Top 5,000 candidates
```

This step is fast and filters out clearly weak profiles.

---

### Stage 2: Detailed Candidate Scoring

The top 5,000 candidates are scored deeply using multiple factors.

```text
Top 5,000 candidates
        ↓
detailed scoring
        ↓
Top 100 candidates
```

The final score is based on:

```text
technical evidence
+ title fit
+ experience fit
+ product/company experience
+ behavioral signals
+ logistics signals
+ stability score
- penalties
- anomaly score
```

---

## Scoring Components

### 1. Technical Evidence Score

Checks whether the candidate has worked on:

```text
ranking
retrieval
semantic search
recommendation systems
vector search
production ML
evaluation metrics
```

Career-history evidence gets high weight. Skill-only mentions get very low weight.

---

### 2. Title Score

Rewards relevant roles such as:

```text
Machine Learning Engineer
AI Engineer
Applied Scientist
NLP Engineer
Search Engineer
Recommendation Engineer
```

Senior titles like `Senior`, `Staff`, and `Lead` are also rewarded.

---

### 3. Experience Score

The role prefers candidates with around **5 to 9 years** of experience.

So the score gives higher weight to candidates in this experience range.

---

### 4. Behavioral Score

Uses Redrob platform signals such as:

```text
open to work
recent activity
recruiter response rate
interview completion rate
profile completeness
```

These signals help identify candidates who are not only technically strong but also reachable.

---

### 5. Logistics Score

Checks practical hiring factors:

```text
location
relocation preference
notice period
```

Candidates with shorter notice periods and better location fit receive better scores.

---

### 6. Anomaly Detection

The system checks for profile inconsistencies such as:

```text
invalid career dates
experience mismatch
current company mismatch
current title mismatch
expert skills with very low usage history
```

Suspicious profiles are penalized strongly.

---

## How to Clone the Repository

```bash
git clone https://github.com/harikishankommu/ai-candidate-ranking-system.git
cd ai-candidate-ranking-system
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## Install Requirements

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

Place the full dataset file inside the `data` folder:

```text
data/candidates.jsonl
```

The full dataset is not included in this GitHub repository because it is large.

Your `data` folder should look like this:

```text
data/
├── candidates.jsonl
└── sample_candidates.jsonl
```

---

## Run the Full Ranking Pipeline

```bash
python rank.py --candidates data/candidates.jsonl --out outputs/submission.csv --review-out outputs/top100_review.csv
```

Expected output:

```text
Processed 100,000 candidates, detailed-scored 5,000, and wrote 100 rows to outputs/submission.csv
```

---

## Validate the Submission

```bash
python validate_submission.py outputs/submission.csv
```

Expected output:

```text
Submission is valid.
```

The final file to submit is:

```text
outputs/submission.csv
```

---

## Run the Streamlit Demo

```bash
streamlit run app.py
```

The app will open in the browser.

Upload:

```text
data/sample_candidates.jsonl
```

The demo ranks the sample candidates and shows the output.

---

## Output Files

### `submission.csv`

This is the final ranked output file.

It contains:

```text
candidate_id
rank
score
reasoning
```

Example:

```csv
candidate_id,rank,score,reasoning
CAND_0046525,1,0.990000,"Senior Machine Learning Engineer with strong ranking and retrieval experience..."
```

### `top100_review.csv`

This file is only for internal review.

It contains detailed score components like:

```text
evidence_score
title_score
experience_score
behavior_score
logistics_score
anomaly_score
concerns
```

Do not submit this file unless the organizers specifically ask for it.

---

## Final Hackathon Submission

Submit these three items:

1. GitHub Repository Link

```text
https://github.com/harikishankommu/ai-candidate-ranking-system
```

2. Final Ranked Output File

```text
outputs/submission.csv
```

3. Approach Deck

```text
PPT converted to PDF
```

---

## Results

The system successfully:

* Processed 100,000 candidate profiles
* Shortlisted 5,000 candidates for detailed scoring
* Generated the final top 100 candidates
* Created an explainable `submission.csv`
* Passed the official validation script
* Ran on CPU without external APIs

---

## Resume Description

```text
Built a CPU-only AI candidate ranking system for 100,000 profiles using evidence-based hybrid scoring, behavioral signals, logistics features, and anomaly detection, generating explainable top-100 recruiter-ready recommendations.
```

---

## Future Improvements

The current system is fast, transparent, and reliable. Future versions can be improved using:

### 1. Semantic Embeddings

Use models like:

```text
sentence-transformers/all-MiniLM-L6-v2
BAAI/bge-large-en-v1.5
intfloat/e5-large-v2
```

This can help understand resume and job description meaning more deeply.

---

### 2. Cross-Encoder Reranking

After selecting the top 500 or 1,000 candidates, a reranker can compare the JD and candidate profile together.

Example models:

```text
BAAI/bge-reranker-base
cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

### 3. LLM-Based Explanation

An LLM can be used only for the final top 100 candidates to generate more natural recruiter-style explanations.

This should not be used on all 100,000 candidates because it is slow and expensive.

---

### 4. Better Feature Learning

If labeled data is available, the rule-based weights can be replaced or improved using:

```text
LightGBM
XGBoost
Learning-to-rank models
```

---

### 5. Better Dashboard

The Streamlit app can be improved by adding:

```text
candidate comparison
score breakdown charts
filter by location
filter by notice period
downloadable reports
```

---

## Important Notes

* Do not upload `data/candidates.jsonl` to GitHub.
* Do not upload `.venv` to GitHub.
* Submit only `outputs/submission.csv` as the ranked output.
* Use `top100_review.csv` only for checking and explanation.
* Always validate before final submission.

---

## Author

**Kommu Hari Kishan**
B.Tech Mathematics and Computing
IIT Patna
GitHub: [harikishankommu](https://github.com/harikishankommu)
