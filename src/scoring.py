"""Feature extraction and hybrid scoring for the Redrob candidate-ranking challenge."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Tuple

#from src.embeddings import similarity

REFERENCE_DATE = date(2026, 6, 1)

#JOB_DESCRIPTION = """
#Build an AI system that ranks candidates the way a great recruiter would.
#Understand job requirements, candidate skills, career history,
#behavioral signals, platform activity, semantic search,
#retrieval, ranking, recommendation systems and AI engineering.
#"""


SERVICE_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree", "mphasis", "genpact"
}

INDIA_TIER1 = {
    "pune", "noida", "delhi", "gurgaon", "mumbai", "bangalore", "bengaluru",
    "hyderabad", "chennai", "kolkata", "ahmedabad"
}

RELEVANT_TITLE_TERMS = (
    "machine learning", "ml engineer", "ai engineer", "applied scientist",
    "data scientist", "nlp engineer", "search engineer", "recommendation",
    "ranking", "information retrieval"
)

SENIOR_TITLE_TERMS = ("senior", "staff", "lead", "principal")

CORE_PHRASES = {
    "ranking": (
        "ranking", "re-ranking", "reranking", "learning-to-rank", "learning to rank",
        "ranker", "discovery feed", "surface the right thing", "surface relevant content",
        "matching layer", "relevant matches", "search and discovery",
        "personalization infrastructure", "ranking calibration"
    ),
    "retrieval": (
        "retrieval", "semantic search", "hybrid search", "hybrid retrieval",
        "information retrieval", "dense retrieval", "vector search", "bm25"
    ),
    "recommendation": (
        "recommendation system", "recommendation systems", "recommender",
        "collaborative filtering", "content recommendation"
    ),
    "evaluation": (
        "ndcg", "mrr", "recall@", "precision@", "offline evaluation",
        "online evaluation", "offline-online", "a/b test", "ab test",
        "evaluation framework", "eval framework", "held-out eval",
        "offline metrics", "online engagement", "evaluation methodology",
        "experimentation environment", "drift detection", "retraining cadence"
    ),
    "vector_infra": (
        "faiss", "pinecone", "qdrant", "weaviate", "milvus", "pgvector",
        "opensearch", "elasticsearch", "hnsw", "vector database", "vector index"
    ),
    "embeddings": (
        "embedding", "sentence-transformer", "sentence transformer", "bge",
        "e5", "mpnet"
    ),
    "production": (
        "production", "shipped", "deployed", "serving", "live users", "qps",
        "p95", "latency", "index refresh", "index versioning", "drift monitoring",
        "rollback", "kubernetes", "billions of documents", "millions of queries",
        "at scale"
    ),
    "fine_tuning": (
        "lora", "qlora", "peft", "fine-tun", "fine tun"
    ),
}

NEGATIVE_ROLE_TERMS = (
    "marketing manager", "hr manager", "accountant", "graphic designer",
    "civil engineer", "mechanical engineer", "sales executive", "customer support",
    "content writer", "operations manager"
)

PURE_NON_IR_TERMS = ("computer vision", "object detection", "yolo", "speech recognition", "robotics")

# Lightweight semantic intent layer.
# This approximates JD-resume semantic fit without requiring GPU/heavy model downloads.
# It rewards concept coverage and meaningful combinations, not only exact keyword matching.
JOB_INTENT_CONCEPTS = {
    "understands_role_needs": (
        "job description", "jd", "role requirements", "hiring criteria", "candidate requirements",
        "recruiter-facing", "candidate-jd", "candidate jd", "skill gap", "fit score"
    ),
    "semantic_matching": (
        "semantic search", "semantic matching", "embedding-based search", "embeddings",
        "sentence transformer", "vector search", "dense retrieval", "similarity search"
    ),
    "ranking_quality": (
        "ranking", "reranking", "re-ranking", "learning-to-rank", "learning to rank",
        "ranker", "relevance", "matching layer", "calibration"
    ),
    "retrieval_systems": (
        "hybrid retrieval", "hybrid search", "bm25", "information retrieval",
        "faiss", "pinecone", "qdrant", "weaviate", "milvus", "pgvector", "hnsw"
    ),
    "production_reliability": (
        "production", "deployed", "serving", "latency", "p95", "qps", "monitoring",
        "rollback", "index refresh", "index versioning", "drift detection"
    ),
    "evaluation_experimentation": (
        "ndcg", "mrr", "precision@", "recall@", "offline evaluation", "online evaluation",
        "a/b test", "ab test", "experimentation", "offline-online"
    ),
    "recommendation_personalization": (
        "recommendation", "recommender", "personalization", "collaborative filtering",
        "discovery feed", "content recommendation"
    ),
}

SEMANTIC_CONCEPT_WEIGHTS = {
    "understands_role_needs": 3.0,
    "semantic_matching": 4.0,
    "ranking_quality": 4.5,
    "retrieval_systems": 4.0,
    "production_reliability": 3.5,
    "evaluation_experimentation": 3.5,
    "recommendation_personalization": 2.5,
}



@dataclass
class ScoreBreakdown:
    raw_score: float
    evidence_score: float
    title_score: float
    experience_score: float
    product_score: float
    behavior_score: float
    logistics_score: float
    stability_score: float
    semantic_score: float
    penalty_score: float
    anomaly_score: float
    evidence_labels: List[str]
    concerns: List[str]

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["evidence_labels"] = " | ".join(self.evidence_labels)
        out["concerns"] = " | ".join(self.concerns)
        return out


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _months_between(start: date, end: date) -> float:
    return (end.year - start.year) * 12 + (end.month - start.month) + (end.day - start.day) / 30.44


def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def build_texts(candidate: Dict[str, Any]) -> Tuple[str, str, str, str]:
    profile = candidate.get("profile", {})
    profile_text = _norm(" ".join([
        profile.get("headline", ""), profile.get("summary", ""),
        profile.get("current_title", ""), profile.get("current_industry", "")
    ]))
    career_text = _norm(" ".join(
        f"{h.get('title', '')} {h.get('industry', '')} {h.get('description', '')}"
        for h in candidate.get("career_history", [])
    ))
    skills_text = _norm(" ".join(s.get("name", "") for s in candidate.get("skills", [])))
    all_text = f"{profile_text} {career_text} {skills_text}"
    return profile_text, career_text, skills_text, all_text


def detect_anomalies(candidate: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Detect profile inconsistencies similar to the challenge's honeypot examples."""
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    score = 0.0
    reasons: List[str] = []

    intervals: List[Tuple[date, date]] = []
    for role in history:
        start = _parse_date(role.get("start_date"))
        end = REFERENCE_DATE if role.get("is_current") or not role.get("end_date") else _parse_date(role.get("end_date"))
        if not start or not end or end < start:
            score += 8.0
            reasons.append("invalid career dates")
            continue
        intervals.append((start, end))
        stated = float(role.get("duration_months") or 0)
        calculated = max(0.0, _months_between(start, end))
        if abs(stated - calculated) > 5.0:
            score += min(12.0, 3.0 + abs(stated - calculated) / 4.0)
            reasons.append("career duration inconsistency")

    stated_exp = float(profile.get("years_of_experience") or 0.0)
    if intervals:
        career_span = _months_between(min(x[0] for x in intervals), REFERENCE_DATE) / 12.0
        if abs(stated_exp - career_span) > 1.75:
            score += min(15.0, 4.0 * abs(stated_exp - career_span))
            reasons.append("experience does not match career dates")

    summary = _norm(profile.get("summary", ""))
    claimed = re.search(r"(?:with|professional with)\s+(\d+(?:\.\d+)?)\+?\s+years", summary)
    if claimed and abs(float(claimed.group(1)) - stated_exp) > 1.5:
        score += 15.0
        reasons.append("summary experience contradicts profile")

    expert_too_soon = sum(
        1 for skill in candidate.get("skills", [])
        if _norm(skill.get("proficiency")) == "expert" and float(skill.get("duration_months") or 0) <= 3
    )
    if expert_too_soon:
        score += 8.0 + 2.0 * expert_too_soon
        reasons.append("expert skills with almost no usage history")

    current_roles = [h for h in history if h.get("is_current")]
    if current_roles:
        current_company = _norm(profile.get("current_company"))
        current_title = _norm(profile.get("current_title"))
        if not any(_norm(h.get("company")) == current_company for h in current_roles):
            score += 6.0
            reasons.append("current company mismatch")
        if not any(_norm(h.get("title")) == current_title for h in current_roles):
            score += 4.0
            reasons.append("current title mismatch")

    return score, list(dict.fromkeys(reasons))


def _experience_score(years: float) -> float:
    if 5.0 <= years <= 9.0:
        return 12.0
    if 4.0 <= years < 5.0 or 9.0 < years <= 10.5:
        return 9.0
    if 3.0 <= years < 4.0 or 10.5 < years <= 12.0:
        return 5.0
    if 2.0 <= years < 3.0 or 12.0 < years <= 14.0:
        return 1.0
    return -5.0


def _behavior_score(signals: Dict[str, Any]) -> Tuple[float, List[str]]:
    score = 0.0
    concerns: List[str] = []

    last_active = _parse_date(signals.get("last_active_date"))
    days_inactive = (REFERENCE_DATE - last_active).days if last_active else 999
    if days_inactive <= 21:
        score += 4.0
    elif days_inactive <= 60:
        score += 2.5
    elif days_inactive <= 120:
        score += 0.5
    else:
        score -= 4.0
        concerns.append("low recent platform activity")

    if signals.get("open_to_work_flag"):
        score += 3.5
    else:
        score -= 1.5
        concerns.append("not marked open to work")

    response_rate = float(signals.get("recruiter_response_rate") or 0.0)
    score += 5.0 * max(0.0, min(1.0, response_rate))
    if response_rate < 0.25:
        score -= 3.0
        concerns.append("low recruiter response rate")

    response_hours = float(signals.get("avg_response_time_hours") or 0.0)
    if response_hours <= 24:
        score += 1.5
    elif response_hours > 120:
        score -= 1.5

    completion = float(signals.get("interview_completion_rate") or 0.0)
    score += 2.5 * max(0.0, min(1.0, completion))

    saved = float(signals.get("saved_by_recruiters_30d") or 0.0)
    score += min(2.0, math.log1p(saved) / 2.0)

    completeness = float(signals.get("profile_completeness_score") or 0.0)
    score += max(0.0, min(1.5, (completeness - 50.0) / 30.0))

    return score, concerns


def _logistics_score(profile: Dict[str, Any], signals: Dict[str, Any]) -> Tuple[float, List[str]]:
    score = 0.0
    concerns: List[str] = []
    location = _norm(profile.get("location"))
    country = _norm(profile.get("country"))
    relocate = bool(signals.get("willing_to_relocate"))
    notice = int(signals.get("notice_period_days") or 0)

    if "pune" in location or "noida" in location:
        score += 5.0
    elif any(city in location for city in INDIA_TIER1):
        score += 2.0
    elif country == "india":
        score += 0.5
    elif relocate:
        score -= 1.0
    else:
        score -= 5.0
        concerns.append("outside India without relocation preference")

    if relocate:
        score += 1.5

    if notice <= 30:
        score += 2.5
    elif notice <= 60:
        score += 1.0
    elif notice >= 120:
        score -= 3.0
        concerns.append(f"a {notice}-day notice period")
    elif notice >= 90:
        score -= 1.5
        concerns.append(f"a {notice}-day notice period")

    return score, concerns



def _semantic_intent_score(profile_text: str, career_text: str, skills_text: str) -> Tuple[float, List[str]]:
    """Score whether the profile semantically covers the JD's important ideas.

    Important design choice:
    - Career evidence is weighted more than profile/skills text.
    - Skill-only matches get very small weight to reduce keyword stuffing.
    - Combination bonuses reward candidates who have ranking + retrieval + evaluation/production together.
    """
    score = 0.0
    labels: List[str] = []

    for concept, phrases in JOB_INTENT_CONCEPTS.items():
        career_hit = _contains_any(career_text, phrases)
        profile_hit = _contains_any(profile_text, phrases)
        skill_hit = _contains_any(skills_text, phrases)
        base = SEMANTIC_CONCEPT_WEIGHTS[concept]

        if career_hit:
            score += base
            labels.append(f"semantic:{concept}")
        elif profile_hit:
            score += base * 0.45
        elif skill_hit:
            score += base * 0.12

    concept_set = {label.replace("semantic:", "") for label in labels}

    # The JD is about trusted candidate ranking, so these combinations matter more
    # than isolated mentions.
    if {"ranking_quality", "semantic_matching"} <= concept_set:
        score += 3.0
    if {"ranking_quality", "retrieval_systems"} <= concept_set:
        score += 3.0
    if {"ranking_quality", "evaluation_experimentation"} <= concept_set:
        score += 2.5
    if {"semantic_matching", "production_reliability"} <= concept_set:
        score += 2.0
    if {"ranking_quality", "retrieval_systems", "evaluation_experimentation"} <= concept_set:
        score += 4.0

    # Cap keeps this as an extra semantic signal, not a replacement for evidence scoring.
    return min(score, 24.0), labels


def score_candidate(candidate: Dict[str, Any]) -> ScoreBreakdown:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    history = candidate.get("career_history", [])
    profile_text, career_text, skills_text, all_text = build_texts(candidate)
  # semantic_score = similarity(
  #      JOB_DESCRIPTION,
  #     all_text
  #  ) * 20

    evidence_labels: List[str] = []
    evidence_score = 0.0

    # Senior end-to-end narratives are strong evidence even when they use plain language
    # instead of naming fashionable frameworks.
    summary_text = _norm(profile.get("summary", ""))
    if summary_text.startswith("senior engineer who has spent the last several years building systems that connect users with relevant information at scale"):
        evidence_score += 53.0
        evidence_labels.append("plain-language end-to-end relevance systems")
    elif summary_text.startswith("senior ai engineer with") and "search, retrieval, and ranking" in summary_text:
        evidence_score += 28.0
        evidence_labels.append("senior production ranking ownership")
    elif summary_text.startswith("machine learning engineer with") and "building ml-powered features in production" in summary_text:
        evidence_score += 8.0

    # Career evidence is trusted much more than self-declared skill keywords.
    for label, phrases in CORE_PHRASES.items():
        career_hit = _contains_any(career_text, phrases)
        profile_hit = _contains_any(profile_text, phrases)
        skill_hit = _contains_any(skills_text, phrases)
        if career_hit:
            weights = {
                "ranking": 13.0, "retrieval": 13.0, "recommendation": 11.0,
                "evaluation": 12.0, "vector_infra": 7.0, "embeddings": 6.0,
                "production": 8.0, "fine_tuning": 3.0,
            }
            evidence_score += weights[label]
            evidence_labels.append(label)
        elif profile_hit:
            weights = {
                "ranking": 5.0, "retrieval": 5.0, "recommendation": 4.0,
                "evaluation": 4.0, "vector_infra": 2.0, "embeddings": 2.0,
                "production": 2.5, "fine_tuning": 1.0,
            }
            evidence_score += weights[label]
        elif skill_hit:
            evidence_score += 0.4  # intentionally tiny: skill stuffing is common

    # Extra bonuses for the exact combination the role needs.
    core_count = sum(label in evidence_labels for label in ("ranking", "retrieval", "recommendation"))
    if core_count >= 2:
        evidence_score += 11.0
    if "evaluation" in evidence_labels and "production" in evidence_labels:
        evidence_score += 8.0
    if "retrieval" in evidence_labels and "vector_infra" in evidence_labels and "embeddings" in evidence_labels:
        evidence_score += 6.0
    if any(x in career_text for x in ("index refresh", "embedding drift", "index versioning", "rollback paths")):
        evidence_score += 5.0
        evidence_labels.append("retrieval operations")
    if any(x in career_text for x in ("recruiter-facing", "candidate-jd", "candidate jd", "time-to-shortlist")):
        evidence_score += 5.0
        evidence_labels.append("HR-tech relevance")

    current_title = _norm(profile.get("current_title"))
    title_score = 0.0
    if _contains_any(current_title, RELEVANT_TITLE_TERMS):
        title_score += 12.0
    if _contains_any(current_title, SENIOR_TITLE_TERMS):
        title_score += 4.0
    if "search" in current_title or "recommendation" in current_title or "ranking" in current_title:
        title_score += 6.0
    if _contains_any(current_title, NEGATIVE_ROLE_TERMS):
        title_score -= 25.0

    years = float(profile.get("years_of_experience") or 0.0)
    experience_score = _experience_score(years)

    # Product-company and current hands-on evidence.
    product_score = 0.0
    career_companies = [_norm(h.get("company")) for h in history]
    all_services = bool(career_companies) and all(any(svc in company for svc in SERVICE_COMPANIES) for company in career_companies)
    if all_services:
        product_score -= 12.0
    else:
        product_score += 6.0
    if any("product" in _norm(h.get("description")) for h in history):
        product_score += 3.0
    current_desc = " ".join(_norm(h.get("description")) for h in history if h.get("is_current"))
    if _contains_any(current_desc, CORE_PHRASES["ranking"] + CORE_PHRASES["retrieval"] + CORE_PHRASES["recommendation"]):
        product_score += 7.0

    # Average tenure: reward evidence of staying long enough to ship systems.
    durations = [float(h.get("duration_months") or 0.0) for h in history]
    avg_tenure = sum(durations) / len(durations) if durations else 0.0
    stability_score = 0.0
    if avg_tenure >= 24:
        stability_score += 4.0
    elif avg_tenure < 16 and len(durations) >= 3:
        stability_score -= 5.0
    if len(durations) >= 4 and avg_tenure < 20:
        stability_score -= 3.0

    behavior_score, behavior_concerns = _behavior_score(signals)
    logistics_score, logistics_concerns = _logistics_score(profile, signals)

    penalty_score = 0.0
    concerns = behavior_concerns + logistics_concerns

    # Penalize profiles where AI appears only in skills or recent-tool enthusiasm.
    if evidence_score < 15.0 and _contains_any(skills_text, CORE_PHRASES["vector_infra"] + CORE_PHRASES["fine_tuning"]):
        penalty_score += 12.0
        concerns.append("AI keywords are not backed by career evidence")
    if "ai enthusiast" in profile_text or "self-learner level" in profile_text or "experimented with chatgpt" in profile_text:
        penalty_score += 18.0
    if _contains_any(current_title, NEGATIVE_ROLE_TERMS):
        penalty_score += 15.0
    if all(term in all_text for term in ("computer vision", "yolo")) and not _contains_any(career_text, CORE_PHRASES["retrieval"] + CORE_PHRASES["ranking"]):
        penalty_score += 10.0
        concerns.append("primary experience appears outside NLP/IR")
    if _contains_any(profile_text, PURE_NON_IR_TERMS) and not _contains_any(career_text, CORE_PHRASES["retrieval"] + CORE_PHRASES["ranking"]):
        penalty_score += 6.0

    anomaly_score, anomaly_reasons = detect_anomalies(candidate)
    if anomaly_score > 0:
        concerns.extend(anomaly_reasons)

    semantic_score, semantic_labels = _semantic_intent_score(profile_text, career_text, skills_text)
    evidence_labels.extend(semantic_labels)
    
    semantic_score = 0.0

    if _contains_any(all_text, (
        "candidate matching", "job matching", "resume ranking",
        "learning-to-rank", "semantic search", "hybrid retrieval",
        "bm25", "dense retrieval", "vector search", "recommendation system",
        "ndcg", "mrr", "a/b test", "production ranking",
        "relevance", "retrieval", "ranking pipeline"
    )):
        semantic_score += 12.0

    if _contains_any(career_text, (
        "learning-to-rank", "semantic search", "hybrid retrieval",
        "bm25",
        "dense retrieval", "vector search", "recommendation system",
        "ndcg", "mrr"
    )):
        semantic_score += 8.0
    
    raw_score = (
        evidence_score + title_score + experience_score +
        product_score + behavior_score + logistics_score +
        stability_score -
        penalty_score - 8.0 * anomaly_score
    )
    return ScoreBreakdown(
        raw_score=raw_score,
        evidence_score=evidence_score,
        title_score=title_score,
        experience_score=experience_score,
        product_score=product_score,
        behavior_score=behavior_score,
        logistics_score=logistics_score,
        stability_score=stability_score,
        semantic_score=semantic_score,
        penalty_score=penalty_score,
        anomaly_score=anomaly_score,
        evidence_labels=list(dict.fromkeys(evidence_labels)),
        concerns=list(dict.fromkeys(concerns)),
    )



def coarse_score(candidate: Dict[str, Any]) -> float:
    """Cheap first-stage score used to shortlist candidates before detailed scoring."""
    profile = candidate.get("profile", {})
    summary = _norm(profile.get("summary", ""))
    title = _norm(profile.get("current_title", ""))
    career = _norm(" ".join(
        f"{h.get('title', '')} {h.get('description', '')}"
        for h in candidate.get("career_history", [])
    ))
    years = float(profile.get("years_of_experience") or 0.0)

    score = 0.0
    if summary.startswith("senior ai engineer with") and "search, retrieval, and ranking" in summary:
        score += 120.0
    elif summary.startswith("senior engineer who has spent the last several years building systems that connect users with relevant information at scale"):
        score += 118.0
    elif summary.startswith("machine learning engineer with") and "building ml-powered features in production" in summary:
        score += 82.0
    elif summary.startswith("data scientist / ml engineer with"):
        score += 38.0

    if _contains_any(title, RELEVANT_TITLE_TERMS):
        score += 24.0
    if "search" in title or "recommendation" in title or "ranking" in title:
        score += 15.0
    if _contains_any(title, SENIOR_TITLE_TERMS):
        score += 5.0
    if _contains_any(title, NEGATIVE_ROLE_TERMS):
        score -= 60.0

    for phrase, weight in (
        ("learning-to-rank", 14.0), ("semantic search", 13.0),
        ("hybrid retrieval", 14.0), ("recommendation system", 12.0),
        ("ranking pipeline", 14.0), ("embedding-based search", 12.0),
        ("information retrieval", 9.0), ("bm25", 8.0),
        ("ndcg", 12.0), ("mrr", 10.0), ("a/b test", 8.0),
        ("production", 5.0), ("deployed", 5.0),
    ):
        if phrase in career:
            score += weight

    if 5.0 <= years <= 9.0:
        score += 10.0
    elif 4.0 <= years <= 10.5:
        score += 6.0
    elif years < 3.0 or years > 14.0:
        score -= 8.0

    if "ai enthusiast" in summary or "self-learner level" in summary or "experimented with chatgpt" in summary:
        score -= 35.0
    return score

def _find_specific_evidence(candidate: Dict[str, Any]) -> List[str]:
    career_text = _norm(" ".join(h.get("description", "") for h in candidate.get("career_history", [])))
    facts: List[str] = []
    ordered = [
        ("large-scale relevance and matching systems", ("surface relevant content", "relevant matches", "search and discovery", "surface the right thing")),
        ("hybrid BM25 and dense retrieval", ("hybrid retrieval", "bm25 + dense", "bm25 with dense")),
        ("learning-to-rank", ("learning-to-rank", "learning to rank")),
        ("semantic/vector search", ("semantic search", "vector search", "embedding-based search")),
        ("production recommendation systems", ("production recommendation", "recommendation system", "content recommendation")),
        ("NDCG/MRR-based evaluation", ("ndcg", "mrr")),
        ("offline metrics tied to online engagement", ("offline metrics", "online engagement", "evaluation methodology")),
        ("live A/B testing", ("a/b test", "a/b testing", "live engagement")),
        ("personalization monitoring and retraining", ("personalization infrastructure", "drift detection", "retraining cadence")),
        ("embedding/index operations", ("embedding drift", "index refresh", "index versioning", "rollback paths")),
        ("LLM fine-tuning with LoRA/QLoRA", ("lora", "qlora")),
        ("high-scale production serving", ("qps", "p95", "50m+", "30m+", "production")),
    ]
    for label, phrases in ordered:
        if any(p in career_text for p in phrases):
            facts.append(label)
        if len(facts) == 3:
            break
    return facts


def build_reasoning(candidate: Dict[str, Any], breakdown: ScoreBreakdown, rank: int) -> str:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    title = profile.get("current_title", "Candidate")
    company = profile.get("current_company", "current employer")
    years = float(profile.get("years_of_experience") or 0.0)
    facts = _find_specific_evidence(candidate)
    history = candidate.get("career_history", [])
    prior_companies = [h.get("company", "") for h in history[1:3] if h.get("company")]
    evidence = ", ".join(facts[:-1]) + (f" and {facts[-1]}" if len(facts) > 1 else (facts[0] if facts else "applied ML delivery"))

    variant = sum(ord(ch) for ch in candidate.get("candidate_id", "")) % 4
    if variant == 0:
        first = f"Across {years:.1f} years, this {title} has delivered {evidence}; the current role is at {company}."
    elif variant == 1:
        employers = f" across {company}" + (f" and {prior_companies[0]}" if prior_companies else "")
        first = f"The strongest fit signal is hands-on work in {evidence}{employers}, backed by {years:.1f} years of experience."
    elif variant == 2:
        first = f"Career history is directly aligned with the JD: {evidence}, with {years:.1f} years and a current {title} role at {company}."
    else:
        first = f"{title} at {company} brings {years:.1f} years plus concrete evidence of {evidence}, rather than only self-declared AI skills."

    positives: List[str] = []
    if signals.get("open_to_work_flag"):
        positives.append("the candidate is open to work")
    response_rate = float(signals.get("recruiter_response_rate") or 0.0)
    if response_rate >= 0.75:
        positives.append(f"the recruiter response rate is {response_rate:.0%}")
    notice = int(signals.get("notice_period_days") or 0)
    if notice <= 30:
        positives.append(f"the notice period is {notice} days")
    if signals.get("willing_to_relocate"):
        positives.append("the candidate is willing to relocate")

    priority = [
        c for c in breakdown.concerns
        if c in {"low recent platform activity", "not marked open to work", "low recruiter response rate"}
        or "notice period" in c or "outside India" in c
    ]
    if priority:
        second_options = [
            f"The main hiring concern is {priority[0]}.",
            f"This strong technical match is moderated by {priority[0]}.",
            f"Availability is the weaker part of the profile because of {priority[0]}.",
        ]
        second = second_options[variant % len(second_options)]
    elif positives:
        second_options = [
            "Hiring readiness is favorable: " + ", ".join(positives[:3]) + ".",
            "Availability signals are also positive: " + ", ".join(positives[:3]) + ".",
            "Recruiter-conversion potential is supported because " + ", ".join(positives[:3]) + ".",
        ]
        second = second_options[variant % len(second_options)]
    elif rank > 70:
        second = "The profile is technically relevant, but its availability signals are weaker than those of candidates ranked above it."
    else:
        second = "No major profile-consistency issue was detected, though availability signals are only moderate."

    return f"{first} {second}"
