# Methodology and Defense Notes

## Problem interpretation

The released JD is not asking for generic AI keyword overlap. It needs a senior, hands-on engineer who has shipped ranking, retrieval, search, or recommendation systems, understands rigorous evaluation, can work in an early-stage product environment, and is realistically reachable for hiring.

## Architecture

```text
100K JSONL candidates
        |
        v
Broad CPU prefilter (top 5,000)
        |
        v
Detailed evidence scoring
  - career proof of ranking/retrieval/recommendation
  - evaluation and production operations
  - role, seniority, experience, product background
        |
        +--> behavioral availability score
        +--> location / notice / relocation score
        +--> consistency and honeypot checks
        |
        v
Deterministic top 100 + grounded reasoning CSV
```

## Why two stages

Detailed scoring every long profile is unnecessary. A broad first stage retains any candidate with relevant titles or career phrases, while rejecting obvious unrelated profiles. The second stage evaluates 5,000 candidates in depth. On the provided 100K file, the complete run takes roughly 15–20 seconds in the current container, far below the five-minute limit.

## Main score components

- **Career evidence:** highest weight. Looks for actual ranking, search, retrieval, recommendation, evaluation, deployment, index operations, and large-scale serving work.
- **Title and seniority:** relevant engineering titles are trusted; unrelated business titles are strongly penalized.
- **Experience:** 5–9 years is optimal, but nearby candidates remain eligible.
- **Product and hands-on engineering:** product-company and current-role evidence is rewarded; services-only histories are down-weighted.
- **Behavioral signals:** recency, open-to-work status, recruiter response, interview completion, saved profiles, and completeness.
- **Logistics:** Pune/Noida, Indian tier-1 locations, relocation willingness, and notice period.
- **Stability:** rewards tenure sufficient to ship systems and penalizes repeated very short switches.

## Keyword-stuffer defense

Skill names alone receive almost no score. Terms such as FAISS, Pinecone, RAG, and QLoRA matter only when work descriptions show how they were used in a production system. Profiles describing AI curiosity or recent ChatGPT experimentation are penalized without deeper ML history.

## Honeypot defense

The model checks:

- career duration versus start/end dates;
- claimed total experience versus career span;
- summary claims versus profile experience;
- current company/title versus current career entry;
- expert proficiency with almost no usage duration.

Candidates with severe contradictions receive a large penalty. The generated top 100 currently contains no detected anomaly.

## Reasoning generation

Reasoning is deterministic and uses only facts present in each profile. It cites evidence types found in career history and adds an honest availability concern when relevant. Several sentence patterns are used to reduce templated repetition.

## Limitations

There is no released relevance label, so weights cannot be tuned against a validation target. The method instead follows the JD and challenge instructions explicitly. A future production version would learn weights using recruiter judgments and optimize NDCG/MRR against an offline benchmark before online A/B testing.

## Interview explanation

A concise defense:

> I avoided semantic similarity as the only ranker because the dataset contains deliberate keyword stuffing. I used career evidence as the primary signal, behavior as a hiring-probability modifier, and consistency checks to remove impossible profiles. The two-stage design keeps runtime low, and every recommendation can be traced to score components and profile facts.
