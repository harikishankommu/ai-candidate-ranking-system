from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def similarity(jd_text, candidate_text):
    jd_emb = model.encode([jd_text])
    cand_emb = model.encode([candidate_text])

    return cosine_similarity(
        jd_emb,
        cand_emb
    )[0][0]