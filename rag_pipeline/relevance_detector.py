from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

THRESHOLD = 0.35


def is_relevant(question, context):
    """
    Check if question is relevant to retrieved context
    """

    if not context or len(context.strip()) == 0:
        return False, 0.0

    q_emb = model.encode(question, convert_to_tensor=True)

    c_emb = model.encode(context, convert_to_tensor=True)

    score = util.cos_sim(q_emb, c_emb).item()

    return score >= THRESHOLD, float(score)