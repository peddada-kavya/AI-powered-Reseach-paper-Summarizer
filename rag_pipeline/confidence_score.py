from sentence_transformers import SentenceTransformer, util

class ConfidenceScorer:

    def __init__(self):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def calculate_confidence(self, question, context, answer):

        emb = self.model.encode(
            [question, context, answer],
            convert_to_tensor=True
        )

        score1 = util.cos_sim(emb[0], emb[2]).item()
        score2 = util.cos_sim(emb[1], emb[2]).item()

        confidence = (score1 + score2) / 2

        return float(confidence) 