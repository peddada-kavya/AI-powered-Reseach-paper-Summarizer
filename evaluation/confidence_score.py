from sentence_transformers import SentenceTransformer, util


class ConfidenceScorer:

    def __init__(self):

        print("Initializing Confidence Scorer...")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Confidence Scorer ready.\n")


    def compute_confidence(self, query, context, answer):

        # Combine query + context
        combined = query + " " + context

        emb1 = self.model.encode(combined, convert_to_tensor=True)
        emb2 = self.model.encode(answer, convert_to_tensor=True)

        score = util.cos_sim(emb1, emb2).item()

        return round(score, 3)
