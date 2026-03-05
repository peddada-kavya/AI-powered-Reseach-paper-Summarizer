import sys
import os
from difflib import SequenceMatcher

# Fix import path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from rag_pipeline.summarizer import Summarizer
from evaluation.test_dataset import test_data
from evaluation.confidence_score import ConfidenceScorer


class LLMEvaluator:

    def __init__(self):

        print("Initializing LLM Evaluator...")

        # Load LLM
        self.llm = Summarizer()

        # Load confidence scorer
        self.confidence_scorer = ConfidenceScorer()


    # Accuracy similarity (ground truth vs prediction)
    def similarity_score(self, ground_truth, prediction):

        return SequenceMatcher(
            None,
            ground_truth.lower(),
            prediction.lower()
        ).ratio()


    def evaluate(self):

        total_score = 0
        total_confidence = 0

        print("\nRunning LLM Evaluation...\n")

        for item in test_data:

            question = item["question"]
            ground_truth = item["ground_truth"]

            # Generate prediction
            prediction = self.llm.summarize("", question)

            # Accuracy score
            score = self.similarity_score(
                ground_truth,
                prediction
            )

            # Confidence score
            confidence = self.confidence_scorer.compute_confidence(
                question,
                ground_truth,   # using ground truth as context for LLM eval
                prediction
            )

            total_score += score
            total_confidence += confidence

            print("Question:", question)
            print("Ground Truth:", ground_truth)
            print("Prediction:", prediction)

            print("Accuracy Score:", round(score, 3))
            print("Confidence Score:", confidence)

            print("-" * 50)


        avg_score = total_score / len(test_data)
        avg_confidence = total_confidence / len(test_data)

        print("\n========== FINAL RESULTS ==========")
        print("LLM Accuracy:", round(avg_score * 100, 2), "%")
        print("Average Confidence:", round(avg_confidence, 3))
        print("===================================")


if __name__ == "__main__":

    evaluator = LLMEvaluator()
    evaluator.evaluate()
