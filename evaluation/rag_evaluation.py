import sys
import os
from difflib import SequenceMatcher

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from rag_pipeline.hybrid_retrieval import hybrid_search
from rag_pipeline.summarizer import Summarizer
from evaluation.confidence_score import ConfidenceScorer
from evaluation.test_dataset import test_data


class RAGEvaluator:

    def __init__(self):

        print("Initializing RAG Evaluator...")

        self.llm = Summarizer()

        # Initialize confidence scorer
        self.confidence_scorer = ConfidenceScorer()


    def similarity_score(self, ground_truth, prediction):

        return SequenceMatcher(
            None,
            ground_truth.lower(),
            prediction.lower()
        ).ratio()


    def evaluate(self):

        total_accuracy = 0
        total_confidence = 0

        print("\nRunning RAG Evaluation...\n")

        for item in test_data:

            question = item["question"]
            ground_truth = item["ground_truth"]

            # Step 1: Retrieve context
            context = hybrid_search(question)

            # Step 2: Generate answer
            prediction = self.llm.summarize(
                context,
                question
            )

            # Step 3: Accuracy score
            accuracy = self.similarity_score(
                ground_truth,
                prediction
            )

            total_accuracy += accuracy

            # Step 4: Confidence score
            confidence = self.confidence_scorer.compute_confidence(
                question,
                context,
                prediction
            )

            total_confidence += confidence

            # Print results
            print("Question:", question)
            print("Ground Truth:", ground_truth)
            print("Prediction:", prediction)

            print("Accuracy Score:", round(accuracy, 3))
            print("Confidence Score:", confidence)

            print("-" * 50)

        # Final averages
        avg_accuracy = total_accuracy / len(test_data)
        avg_confidence = total_confidence / len(test_data)

        print("\n========== FINAL RAG RESULTS ==========")

        print("Final RAG Accuracy:",
              round(avg_accuracy * 100, 2), "%")

        print("Average Confidence Score:",
              round(avg_confidence, 3))

        print("=====================================")


if __name__ == "__main__":

    evaluator = RAGEvaluator()
    evaluator.evaluate()
