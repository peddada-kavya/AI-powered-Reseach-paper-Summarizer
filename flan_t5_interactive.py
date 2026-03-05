import sys
import os

# Fix imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from rag_pipeline.hybrid_retrieval import hybrid_search
from rag_pipeline.summarizer import Summarizer


def main():

    print("\n====================================")
    print("   Hybrid RAG + FLAN-T5 System")
    print("====================================")

    print("\nInitializing LLM...")
    summarizer = Summarizer()

    while True:

        query = input("\nEnter your question (or type 'exit'): ")

        if query.lower() == "exit":
            print("Exiting...")
            break

        print("\nRetrieving context...")
        context = hybrid_search(query)

        print("\nGenerating answer...")
        answer = summarizer.summarize(context, query)

        print("\n========== FINAL ANSWER ==========\n")
        print(answer)
        print("\n==================================\n")


if __name__ == "__main__":
    main()