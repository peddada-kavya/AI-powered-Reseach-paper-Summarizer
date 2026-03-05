from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class Summarizer:

    def __init__(self):

        model_name = "google/flan-t5-small"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def summarize(self, context, query):

        prompt = f"""
Answer using only the context.

Context:
{context}

Question:
{query}
"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        outputs = self.model.generate(inputs["input_ids"], max_new_tokens=200)

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return answer