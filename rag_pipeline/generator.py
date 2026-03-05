import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# =============================
# MODEL CONFIG
# =============================

MODEL_NAME = "google/flan-t5-base"

print("Loading FLAN-T5 Base model...")

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Use CPU (safe default)
device = torch.device("cpu")
model = model.to(device)

print("Generator model loaded successfully")


# =============================
# ANSWER GENERATOR FUNCTION
# =============================

def generate_answer(question, context):
    """
    Generate answer using FLAN-T5 Base model
    """

    if not context or len(context.strip()) == 0:
        return "No relevant context found."

    # Strong prompt for research QA
    prompt = f"""
You are an expert research assistant.

Answer the question using ONLY the given context.

Rules:
- Give clear answer
- Use 2 to 4 sentences
- Be precise
- Do NOT repeat question
- Do NOT give one word answer

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            do_sample=False,
            repetition_penalty=1.2
        )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer.strip()