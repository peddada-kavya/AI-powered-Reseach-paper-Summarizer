import streamlit as st
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# =============================
# 1. AUTO PATH DETECTION
# =============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAG_DIR = os.path.join(BASE_DIR, "rag_pipeline")
PAPERS_DIR = os.path.join(BASE_DIR, "research_papers")

CHUNKS_FILE = os.path.join(RAG_DIR, "chunks.json")
EMBEDDINGS_FILE = os.path.join(RAG_DIR, "embeddings.npy")

os.makedirs(PAPERS_DIR, exist_ok=True)

# =============================
# 2. LOAD DATA & MODELS
# =============================
@st.cache_resource
def load_models():
    # Embedding model (Sentence-Transformer)
    embed_mod = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Generation model (Google FLAN-T5 Base)
    # Using 'text2text-generation' for T5
    gen_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    
    return embed_mod, gen_pipeline

@st.cache_data
def load_data():
    chunks_data = []
    embeddings_data = None
    if os.path.exists(CHUNKS_FILE):
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
    if os.path.exists(EMBEDDINGS_FILE):
        embeddings_data = np.load(EMBEDDINGS_FILE)
    return chunks_data, embeddings_data

model, generator = load_models()
chunks, embeddings = load_data()

# =============================
# 3. CORE FUNCTIONS
# =============================
def hybrid_search(question, selected_paper, top_k=3):
    if embeddings is None or len(chunks) == 0:
        return [], 0.0

    question_embedding = model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, embeddings)[0]

    results = []
    for i, score in enumerate(scores):
        chunk = chunks[i]
        if selected_paper == "All Papers" or chunk["paper_name"] == selected_paper:
            results.append({
                "text": chunk["text"],
                "paper_name": chunk["paper_name"],
                "score": float(score)
            })

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    top_score = results[0]["score"] if results else 0.0
    return results[:top_k], top_score

def generate_flan_answer(question, context):
    """Uses FLAN-T5 to generate an answer quickly."""
    # FLAN-T5 works best with a clear instruction prefix
    input_text = f"answer the question using the context: context: {context} question: {question}"
    
    # We limit max_new_tokens for even faster response
    output = generator(input_text, max_new_tokens=150, temperature=0.1, do_sample=False)
    return output[0]['generated_text']

# =============================
# 4. STREAMLIT UI
# =============================
st.set_page_config(page_title=" Research paper QA", page_icon="📄", layout="wide")
st.title("📄 Research paper QA (FLAN-T5)")

# Paper Selection
papers = ["All Papers"]
if os.path.exists(PAPERS_DIR):
    papers.extend([f for f in os.listdir(PAPERS_DIR) if f.endswith(".pdf")])
selected_paper = st.selectbox("Select Research Paper", papers)

# Question Input
question = st.text_input("Ask Question")

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question")
    else:
        results, top_score = hybrid_search(question, selected_paper)

        if not results:
            st.error("Data files not found.")
        elif top_score < 0.35:
            st.error(f"question not relevant to research paper ({top_score:.4f}).")
        else:
            # Prepare Context (limiting to top 2-3 to keep T5 efficient)
            context_text = " ".join([r["text"] for r in results[:2]])
            
            with st.spinner("⚡ Fast AI is generating an answer..."):
                answer = generate_flan_answer(question, context_text)
            
            st.subheader("Best Answer")
            st.info(answer)
            
            # Confidence Section
            c1, c2 = st.columns([1, 3])
            c1.metric("Confidence", f"{top_score:.4f}")
            c2.progress(min(max(top_score, 0.0), 1.0))

            with st.expander("View Source Context"):
                for r in results:
                    st.write(f"**{r['paper_name']}**")
                    st.write(r["text"])
                    st.divider()

# Sidebar Status
st.sidebar.title("System Status")
st.sidebar.write(f"Mode: Local (FLAN-T5 Base)")
st.sidebar.write(f"Chunks: {len(chunks)}")