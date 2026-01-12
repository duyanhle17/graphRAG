import pickle
import numpy as np

from src.llm.qwen_local import load_qwen_base, qwen_reasoning
from src.rag.pipeline import GraphRAGPipeline
from src.rag.retrieval import HybridRetriever



print("🔄 Loading cache...")

with open("cache/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

with open("cache/chunk_entities.pkl", "rb") as f:
    chunk_entities = pickle.load(f)

with open("cache/kg.pkl", "rb") as f:
    kg = pickle.load(f)

with open("cache/clusters.pkl", "rb") as f:
    clusters = pickle.load(f)

retriever = HybridRetriever(chunks)
retriever.embeddings = np.load("cache/retriever_embeddings.npy")

print("✅ Cache loaded")

# load Qwen base (inference only)
tokenizer, model = load_qwen_base()

def reasoner(q, c):
    return qwen_reasoning(q, c, tokenizer, model)

# KIMI remote LLM generator
from src.llm.kimi_remote import KimiGenerator
generator = KimiGenerator()

pipeline = GraphRAGPipeline(
    retriever=retriever,
    graph=kg,
    reasoner=None,
    generator=generator
)


print("🤖 Chatbot ready (type exit)")

while True:
    q = input("\nYou: ")
    if q == "exit":
        break
    print("Bot:", pipeline.answer(q))
