import pickle
import numpy as np

from src.rag.pipeline import GraphRAGPipeline
from src.rag.retrieval import HybridRetriever
from src.llm.kimi_remote import KimiGenerator

print("🔄 Loading cache...")

with open("cache/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

with open("cache/chunk_entities.pkl", "rb") as f:
    chunk_entities = pickle.load(f)

with open("cache/kg.pkl", "rb") as f:
    kg = pickle.load(f)

# clusters (list[list[str]] OR dict[str,int])
with open("cache/clusters.pkl", "rb") as f:
    clusters = pickle.load(f)

# node_embeddings: dict[str, np.ndarray]
with open("cache/node_embeddings.pkl", "rb") as f:
    node_embeddings = pickle.load(f)

# retriever embeddings
retriever = HybridRetriever(chunks)
retriever.embeddings = np.load("cache/retriever_embeddings.npy")

print("✅ Cache loaded")

# ---- normalize clusters if needed ----
# If clusters is {node: cluster_id}, convert to list of clusters
if isinstance(clusters, dict):
    tmp = {}
    for node, cid in clusters.items():
        tmp.setdefault(cid, []).append(node)
    clusters = list(tmp.values())

# quick sanity
try:
    n_clusters = len(clusters)
except Exception:
    n_clusters = 0

print(f"📌 clusters loaded: {n_clusters}")
print(f"📌 node_embeddings loaded: {len(node_embeddings) if isinstance(node_embeddings, dict) else 'unknown'}")

generator = KimiGenerator()

pipeline = GraphRAGPipeline(
    retriever=retriever,
    graph=kg,
    reasoner=None,
    generator=generator,
    node_embeddings=node_embeddings,
    clusters=clusters,
    chunks=chunks,
    chunk_entities=chunk_entities
)


print("🤖 Chatbot ready (type exit)")

while True:
    q = input("\nYou: ").strip()
    if q.lower() == "exit":
        break
    print("Bot:", pipeline.answer(q))
