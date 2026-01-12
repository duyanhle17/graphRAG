import pickle
import numpy as np
import os

from src.rag.graph.builder import KnowledgeGraphBuilder
from src.rag.graph.node_embed import GraphNodeEmbedder
from src.rag.graph.clustering import GraphClusterer
from src.rag.retrieval import HybridRetriever

os.makedirs("cache", exist_ok=True)

print("🚀 Building GraphRAG...")

# 1️⃣ Build graph + chunks
graph = KnowledgeGraphBuilder()
graph.build_from_json("data/corpus/novel.json", text_key="context")

print("Chunks:", len(graph.chunks))

# save chunks
with open("cache/chunks.pkl", "wb") as f:
    pickle.dump(graph.chunks, f)

with open("cache/chunk_entities.pkl", "wb") as f:
    pickle.dump(graph.chunk_entities, f)

with open("cache/kg.pkl", "wb") as f:
    pickle.dump(graph.kg, f)

# 2️⃣ Retriever embeddings
retriever = HybridRetriever(graph.chunks)

np.save("cache/retriever_embeddings.npy", retriever.embeddings)

# 3️⃣ Node embeddings
node_embedder = GraphNodeEmbedder()
node_embeddings = node_embedder.fit(graph.kg)

np.save("cache/node_embeddings.npy", node_embeddings)

# 4️⃣ Clustering
clusterer = GraphClusterer(method="greedy")
clusters = clusterer.cluster(graph.kg)

with open("cache/clusters.pkl", "wb") as f:
    pickle.dump(clusters, f)

print("✅ GraphRAG build & saved")
