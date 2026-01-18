import os
import pickle
import numpy as np

from src.rag.graph.builder import KnowledgeGraphBuilder
from src.rag.graph.node_embed import GraphNodeEmbedder
from src.rag.graph.clustering import GraphClusterer
from src.rag.retrieval import HybridRetriever

os.makedirs("cache", exist_ok=True)

print("🚀 Building GraphRAG...")

# 1) Build graph + chunks
graph = KnowledgeGraphBuilder()
graph.build_from_json("data/corpus/novel.json", text_key="context")

print("Chunks:", len(graph.chunks))

# Save chunks + entities + KG
with open("cache/chunks.pkl", "wb") as f:
    pickle.dump(graph.chunks, f)

with open("cache/chunk_entities.pkl", "wb") as f:
    pickle.dump(graph.chunk_entities, f)

with open("cache/kg.pkl", "wb") as f:
    pickle.dump(graph.kg, f)

# (optional) quick KG stats if it's a networkx graph
try:
    print(f"KG nodes: {graph.kg.number_of_nodes()}, edges: {graph.kg.number_of_edges()}")
except Exception:
    pass

# 2) Retriever embeddings
retriever = HybridRetriever(graph.chunks)

# Ensure it's a numpy array then save
retriever_emb = np.asarray(retriever.embeddings)
np.save("cache/retriever_embeddings.npy", retriever_emb)

# 3) Node embeddings (node2vec)
node_embedder = GraphNodeEmbedder()
node_embeddings = node_embedder.fit(graph.kg)

# ✅ Save node embeddings robustly
# Expecting dict[str, np.ndarray] OR (nodes, matrix)
# We support both safely via pickle.

with open("cache/node_embeddings.pkl", "wb") as f:
    pickle.dump(node_embeddings, f)

# Optional: if node_embeddings is dict, also save a dense matrix version for debugging/fast ops
if isinstance(node_embeddings, dict) and len(node_embeddings) > 0:
    nodes = list(node_embeddings.keys())
    mat = np.stack([np.asarray(node_embeddings[n]) for n in nodes], axis=0)
    with open("cache/node_embedding_nodes.pkl", "wb") as f:
        pickle.dump(nodes, f)
    np.save("cache/node_embedding_matrix.npy", mat)
    print(f"Node embeddings saved: {len(nodes)} nodes, dim={mat.shape[1]}")
else:
    print("Node embeddings saved (pickle).")

# 4) Clustering
clusterer = GraphClusterer(method="greedy")
clusters = clusterer.cluster(graph.kg)

with open("cache/clusters.pkl", "wb") as f:
    pickle.dump(clusters, f)

# quick sanity logs
try:
    if isinstance(clusters, dict):
        print(f"Clusters saved: dict with {len(set(clusters.values()))} unique cluster ids")
    else:
        print(f"Clusters saved: {len(clusters)} clusters")
except Exception:
    print("Clusters saved.")

print("✅ GraphRAG build & saved")
