from sentence_transformers import SentenceTransformer
import numpy as np


class HybridRetriever:
    def __init__(self, chunks):
        """
        chunks: list[str]
        """
        self.chunks = chunks
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # build embeddings once
        self.embeddings = self.model.encode(
            chunks, normalize_embeddings=True
        )

    def retrieve(self, query: str, top_k: int = 5):
        q_emb = self.model.encode(
            [query], normalize_embeddings=True
        )[0]

        scores = np.dot(self.embeddings, q_emb)
        top_idx = np.argsort(scores)[::-1][:top_k]

        return [self.chunks[i] for i in top_idx]
