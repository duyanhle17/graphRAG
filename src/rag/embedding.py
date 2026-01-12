class EmbeddingIndex:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.index = None

    def build(self, texts):
        emb = self.model.encode(texts, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb.astype("float32"))
        return emb

    def search(self, query, k):
        q = self.model.encode([query], normalize_embeddings=True)
        return self.index.search(q.astype("float32"), k)
