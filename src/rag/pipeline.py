from src.llm.kimi_remote import KimiGenerator
from src.rag.graph.node_embed import GraphNodeEmbedder
# from src.rag.graph.clustering import GraphClusterer

import numpy as np



class GraphRAGPipeline:
    def __init__(self, retriever, graph, reasoner, generator=None):
        self.retriever = retriever
        self.graph = graph
        self.reasoner = reasoner
        self.generator = generator

        # 🔹 LLM generator
        self.generator = KimiGenerator()

        # 🔹 Graph representation
        self.node_embedder = GraphNodeEmbedder(
            dimensions=128,
            walk_length=40,
            num_walks=10
        )
        # self.clusterer = GraphClusterer(method="greedy")

        # sẽ được build sau
        self.node_embeddings = None
        self.clusters = None

    def build_graph_representation(self):
        """
        Run once after KG is built
        """
        # # 1️⃣ Node embedding
        # self.node_embeddings = self.node_embedder.fit(self.graph.kg)

        # # 2️⃣ Graph clustering
        # self.clusters = self.clusterer.cluster(self.graph.kg)
    
    def _find_relevant_clusters(self, query: str):
        """
        Return clusters that contain entities mentioned in query
        """
        doc = self.graph.nlp(query)
        query_entities = {ent.text for ent in doc.ents}

        matched_clusters = []
        for cluster in self.clusters:
            if any(node in query_entities for node in cluster):
                matched_clusters.append(cluster)

        return matched_clusters

    def _expand_context_from_clusters(self, clusters):
        """
        Convert node clusters to text context
        """
        contexts = []

        for cluster in clusters:
            for node in cluster:
                # node là entity → lấy các chunk chứa entity đó
                contexts.extend(
                    self.graph.get_chunks_containing_entity(node)
                )

        return list(set(contexts))  # remove duplicates
    
    def _find_similar_graph_nodes(self, query: str, top_k: int = 5):
        """
        Find graph nodes similar to query entities using node embeddings
        """
        if self.node_embeddings is None:
            return []

        # 1️⃣ Extract entities from query
        doc = self.graph.nlp(query)
        query_entities = [ent.text for ent in doc.ents]

        if not query_entities:
            return []

        # 2️⃣ Collect embeddings of query-matched nodes
        matched_vectors = []
        for ent in query_entities:
            if ent in self.node_embeddings:
                matched_vectors.append(self.node_embeddings[ent])

        if not matched_vectors:
            return []

        query_vec = np.mean(matched_vectors, axis=0)

        # 3️⃣ Cosine similarity in graph embedding space
        sims = []
        for node, vec in self.node_embeddings.items():
            score = np.dot(query_vec, vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-9
            )
            sims.append((node, score))

        sims.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in sims[:top_k]]

    def _expand_context_with_node_similarity(self, query: str):
        """
        Expand context using node embedding similarity
        """
        similar_nodes = self._find_similar_graph_nodes(query)

        chunks = []
        for node in similar_nodes:
            chunks.extend(
                self.graph.get_chunks_containing_entity(node)
            )

        return list(set(chunks))


    def answer(self, query: str) -> str:
        # 1️⃣ Semantic retrieval (baseline)
        semantic_chunks = self.retriever.retrieve(query, top_k=5)

        # 2️⃣ Graph-aware expansion via node embedding similarity
        graph_chunks = self._expand_context_with_node_similarity(query)

        # 3️⃣ Merge & de-duplicate
        all_chunks = []
        seen = set()
        for ch in semantic_chunks + graph_chunks:
            if ch not in seen:
                all_chunks.append(ch)
                seen.add(ch)

        # 4️⃣ Truncate context
        context = "\n".join(all_chunks[:8])

        # 5️⃣ SAT reasoning (local)
        # reasoning = self.reasoner(query, context)

        # 2️⃣ Nếu có reasoner (Qwen local) → dùng
        if self.reasoner is not None:
            reasoning = self.reasoner(query, context)

            prompt = f"""
    CONTEXT:
    {context}

    QUESTION:
    {query}

    REASONING:
    {reasoning}

    ANSWER:
    """
            return self.generator.generate(prompt)

        # 3️⃣ Nếu KHÔNG có reasoner → generator trả lời trực tiếp
        else:
            prompt = f"""
    CONTEXT:
    {context}

    QUESTION:
    {query}

    ANSWER:
    """
            return self.generator.generate(prompt)