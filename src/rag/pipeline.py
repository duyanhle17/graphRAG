from src.llm.kimi_remote import KimiGenerator
import numpy as np
import spacy 


class GraphRAGPipeline:
    def __init__(self, retriever, graph, reasoner=None, generator=None,
                 node_embeddings=None, clusters=None, chunks=None, chunk_entities=None):
        import spacy
        self.retriever = retriever
        self.graph = graph                  # networkx graph OK
        self.reasoner = reasoner
        self.generator = generator or KimiGenerator()

        self.node_embeddings = node_embeddings
        self.clusters = clusters

        self.chunks = chunks
        self.chunk_entities = chunk_entities

        # ✅ pipeline tự có NLP, không cần self.graph.nlp nữa
        self.nlp = spacy.load("en_core_web_sm")
        
    def _get_chunks_containing_entity(self, entity: str):
        if self.chunks is None or self.chunk_entities is None:
            return []
        results = []
        for chunk, ents in zip(self.chunks, self.chunk_entities):
            # ents có thể là set/list
            if entity in ents:
                results.append(chunk)
        return results


    def _find_relevant_clusters(self, query: str):
        if not self.clusters:
            return []
        doc = self.nlp(query)   # ✅ dùng self.nlp
        query_entities = {ent.text for ent in doc.ents}
        if not query_entities:
            return []

        matched_clusters = []
        for cluster in self.clusters:
            if any(node in query_entities for node in cluster):
                matched_clusters.append(cluster)
        return matched_clusters

    def _expand_context_from_clusters(self, clusters):
        contexts = []
        for cluster in clusters:
            for node in cluster:
                contexts.extend(self._get_chunks_containing_entity(node))  # ✅
        # dedupe giữ thứ tự
        seen = set()
        out = []
        for c in contexts:
            if c not in seen:
                out.append(c)
                seen.add(c)
        return out

    def _find_similar_graph_nodes(self, query: str, top_k: int = 5):
        """
        Find graph nodes similar to query entities using node embeddings
        """
        if not self.node_embeddings:
            return []

        doc = self.nlp(query)
        query_entities = [ent.text for ent in doc.ents]
        if not query_entities:
            return []

        matched_vectors = []
        for ent in query_entities:
            if ent in self.node_embeddings:
                matched_vectors.append(self.node_embeddings[ent])

        if not matched_vectors:
            return []

        query_vec = np.mean(matched_vectors, axis=0)

        sims = []
        qn = np.linalg.norm(query_vec) + 1e-9
        for node, vec in self.node_embeddings.items():
            score = float(np.dot(query_vec, vec) / (qn * (np.linalg.norm(vec) + 1e-9)))
            sims.append((node, score))

        sims.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in sims[:top_k]]

    def _expand_context_with_node_similarity(self, query: str):
        similar_nodes = self._find_similar_graph_nodes(query)
        chunks = []
        for node in similar_nodes:
            chunks.extend(self._get_chunks_containing_entity(node))  # ✅
        # dedupe
        seen = set()
        out = []
        for c in chunks:
            if c not in seen:
                out.append(c)
                seen.add(c)
        return out


    def answer(self, query: str) -> str:
        # 1) semantic retrieval
        semantic_chunks = self.retriever.retrieve(query, top_k=5)

        # 2) cluster-aware expansion
        cluster_chunks = []
        if self.clusters:
            rel_clusters = self._find_relevant_clusters(query)
            cluster_chunks = self._expand_context_from_clusters(rel_clusters)

        # 3) node2vec similarity expansion
        graph_chunks = self._expand_context_with_node_similarity(query)

        # 4) merge (semantic -> cluster -> node2vec)
        all_chunks = []
        seen = set()
        for ch in semantic_chunks + cluster_chunks + graph_chunks:
            if ch not in seen:
                all_chunks.append(ch)
                seen.add(ch)

        # 5) truncate
        context = "\n".join(all_chunks[:8])

        # 6) if reasoner exists -> reasoning + generator
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

        # 7) else generator direct
        prompt = f"""
CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
        return self.generator.generate(prompt)
