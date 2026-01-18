import itertools
import json
import networkx as nx
import spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter


class KnowledgeGraphBuilder:
    def __init__(self, chunk_size=800, chunk_overlap=100, min_edge_weight=2, directed=False):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_edge_weight = min_edge_weight
        self.directed = directed

        self.nlp = spacy.load("en_core_web_sm")
        self.kg = None
        self.chunks = []
        self.chunk_entities = []

    def build_from_json(self, json_path: str, text_key: str = "context"):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts = []
        for item in data:
            if isinstance(item, dict) and text_key in item:
                texts.append(item[text_key])
            elif isinstance(item, str):
                texts.append(item)

        self.chunks = self._chunk_texts(texts)
        self.kg, self.chunk_entities = self.build(self.chunks)

    def _chunk_texts(self, texts):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        docs = splitter.create_documents(texts)
        return [d.page_content for d in docs]

    def build(self, chunks):
        # ✅ Co-occurrence KG: entities that appear in same chunk are connected
        kg = nx.DiGraph() if self.directed else nx.Graph()
        chunk_entities = []

        for chunk in chunks:
            doc = self.nlp(chunk)

            # entity set in this chunk
            ents = [e.text.strip() for e in doc.ents if e.text.strip()]
            ents = list(set(ents))  # unique
            chunk_entities.append(set(ents))

            # add nodes
            for e in ents:
                if not kg.has_node(e):
                    kg.add_node(e)

            # add edges by co-occurrence within the same chunk
            if len(ents) >= 2:
                for u, v in itertools.combinations(ents, 2):
                    if kg.has_edge(u, v):
                        kg[u][v]["weight"] = kg[u][v].get("weight", 1) + 1
                    else:
                        kg.add_edge(u, v, weight=1)

        # prune weak edges (optional, helps reduce noise)
        if self.min_edge_weight and self.min_edge_weight > 1:
            to_remove = [(u, v) for u, v, d in kg.edges(data=True) if d.get("weight", 1) < self.min_edge_weight]
            kg.remove_edges_from(to_remove)

        return kg, chunk_entities

    def get_chunks_containing_entity(self, entity: str):
        results = []
        for chunk, ents in zip(self.chunks, self.chunk_entities):
            if entity in ents:
                results.append(chunk)
        return results
