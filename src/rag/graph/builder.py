import json
import networkx as nx
import spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter


class KnowledgeGraphBuilder:
    def __init__(self, chunk_size=800, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.nlp = spacy.load("en_core_web_sm")
        self.kg = None
        self.chunks = []
        self.chunk_entities = []

    # ---------- NEW: build from JSON ----------
    def build_from_json(self, json_path: str, text_key: str = "context"):
        """
        json format: list[{"text": "..."}] or list[str]
        """
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

    # ---------- chunk texts ----------
    def _chunk_texts(self, texts):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        docs = splitter.create_documents(texts)
        return [d.page_content for d in docs]

    # ---------- build KG from chunks ----------
    def build(self, chunks):
        kg = nx.DiGraph()
        chunk_entities = []

        for chunk in chunks:
            doc = self.nlp(chunk)
            ents = {e.text.strip() for e in doc.ents if e.text.strip()}
            chunk_entities.append(ents)

            for e in ents:
                if not kg.has_node(e):
                    kg.add_node(e)

        return kg, chunk_entities

    # ---------- helper for GraphRAG ----------
    def get_chunks_containing_entity(self, entity: str):
        """
        Return chunks that contain a given entity
        """
        results = []
        for chunk, ents in zip(self.chunks, self.chunk_entities):
            if entity in ents:
                results.append(chunk)
        return results
