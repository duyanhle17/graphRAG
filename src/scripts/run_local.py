# ========= FIX IMPORT PATH =========
import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
print("PROJECT_ROOT =", PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ========= TEST IMPORT =========
from src.rag.graph.node_embed import GraphNodeEmbedder
print("✅ Import GraphNodeEmbedder OK")

# ========= IMPORT MODULE =========
from src.llm.qwen_local import load_qwen_base, qwen_reasoning
from src.llm.kimi_remote import KimiGenerator
from src.rag.pipeline import GraphRAGPipeline
from src.rag.graph.builder import KnowledgeGraphBuilder
from src.rag.retrieval import HybridRetriever

# ========= 1️⃣ LOAD QWEN BASE (LOCAL REASONER) =========
tokenizer, model = load_qwen_base()

def reasoner_fn(query: str, context: str) -> str:
    return qwen_reasoning(query, context, tokenizer, model)

# ========= 2️⃣ BUILD GRAPH =========
graph = KnowledgeGraphBuilder()
graph.build_from_json("data/corpus/novel.json", text_key="context")

# ========= 3️⃣ RETRIEVER =========
retriever = HybridRetriever(graph.chunks)

# ========= 4️⃣ GENERATOR (KIMI REMOTE) =========
# ⚠️ NVAPI_KEY MUST be set in terminal
generator = KimiGenerator()

# ========= 5️⃣ PIPELINE (CHỈ TẠO 1 LẦN) =========
pipeline = GraphRAGPipeline(
    retriever=retriever,
    graph=graph,
    reasoner=reasoner_fn,   # Qwen reasoning
    generator=generator    # Kimi final answer
)

# ========= 6️⃣ BUILD GRAPH REPRESENTATION =========
pipeline.build_graph_representation()

print("NUM CHUNKS =", len(graph.chunks))


# ========= 7️⃣ ASK QUESTION =========
if __name__ == "__main__":
    question = "What is basal cell carcinoma?"
    answer = pipeline.answer(question)

    print("\nQUESTION:", question)
    print("ANSWER:", answer)
