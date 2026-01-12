from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable

@dataclass
class GraphRAGConfig:
    working_dir: str = field(
        default_factory=lambda: f"./nano_graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )

    enable_local: bool = True
    enable_naive_rag: bool = False

    tokenizer_type: str = "tiktoken"
    tiktoken_model_name: str = "cl100k_base"
    huggingface_model_name: str = "bert-base-uncased"

    chunk_token_size: int = 1000
    chunk_overlap_token_size: int = 100
    chunk_char_size: int = 1200
    chunk_char_overlap: int = 100

    chunk_func: Optional[Callable] = None

    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    graph_cluster_algorithm: str = "greedy"
    max_graph_cluster_size: int = 8
    graph_cluster_seed: int = 0xDEADBEEF

    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 128,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 3,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16
    query_better_than_threshold: float = 0.2
