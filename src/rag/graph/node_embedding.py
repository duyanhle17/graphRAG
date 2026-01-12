import networkx as nx

try:
    from node2vec import Node2Vec
except ImportError:
    Node2Vec = None


class GraphNodeEmbedder:
    def __init__(self, dimensions=128, walk_length=40, num_walks=10):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.embeddings = None

    def fit(self, graph: nx.Graph):
        if Node2Vec is None:
            raise RuntimeError("node2vec not installed. Run: pip install node2vec")

        n2v = Node2Vec(
            graph,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            workers=2,
        )
        model = n2v.fit(window=5, min_count=1)
        self.embeddings = {
            str(node): model.wv[str(node)]
            for node in graph.nodes()
        }
        return self.embeddings
