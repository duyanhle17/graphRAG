import networkx as nx
from networkx.algorithms import community


class GraphClusterer:
    def __init__(self, method="greedy"):
        self.method = method

    def cluster(self, graph: nx.Graph):
        if self.method == "greedy":
            communities = community.greedy_modularity_communities(
                graph.to_undirected()
            )
            return [list(c) for c in communities]

        raise ValueError(f"Unsupported clustering method: {self.method}")
