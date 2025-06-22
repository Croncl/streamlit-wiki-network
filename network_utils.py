def build_graph(seed, max_layers=2, max_nodes=2000):
    # lógica de coleta com wikipedia + networkx
    return g

def compute_metrics(G):
    return {
        "density": nx.density(G),
        "clustering": nx.transitivity(G),
        "assortativity": nx.degree_assortativity_coefficient(G),
        "scc": list(nx.strongly_connected_components(G)),
        "wcc": list(nx.weakly_connected_components(G)),
        "centrality": {
            "degree": nx.degree_centrality(G),
            ...
        }
    }
