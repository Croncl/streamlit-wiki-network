import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os

st.set_page_config(layout="wide")
st.title("🔍 Análise e Visualização de Grafo da Wikipedia")
st.markdown("Envie o CSV (**source**, **target**)")

uploaded_file = st.file_uploader("CSV com colunas source,target", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.dropna(subset=["source", "target"], inplace=True)
    G = nx.from_pandas_edgelist(
        df, source="source", target="target", create_using=nx.DiGraph()
    )

    st.write(f"Grafo: {G.number_of_nodes()} nós → {G.number_of_edges()} arestas")

    # Remover self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Seletor de subgrafo
    st.sidebar.subheader("🔍 Filtros de Subgrafo")
    min_degree = st.sidebar.slider("Grau mínimo do nó", 0, 10, 2)
    show_largest_scc = st.sidebar.checkbox("Mostrar maior SCC (dirigido)", value=False)
    show_largest_wcc = st.sidebar.checkbox("Mostrar maior WCC", value=False)

    if show_largest_scc and nx.is_directed(G):
        largest_scc = max(nx.strongly_connected_components(G), key=len)
        SG = G.subgraph(largest_scc).copy()
        st.write(
            f"Subgrafo: Maior SCC → {SG.number_of_nodes()} nós, {SG.number_of_edges()} arestas"
        )

    elif show_largest_wcc:
        largest_wcc = max(nx.weakly_connected_components(G), key=len)
        SG = G.subgraph(largest_wcc).copy()
        st.write(
            f"Subgrafo: Maior WCC → {SG.number_of_nodes()} nós, {SG.number_of_edges()} arestas"
        )

    else:
        SG_nodes = [n for n in G.nodes() if G.degree(n) > min_degree]
        SG = G.subgraph(SG_nodes).copy()
        st.write(
            f"Subgrafo (grau > {min_degree}): {SG.number_of_nodes()} nós → {SG.number_of_edges()} arestas"
        )

    # Visualização interativa com Pyvis
    st.subheader("🌐 Visualização Interativa (Pyvis)")
    net = Network(
        notebook=False,
        height="750px",
        width="100%",
        directed=True,
        bgcolor="#222222",
        font_color="white",
        cdn_resources="remote",
    )

    net.force_atlas_2based(
        gravity=-30,
        central_gravity=0.05,
        spring_length=150,
        spring_strength=0.03,
        damping=0.85,
        overlap=0.5,
    )
    net.toggle_physics(True)
    net.options.configure = {"enabled": False}

    degrees = dict(SG.degree())
    for node in SG.nodes():
        net.add_node(
            node,
            label=str(node),
            size=10 + degrees[node] * 0.7,
            color=f"hsl({240 - degrees[node]*7}, 75%, 50%)",
            title=f"Nó: {node} (Grau: {degrees[node]})",
        )

    for u, v, data in SG.edges(data=True):
        weight = data.get("weight", 1.0)
        net.add_edge(
            u,
            v,
            width=0.5 + float(weight) * 0.1,
            title=f"Peso: {weight:.2f}",
            color="#97C2FC",
        )

    tmp_dir = tempfile.mkdtemp()
    path = os.path.join(tmp_dir, "graph.html")
    net.save_graph(path)
    st.components.v1.html(open(path, "r", encoding="utf-8").read(), height=750)

    # Visualização Pyvis dos top-k
    st.subheader("🔝 Visualização dos Top-k Nós")
    dc = nx.degree_centrality(G)
    top_nodes = sorted(dc.items(), key=lambda x: x[1], reverse=True)[:10]
    H = G.subgraph([n for n, _ in top_nodes])
    net2 = Network(height="600px", width="100%", directed=True, notebook=False)
    net2.from_nx(H)
    tmp_dir2 = tempfile.mkdtemp()
    path2 = os.path.join(tmp_dir2, "topk.html")
    net2.save_graph(path2)
    st.components.v1.html(open(path2, "r", encoding="utf-8").read(), height=600)

    # Métricas estruturais
    st.header("📊 Métricas Estruturais")
    density = nx.density(G)
    assortativity = nx.degree_assortativity_coefficient(G)
    clustering = nx.average_clustering(G.to_undirected())
    scc = nx.number_strongly_connected_components(G) if nx.is_directed(G) else "-"
    wcc = nx.number_weakly_connected_components(G)

    st.markdown(
        f"""
    - **Densidade**: {density:.4f}
    - **Assortatividade**: {assortativity:.4f}
    - **Clustering Global**: {clustering:.4f}
    - **Componentes Fortemente Conectados**: {scc}
    - **Componentes Fracamente Conectados**: {wcc}
    """
    )

    # Distribuições de grau
    st.header("📈 Distribuição de Grau")
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(list(in_degrees.values()), bins=30, ax=ax[0], color="skyblue")
    ax[0].set_title("Distribuição do Grau de Entrada")
    sns.histplot(list(out_degrees.values()), bins=30, ax=ax[1], color="salmon")
    ax[1].set_title("Distribuição do Grau de Saída")
    st.pyplot(fig)

    # Centralidades
    st.header("🏆 Centralidades (Top 10)")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Eigenvector Centrality")
        ec = nx.eigenvector_centrality(G, max_iter=1000)
        st.table(pd.Series(ec).sort_values(ascending=False).head(10))

        st.subheader("Closeness Centrality")
        cc = nx.closeness_centrality(G)
        st.table(pd.Series(cc).sort_values(ascending=False).head(10))

    with col2:
        st.subheader("Degree Centrality")
        st.table(pd.Series(dc).sort_values(ascending=False).head(10))

        st.subheader("Betweenness Centrality")
        bc = nx.betweenness_centrality(G)
        st.table(pd.Series(bc).sort_values(ascending=False).head(10))
