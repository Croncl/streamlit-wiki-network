import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pyvis.network import Network
import tempfile
import os

st.set_page_config(layout="wide")
st.title("🔍 Análise e Visualização de Grafo da Wikipedia")
st.markdown("Envie o CSV (**source**, **target**) para análise.")

uploaded_file = st.file_uploader("CSV (source, target)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Prévia dos dados:", df.head())

    # Criação do grafo dirigido
    G = nx.from_pandas_edgelist(
        df, source="source", target="target", create_using=nx.DiGraph()
    )

    st.subheader(f"Grafo: {G.number_of_nodes()} nós → {G.number_of_edges()} arestas")

    # Remove self-loops para evitar erro no core_number
    G.remove_edges_from(nx.selfloop_edges(G))

    # Subgrafo com nós de grau maior que 2
    SG_nodes = [n for n in G.nodes() if G.degree(n) > 2]
    SG = G.subgraph(SG_nodes).copy()
    st.write(
        f"Subgrafo (grau > 2): {SG.number_of_nodes()} nós → {SG.number_of_edges()} arestas"
    )

    # Visualização interativa com Pyvis
    st.subheader("🌐 Visualização Interativa (Pyvis)")
    net = Network(height="600px", width="100%", directed=True, notebook=False)
    net.from_nx(SG)
    tmp_dir = tempfile.mkdtemp()
    path = os.path.join(tmp_dir, "graph.html")
    net.save_graph(path)
    st.components.v1.html(open(path, "r", encoding="utf-8").read(), height=600)

    # Métricas estruturais
    st.subheader("📊 Métricas Estruturais")
    density = nx.density(G)
    assortativity = nx.degree_assortativity_coefficient(G)
    clustering = nx.average_clustering(G.to_undirected())
    weakly_cc = nx.number_weakly_connected_components(G)
    strongly_cc = nx.number_strongly_connected_components(G)

    st.markdown(f"- **Densidade**: {density:.4f}")
    st.markdown(f"- **Assortatividade**: {assortativity:.4f}")
    st.markdown(f"- **Coef. de Clustering (global)**: {clustering:.4f}")
    st.markdown(f"- **Componentes Fracamente Conectados**: {weakly_cc}")
    st.markdown(f"- **Componentes Fortemente Conectados**: {strongly_cc}")

    # Distribuição de graus
    st.subheader("📈 Distribuição de Grau")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(
        [G.in_degree(n) for n in G.nodes()], bins=30, ax=axs[0], color="skyblue"
    )
    axs[0].set_title("In-Degree")
    sns.histplot(
        [G.out_degree(n) for n in G.nodes()], bins=30, ax=axs[1], color="orange"
    )
    axs[1].set_title("Out-Degree")
    st.pyplot(fig)

    # Centralidades
    st.subheader("⭐ Centralidades e Nós Mais Importantes")
    st.markdown("Selecione uma métrica de centralidade para visualizar os top-k nós:")
    metric = st.selectbox(
        "Métrica", ["Degree", "Closeness", "Betweenness", "Eigenvector"]
    )
    k = st.slider("Top-k nós", min_value=5, max_value=50, value=10)

    if metric == "Degree":
        centrality = nx.degree_centrality(G)
    elif metric == "Closeness":
        centrality = nx.closeness_centrality(G)
    elif metric == "Betweenness":
        centrality = nx.betweenness_centrality(G)
    elif metric == "Eigenvector":
        try:
            centrality = nx.eigenvector_centrality(G)
        except nx.PowerIterationFailedConvergence:
            st.error("❌ Erro ao calcular eigenvector centrality. Grafo muito esparso.")
            centrality = {}

    if centrality:
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:k]
        df_central = pd.DataFrame(top_nodes, columns=["Nó", "Centralidade"])
        st.dataframe(df_central)

        # Visualização Pyvis dos top-k
        st.subheader("🔝 Visualização dos Top-k Nós")
        H = G.subgraph([n for n, _ in top_nodes])
        net2 = Network(height="600px", width="100%", directed=True, notebook=False)
        net2.from_nx(H)
        tmp_dir2 = tempfile.mkdtemp()
        path2 = os.path.join(tmp_dir2, "topk.html")
        net2.save_graph(path2)
        st.components.v1.html(open(path2, "r", encoding="utf-8").read(), height=600)
