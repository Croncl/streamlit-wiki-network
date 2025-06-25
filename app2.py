import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os

# Configuração da página
st.set_page_config(layout="wide", page_title="Análise de Redes Complexas")
st.title("🔍 Análise e Visualização de Redes Complexas")
st.markdown(
    """
Esta aplicação analisa redes complexas a partir de dados de relacionamento.
Carregue um arquivo CSV com colunas 'source' e 'target' para começar.
"""
)

# Upload do arquivo
uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

if uploaded_file:
    # Processamento dos dados
    df = pd.read_csv(uploaded_file)
    df.dropna(subset=["source", "target"], inplace=True)
    G = nx.from_pandas_edgelist(
        df, source="source", target="target", create_using=nx.DiGraph()
    )
    G.remove_edges_from(nx.selfloop_edges(G))

    st.success(
        f"Grafo carregado: {G.number_of_nodes()} nós e {G.number_of_edges()} arestas"
    )

    # Métricas estruturais da rede (sem filtro)
    with st.expander("📊 Métricas Estruturais da Rede(Sem filtro)", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Densidade", f"{nx.density(G):.4f}")
            st.metric(
                "Assortatividade(grau do nó)",
                f"{nx.degree_assortativity_coefficient(G):.4f}",
            )
            st.metric(
                "Coef. Clustering", f"{nx.average_clustering(G.to_undirected()):.4f}"
            )

        with col2:
            scc = (
                nx.number_strongly_connected_components(G)
                if nx.is_directed(G)
                else "N/A"
            )
            st.metric("Componentes Fortemente Conectados", scc)
            st.metric(
                "Componentes Fracamente Conectados",
                nx.number_weakly_connected_components(G),
            )

    # Filtros
    st.subheader("🔍 Filtros do Grafo")
    col1, col2, col3 = st.columns(3)

    with col1:
        min_degree = st.slider("Grau mínimo do nó", 1, 10, 6)
    with col2:
        show_largest_scc = st.checkbox("Mostrar maior SCC (dirigido)", value=False)
    with col3:
        show_largest_wcc = st.checkbox("Mostrar maior WCC", value=False)

    if show_largest_scc and nx.is_directed(G):
        largest_scc = max(nx.strongly_connected_components(G), key=len)
        SG = G.subgraph(largest_scc).copy()
        st.info(
            f"Subgrafo: Maior SCC → {SG.number_of_nodes()} nós, {SG.number_of_edges()} arestas"
        )
    elif show_largest_wcc:
        largest_wcc = max(nx.weakly_connected_components(G), key=len)
        SG = G.subgraph(largest_wcc).copy()
        st.info(
            f"Subgrafo: Maior WCC → {SG.number_of_nodes()} nós, {SG.number_of_edges()} arestas"
        )
    else:
        SG_nodes = [n for n in G.nodes() if G.degree(n) >= min_degree]
        SG = G.subgraph(SG_nodes).copy()
        st.info(
            f"Subgrafo (grau ≥ {min_degree}): {SG.number_of_nodes()} nós, {SG.number_of_edges()} arestas"
        )

    # Visualização interativa do grafo
    with st.expander("🌐 Visualização Interativa do Grafo", expanded=True):
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
            central_gravity=0.005,
            spring_length=150,
            spring_strength=0.03,
            damping=0.85,
            overlap=1,
        )
        net.toggle_physics(True)
        net.options.configure = {"enabled": False}

        in_degrees = dict(SG.in_degree())
        out_degrees = dict(SG.out_degree())

        max_in = max(in_degrees.values()) if in_degrees else 1
        max_out = max(out_degrees.values()) if out_degrees else 1

        for node in SG.nodes():
            k_in = in_degrees.get(node, 0)
            k_out = out_degrees.get(node, 0)

            diff = abs(k_in - k_out)
            total = k_in + k_out if (k_in + k_out) > 0 else 1
            balance_ratio = diff / total

            if k_in > k_out:
                hue = 290 - (70 * balance_ratio)
                node_type = "Receptor"
            elif k_out > k_in:
                hue = 290 - (280 * balance_ratio)
                node_type = "Emissor"
            else:
                hue = 290
                node_type = "Balanceado"

            saturation = 70
            lightness = 60 - min(20, total / max(max_in, max_out) * 20)
            degree_info = f"Grau entrada: {k_in}, saída: {k_out}"

            net.add_node(
                node,
                label=str(node),
                size=10 + (k_in + k_out) * 0.2,
                color=f"hsl({hue}, {saturation}%, {lightness}%)",
                title=f"Nó: {node}\nTipo: {node_type}\n{degree_info}",
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

    # Métricas estruturais do subgrafo
    with st.expander(
        "\ud83d\udcca Métricas Estruturais da Rede(Filtro Grau WCC ou SCC )",
        expanded=False,
    ):
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Densidade", f"{nx.density(SG):.4f}")
            st.metric(
                "Assortatividade(grau do nó)",
                f"{nx.degree_assortativity_coefficient(SG):.4f}",
            )
            st.metric(
                "Coef. Clustering", f"{nx.average_clustering(SG.to_undirected()):.4f}"
            )

        with col2:
            scc = (
                nx.number_strongly_connected_components(SG)
                if nx.is_directed(SG)
                else "N/A"
            )
            st.metric("Componentes Fortemente Conectados", scc)
            st.metric(
                "Componentes Fracamente Conectados",
                nx.number_weakly_connected_components(SG),
            )

    # Distribuição de grau
    with st.expander("📈 Distribuição de Grau", expanded=False):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        sns.histplot(
            list(dict(G.in_degree()).values()), bins=30, ax=ax[0], color="skyblue"
        )
        ax[0].set_title("Distribuição do Grau de Entrada")
        ax[0].set_xlabel("Grau de entrada")
        ax[0].set_ylabel("Frequência")
        ax[0].set_yscale("log")

        sns.histplot(
            list(dict(G.out_degree()).values()), bins=30, ax=ax[1], color="salmon"
        )
        ax[1].set_title("Distribuição do Grau de Saída")
        ax[1].set_xlabel("Grau de saída")
        ax[1].set_ylabel("Frequência")
        ax[1].set_yscale("log")

        st.pyplot(fig)
        st.markdown(
            """
            A distribuição de grau mostra como os nós estão conectados na rede. 
            Uma distribuição com cauda longa indica que poucos nós têm muitos links, enquanto a maioria tem poucos.
            """
        )
