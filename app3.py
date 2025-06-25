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

    # =============================================
    # MÉTRICAS ESTRUTURAIS(Sem filtros)
    # =============================================
    # TEXTOS DE AJUDA (Métricas Estruturais)
    help_densidade = (
        "A densidade é a razão entre o número de arestas existentes e o número máximo possível. "
        "Varia de 0 a 1.\n\n"
        "- 0: grafo extremamente esparso (poucas conexões)\n"
        "- 1: grafo completamente conectado (todos os nós se conectam entre si)\n\n"
        "Útil para entender quão interligada é a rede."
    )

    help_assortatividade = (
        "Mede a tendência de nós se conectarem com outros de grau similar. "
        "Varia de -1 a +1.\n\n"
        "- Valor > 0: nós com grau alto conectam-se a outros com grau alto (ex: redes sociais)\n"
        "- Valor < 0: nós com grau alto conectam-se a nós com grau baixo (ex: redes tecnológicas)\n"
        "- Valor ≈ 0: conexões são aleatórias em relação ao grau dos nós\n\n"
        "Ajuda a entender o padrão de conexões da rede."
    )

    help_clustering = (
        "Mede o grau de agrupamento (formação de triângulos) entre os vizinhos de um nó. "
        "Varia de 0 a 1.\n\n"
        "- Valor próximo de 1: forte tendência de formar grupos (alta coesão local)\n"
        "- Valor próximo de 0: pouca ou nenhuma formação de grupos\n\n"
        "Comum em redes sociais e redes pequenas-mundo."
    )

    help_scc = (
        "Número de subgrafos nos quais **cada nó pode alcançar todos os outros seguindo a direção das arestas**.\n\n"
        "- Relevante em redes dirigidas (como grafos de citações ou hyperlinks).\n"
        "- Valor maior indica uma rede mais fragmentada em termos de alcance direcional.\n"
        "- Um único componente forte sugere alta conectividade mútua.\n\n"
        "Ex: Em um SCC, se A alcança B, então B também alcança A por algum caminho dirigido."
    )

    help_wcc = (
        "Número de subgrafos nos quais os nós estão conectados **se ignorarmos a direção das arestas**.\n\n"
        "- Mede a conectividade geral da estrutura, desconsiderando direcionalidade.\n"
        "- Útil para avaliar fragmentação estrutural bruta da rede.\n"
        "- Um único componente fraco indica que todos os nós estão ligados por algum caminho (mesmo que indirecional).\n\n"
        "Ex: A pode não alcançar B na direção correta, mas ainda faz parte do mesmo grupo fraco."
    )

    # =============================================
    # MÉTRICAS ESTRUTURAIS
    # =============================================
    with st.expander("📊 Métricas Estruturais da Rede", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Densidade", f"{nx.density(G):.4f}", help=help_densidade)
            st.metric(
                "Assortatividade(grau do nó)",
                f"{nx.degree_assortativity_coefficient(G):.4f}",
                help=help_assortatividade,
            )
            st.metric(
                "Coef. Clustering",
                f"{nx.average_clustering(G.to_undirected()):.4f}",
                help=help_clustering,
            )

        with col2:
            scc = (
                nx.number_strongly_connected_components(G)
                if nx.is_directed(G)
                else "N/A"
            )
            st.metric("Componentes Fortemente Conectados", scc, help=help_scc)
            st.metric(
                "Componentes Fracamente Conectados",
                nx.number_weakly_connected_components(G),
                help=help_wcc,
            )

    # =============================================
    # DISTRIBUIÇÃO DE GRAU
    # =============================================
    with st.expander("📈 Distribuição de Grau", expanded=False):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        # Grau de entrada
        sns.histplot(
            list(dict(G.in_degree()).values()),
            bins=30,
            ax=ax[0],
            color="skyblue",
        )
        ax[0].set_title("Distribuição do Grau de Entrada")
        ax[0].set_xlabel("Grau de entrada")
        ax[0].set_ylabel("Frequência")
        ax[0].set_yscale("log")  # <- Aplica escala log no eixo Y

        # Grau de saída
        sns.histplot(
            list(dict(G.out_degree()).values()),
            bins=30,
            ax=ax[1],
            color="salmon",
        )
        ax[1].set_title("Distribuição do Grau de Saída")
        ax[1].set_xlabel("Grau de saída")
        ax[1].set_ylabel("Frequência")
        ax[1].set_yscale("log")  # <- Também aplica no segundo gráfico

        st.pyplot(fig)
        st.markdown(
            """A distribuição de grau mostra como os nós estão conectados na rede. A escala logarítmica no eixo Y ajuda a visualizar melhor as frequências, especialmente em redes com muitos nós de grau baixo e poucos de grau alto.
            Uma distribuição com cauda longa indica que poucos nós têm muitos links, enquanto a maioria tem poucos."""
        )

    # =============================================
    # CONTROLES DE FILTRO (NO CORPO PRINCIPAL)
    # =============================================
    st.subheader("🔍 Filtros do Grafo")
    col1, col2, col3 = st.columns(3)

    with col1:
        min_degree = st.slider("Grau mínimo do nó", 1, 10, 6)
    with col2:
        show_largest_scc = st.checkbox("Mostrar maior SCC (dirigido)", value=False)
    with col3:
        show_largest_wcc = st.checkbox("Mostrar maior WCC", value=False)

    # =============================================
    # APLICAÇÃO DOS FILTROS
    # =============================================
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

    # =============================================
    # VISUALIZAÇÃO PRINCIPAL DO GRAFO
    # =============================================
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

        # Configuração FIXA do layout (sem sliders)
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

        # Calcular graus de entrada e saída
        in_degrees = dict(SG.in_degree())
        out_degrees = dict(SG.out_degree())

        # Evita divisão por zero
        max_in = max(in_degrees.values()) if in_degrees else 1
        max_out = max(out_degrees.values()) if out_degrees else 1

        # Adicionando nós com personalização por tipo e grau
        for node in SG.nodes():
            k_in = in_degrees.get(node, 0)
            k_out = out_degrees.get(node, 0)

            diff = abs(k_in - k_out)
            total = k_in + k_out if (k_in + k_out) > 0 else 1
            balance_ratio = (
                diff / total
            )  # 0 = totalmente balanceado, 1 = totalmente desbalanceado
            # Determina o tipo de nó e sua coloração
            if k_in > k_out:
                hue = 290 - (70 * balance_ratio)  # Vai de 290 (roxo) até 220 (azul)
                node_type = "Receptor"
                degree_info = f"Grau entrada: {k_in}, saída: {k_out}"
            elif k_out > k_in:
                hue = 290 - (280 * balance_ratio)  # Vai de 290 (roxo) até 10 (vermelho)
                node_type = "Emissor"
                degree_info = f"Grau saída: {k_out}, entrada: {k_in}"
            else:
                hue = 290  # Roxo vivo
                node_type = "Balanceado"
                degree_info = f"Grau entrada/saída: {k_in}"

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

        # Adicionando arestas com personalização FIXA
        for u, v, data in SG.edges(data=True):
            weight = data.get("weight", 1.0)
            net.add_edge(
                u,
                v,
                width=0.5 + float(weight) * 0.1,
                title=f"Peso: {weight:.2f}",
                color="#97C2FC",
            )

        # Salvar e exibir
        tmp_dir = tempfile.mkdtemp()
        path = os.path.join(tmp_dir, "graph.html")
        net.save_graph(path)
        st.components.v1.html(open(path, "r", encoding="utf-8").read(), height=750)

    # =============================================
    # MÉTRICAS ESTRUTURAIS
    # =============================================
    with st.expander(f"📊 Métricas Estruturais da Rede(C/Filtro)", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Densidade", f"{nx.density(SG):.4f}", help=help_densidade)
            st.metric(
                "Assortatividade(grau do nó)",
                f"{nx.degree_assortativity_coefficient(SG):.4f}",
                help=help_assortatividade,
            )
            st.metric(
                "Coef. Clustering",
                f"{nx.average_clustering(SG.to_undirected()):.4f}",
                help=help_clustering,
            )

        with col2:
            scc = (
                nx.number_strongly_connected_components(SG)
                if nx.is_directed(SG)
                else "N/A"
            )
            st.metric("Componentes Fortemente Conectados", scc, help=help_scc)
            st.metric(
                "Componentes Fracamente Conectados",
                nx.number_weakly_connected_components(SG),
                help=help_wcc,
            )

    # =============================================
    # DISTRIBUIÇÃO DE GRAU
    # =============================================
    with st.expander("📈 Distribuição de Grau (C/filtro)", expanded=False):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        # Grau de entrada
        sns.histplot(
            list(dict(SG.in_degree()).values()),
            bins=30,
            ax=ax[0],
            color="skyblue",
        )
        ax[0].set_title("Distribuição do Grau de Entrada")
        ax[0].set_xlabel("Grau de entrada")
        ax[0].set_ylabel("Frequência")
        # Escala linear padrão (sem set_yscale)

        # Grau de saída
        sns.histplot(
            list(dict(SG.out_degree()).values()),
            bins=30,
            ax=ax[1],
            color="salmon",
        )
        ax[1].set_title("Distribuição do Grau de Saída")
        ax[1].set_xlabel("Grau de saída")
        ax[1].set_ylabel("Frequência")
        # Escala linear padrão

        st.pyplot(fig)
        st.markdown(
            """
            A distribuição de grau mostra como os nós estão conectados na rede. 
            Uma distribuição com cauda longa indica que poucos nós têm muitos links, 
            enquanto a maioria tem poucos.
            """
        )

    # =============================================
    # ANÁLISE DE CENTRALIDADE (COM CONTROLES NO CORPO)
    # =============================================

    with st.expander("⭐ Análise de Centralidade", expanded=False):
        st.markdown(
            """
        Compare as diferentes medidas de centralidade para identificar os nós mais importantes:
        - **Degree**: Nós mais conectados
        - **Closeness**: Nós que podem alcançar outros mais rapidamente
        - **Betweenness**: Nós que atuam como pontes
        - **Eigenvector**: Nós conectados a outros nós importantes
        """
        )

        # Controles no corpo principal (não na sidebar)
        col1, col2 = st.columns(2)
        with col1:
            metric = st.selectbox(
                "Métrica de Centralidade",
                ["Degree", "Closeness", "Betweenness", "Eigenvector"],
            )
        with col2:
            k = st.slider("Número de nós para mostrar", 5, 50, 10)

        # Cálculo da centralidade
        if metric == "Degree":
            centrality = nx.degree_centrality(G)
        elif metric == "Closeness":
            centrality = nx.closeness_centrality(G)
        elif metric == "Betweenness":
            centrality = nx.betweenness_centrality(G)
        elif metric == "Eigenvector":
            try:
                centrality = nx.eigenvector_centrality(G, max_iter=1000)
            except nx.PowerIterationFailedConvergence:
                st.error(
                    "Não foi possível calcular Eigenvector Centrality (grafo muito esparso)"
                )
                centrality = {}

        if centrality:
            # Mostra os nós mais centrais em uma tabela
            top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:k]
            st.dataframe(pd.DataFrame(top_nodes, columns=["Nó", "Centralidade"]))

            # Visualização dos top-k nós (com configurações FIXAS)
            st.markdown("### 🔝 Visualização dos Nós Mais Importantes")
            H = G.subgraph([n for n, _ in top_nodes])

            net2 = Network(
                notebook=False,
                height="500px",
                width="100%",
                directed=True,
                bgcolor="#222222",
                font_color="white",
                cdn_resources="remote",
            )

            # Mesma configuração FIXA do layout
            net2.force_atlas_2based(
                gravity=-30,
                central_gravity=0.005,
                spring_length=150,
                spring_strength=0.03,
                damping=0.85,
                overlap=1,
            )
            net2.toggle_physics(True)
            net2.options.configure = {"enabled": False}

            # Calcular graus de entrada e saída
            in_degrees = dict(H.in_degree())
            out_degrees = dict(H.out_degree())

            # Evita divisão por zero se grafo estiver vazio
            max_in = max(in_degrees.values()) if in_degrees else 1
            max_out = max(out_degrees.values()) if out_degrees else 1

            # Adiciona nós com coloração e tamanho baseados em grau
            for node in H.nodes():
                k_in = in_degrees.get(node, 0)
                k_out = out_degrees.get(node, 0)

                diff = abs(k_in - k_out)
                total = k_in + k_out if (k_in + k_out) > 0 else 1
                balance_ratio = (
                    diff / total
                )  # 0 = totalmente balanceado, 1 = totalmente desbalanceado
                # Determina o tipo de nó e sua coloração

                if k_in > k_out:
                    hue = 290 - (70 * balance_ratio)
                    node_type = "Receptor"
                    degree_info = f"Grau entrada: {k_in}, Grau saída: {k_out}"
                elif k_out > k_in:
                    hue = 290 - (
                        280 * balance_ratio
                    )  # Vai de 290 (roxo) até 10(vermelho)
                    node_type = "Emissor"
                    degree_info = f"Grau saída: {k_out}, Grau entrada: {k_in}"
                else:
                    hue = 290  # Roxo vivo
                    node_type = "Balanceado"
                    degree_info = f"Grau entrada/saída: {k_in}"

                saturation = 70
                lightness = 60 - min(20, total / max(max_in, max_out) * 20)
                degree_info = f"Grau entrada: {k_in}, saída: {k_out}"

                net2.add_node(
                    node,
                    label=str(node),
                    size=10 + (k_in + k_out) * 0.2,
                    color=f"hsl({hue}, {saturation}%, {lightness}%)",
                    title=(
                        f"Nó: {node}\n"
                        f"Tipo: {node_type}\n"
                        f"{degree_info}\n"
                        f"{metric}: {centrality.get(node, 0):.4f}"
                    ),
                )

            # Adiciona arestas (configuração FIXA)
            for u, v, data in H.edges(data=True):
                weight = data.get("weight", 1.0)
                net2.add_edge(
                    u,
                    v,
                    width=0.5 + float(weight) * 0.1,
                    title=f"Peso: {weight:.2f}",
                    color="#FF9999",  # Cor mais clara para arestas
                )

            # Exibe o grafo
            path2 = os.path.join(tempfile.mkdtemp(), "topk.html")
            net2.save_graph(path2)
            st.components.v1.html(open(path2, "r", encoding="utf-8").read(), height=500)

    # =============================================
    # EXPORTAÇÃO DO GRAFO
    # =============================================

    with st.expander("📤 Exportar Grafo", expanded=False):
        st.markdown(
            "Você pode exportar o grafo filtrado para um arquivo GraphML ou GEXF."
        )
        col1, col2 = st.columns(2)

        with col1:
            export_format = st.selectbox("Formato de Exportação", ["GraphML", "GEXF"])
        with col2:
            export_button = st.button("Exportar Grafo")

        if export_button:
            # Define nome com base nos filtros aplicados
            filtros_aplicados = []
            if show_largest_scc:
                filtros_aplicados.append("scc")
            if show_largest_wcc:
                filtros_aplicados.append("wcc")
            if min_degree > 1:
                filtros_aplicados.append(f"grau_min_{min_degree}")

            grafo_filtrado = (
                "_".join(filtros_aplicados) if filtros_aplicados else "original"
            )
            export_filename = f"grafo_{grafo_filtrado}.{export_format.lower()}"

            # Cria arquivo temporário com nome significativo
            tmp_dir = tempfile.gettempdir()
            tmp_path = os.path.join(tmp_dir, export_filename)

            # Salva o grafo
            if export_format == "GraphML":
                nx.write_graphml(SG, tmp_path)
            elif export_format == "GEXF":
                nx.write_gexf(SG, tmp_path)

            # Informa sucesso e botão de download
            st.success(f"Grafo exportado como {export_filename}")
            with open(tmp_path, "rb") as f:
                st.download_button(
                    label="📥 Baixar Arquivo",
                    data=f.read(),
                    file_name=export_filename,
                    mime="application/octet-stream",
                )

            # Remove arquivo após leitura
            os.remove(tmp_path)
            st.info(
                "O arquivo será removido após o download para evitar acúmulo de arquivos temporários."
            )
