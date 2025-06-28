import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import tempfile
import os

# Configuração da página
st.set_page_config(layout="wide", page_title="Análise de Redes")
st.title("🔍 Análise e Visualização de Redes")
st.markdown(
    """
Esta aplicação analisa redes a partir de dados de relacionamento.
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

    def calcular_assortatividade_segura(grafo):
        """
        Calcula o coeficiente de assortatividade de forma segura, evitando erros em grafos esparsos.
        
        Args:
            grafo: Um grafo NetworkX (Graph ou DiGraph)
        
        Returns:
            float: Valor da assortatividade ou None se não for possível calcular
        """
        try:
            # Verifica se o grafo tem nós suficientes e variação de graus
            if grafo.number_of_nodes() < 2:
                return None
                
            degrees = [d for _, d in grafo.degree()]
            if len(set(degrees)) < 2:  # Todos os nós têm o mesmo grau
                return None
                
            return nx.degree_assortativity_coefficient(grafo)
        except (ZeroDivisionError, nx.NetworkXError):
            return None
    

# =============================================
# MÉTRICAS ESTRUTURAIS E VISUALIZAÇÃO ESTÁTICA
# =============================================
    with st.expander("📊 Métricas Estruturais da Rede", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Densidade", f"{nx.density(G):.4f}", help=help_densidade)
            assort = calcular_assortatividade_segura(G)
            st.metric(
                "Assortatividade(grau do nó)",
                f"{assort:.4f}" if assort is not None else "N/A",
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
# VISUALIZAÇÃO ESTÁTICA
# =============================================

    with st.expander("📊 Visualização Estática do Grafo (Métricas de Centralidade)", expanded=False):
        st.markdown("""
        ## Comparação Visual das Métricas de Centralidade
        
        - **Tamanho do nó**: proporcional à centralidade
        - **Cor**: escala contínua de importância (azul → vermelho, personalizável)
        """)

        col1, col2 = st.columns(2)
        with col1:
            layout_option = st.selectbox(
                "Layout de visualização:",
                ["Spring", "Circular", "Kamada-Kawai", "Random"],
                key="layout_option"
            )
        with col2:
            metric_option = st.selectbox(
                "Métrica de Centralidade:",
                ["Degree Centrality", "Closeness Centrality", "Betweenness Centrality", "Eigenvector Centrality"],
                key="static_viz_metric"
            )

        generate_plot = st.button("📈 Gerar Visualização Estática")

        if generate_plot:
            # Layouts
            layouts = {
                "Spring": nx.spring_layout(G, k=0.15, iterations=20, seed=42),
                "Circular": nx.circular_layout(G),
                "Kamada-Kawai": nx.kamada_kawai_layout(G),
                "Random": nx.random_layout(G, seed=42)
            }
            pos = layouts.get(layout_option, nx.spring_layout(G))

            # Cálculo da métrica
            centrality = {}
            try:
                centrality_methods = {
                    "Degree Centrality": (nx.degree_centrality, "Degree Centrality (Mais conexões)"),
                    "Closeness Centrality": (nx.closeness_centrality, "Closeness Centrality (Acesso rápido)"),
                    "Betweenness Centrality": (nx.betweenness_centrality, "Betweenness Centrality (Pontes estratégicas)"),
                    "Eigenvector Centrality": (lambda G: nx.eigenvector_centrality(G, max_iter=1000), "Eigenvector Centrality (Influência)")
                }
                func, title = centrality_methods[metric_option]
                centrality = func(G)
            except nx.NetworkXError as e:
                st.error(f"Erro ao calcular {metric_option}: {e}")

            if centrality:
                # Cores ajustáveis
                cmap = plt.cm.get_cmap("coolwarm")  # pode trocar para "viridis", "plasma", "Reds", etc.
                background_color = "#04051B"

                cent_vals = np.array(list(centrality.values()))
                norm = (cent_vals - cent_vals.min()) / (cent_vals.ptp() + 1e-5)
                sizes = 50 + 500 * norm
                colors = cmap(norm)

                # Plot
                plt.figure(figsize=(12, 10), facecolor=background_color)
                ax = plt.gca()
                ax.set_facecolor(background_color)

                nodes = nx.draw_networkx_nodes(
                    G, pos,
                    node_size=sizes,
                    node_color=colors,
                    cmap=cmap,
                    alpha=0.85
                )
                nx.draw_networkx_edges(
                    G, pos,
                    edge_color='white',
                    width=0.5,
                    alpha=0.15,
                    arrows=True,
                    arrowsize=6
                )

                # Labels top 5
                top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                for node, _ in top_nodes:
                    x, y = pos[node]
                    plt.text(x, y, str(node), fontsize=9, ha='center', va='center',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.2'))

                plt.title(f"{title} — Layout: {layout_option}", fontsize=13, color='white')

                # Adiciona colorbar com texto em branco
                cb = plt.colorbar(nodes, label="Centralidade", shrink=0.75)
                cb.ax.yaxis.label.set_color('white')  # Cor do rótulo da barra
                cb.ax.tick_params(colors='white')     # Cor dos números da escala

                plt.axis('off')

                st.pyplot(plt, clear_figure=True)

                # Interpretação e Top 10
                st.markdown(f"""
                ### Interpretação:
                - **Nós maiores e mais vermelhos**: maior {metric_option.split()[0]} centrality
                - **Nós menores e azulados**: menor centralidade
                - **Top 5 nós** estão rotulados
                """)
                top_10_df = pd.DataFrame(
                    sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10],
                    columns=["Nó", "Centralidade"]
                )
                st.markdown("### 🔝 Top 10 Nós por Centralidade")
                st.dataframe(top_10_df.style.format({"Centralidade": "{:.4f}"}))
            else:
                st.warning("Não foi possível calcular a centralidade selecionada.")


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
    with st.expander("🌐 Visualização Interativa do Grafo", expanded=False):
        st.subheader("🔍 Filtros do Grafo")
        col1, col2, col3 = st.columns(3)

        with col1:
            # Calcula os valores mínimo e máximo de grau de entrada
            in_degrees = [d for n, d in G.in_degree()]
            min_in_degree = min(in_degrees) if in_degrees else 0
            max_in_degree = max(in_degrees) if in_degrees else 1
            
            in_degree_range = st.slider(
                "Intervalo de Grau de Entrada (in-degree)",
                min_value=min_in_degree,
                max_value=max_in_degree,
                value=(2, max_in_degree//4)
            )

            # Calcula os valores mínimo e máximo de grau de saída
            out_degrees = [d for n, d in G.out_degree()]
            min_out_degree = min(out_degrees) if out_degrees else 0
            max_out_degree = max(out_degrees) if out_degrees else 1
            
            out_degree_range = st.slider(
                "Intervalo de Grau de Saída (out-degree)",
                min_value=min_out_degree,
                max_value=max_out_degree,
                value=(2, max_out_degree//4)
            )

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
                f"Subgrafo: Maior SCC → {G.number_of_nodes()} nós, {G.number_of_edges()} arestas"
            )
        elif show_largest_wcc:
            largest_wcc = max(nx.weakly_connected_components(G), key=len)
            SG = G.subgraph(largest_wcc).copy()
            st.info(
                f"Subgrafo: Maior WCC → {G.number_of_nodes()} nós, {G.number_of_edges()} arestas"
            )
        else:
            # Filtragem de nós com base nos intervalos selecionados
            SG_nodes = [
                n for n in G.nodes()
                if in_degree_range[0] <= G.in_degree(n) <= in_degree_range[1]
                or out_degree_range[0] <= G.out_degree(n) <= out_degree_range[1]
            ]

            # Subgrafo induzido pelos nós filtrados
            SG = G.subgraph(SG_nodes).copy()

            # Exibir estatísticas do subgrafo
            st.info(
                f"Subgrafo (grau de entrada ∈ [{in_degree_range[0]}, {in_degree_range[1]}], "
                f"grau de saída ∈ [{out_degree_range[0]}, {out_degree_range[1]}]): "
                f"{G.number_of_nodes()} nós, {G.number_of_edges()} arestas"
            )

    # =============================================
    # VISUALIZAÇÃO PRINCIPAL DO GRAFO
    # =============================================
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
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())

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
            denominator = max(max_in, max_out)
            lightness = 60 - min(20, total / denominator * 20) if denominator > 0 else 60
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
            
            assort = calcular_assortatividade_segura(SG)
            st.metric(
                "Assortatividade(grau do nó)",
                f"{assort:.4f}" if assort is not None else "N/A",
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
            k = st.slider("Número de nós para mostrar", 1, 100, 10)

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

            # Calcular graus de entrada e saída #alterei H por G nos dois
            in_degrees = dict(G.in_degree())
            out_degrees = dict(G.out_degree())

            # Evita divisão por zero se grafo estiver vazio
            max_in = max(in_degrees.values()) if in_degrees else 1
            max_out = max(out_degrees.values()) if out_degrees else 1

            # Adiciona nós com coloração e tamanho baseados em grau
            for node in H.nodes():
                k_in = in_degrees.get(node, 0)
                k_out = out_degrees.get(node, 0)

                diff = abs(k_in - k_out)
                total = k_in + k_out
                balance_ratio = diff / (total + 1e-10)  # Evita divisão por zero
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
                
                max_connections = max(max_in, max_out)
                if max_connections == 0:
                    lightness = 60  # Valor padrão quando não há conexões
                else:
                    lightness = 60 - min(20, total / max_connections * 20)
                
                
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
