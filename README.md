# 🔍 Análise e Visualização de Redes

Este projeto é uma aplicação web interativa desenvolvida com **Streamlit** para análise e visualização de redes direcionadas a partir de dados tabulares.

---

## 🚀 Funcionalidades

- Upload de arquivos CSV contendo colunas `source` e `target` para criação do grafo direcionado.
- Cálculo e exibição de métricas estruturais da rede:
  - Densidade
  - Assortatividade de grau
  - Coeficiente de clustering
  - Número de componentes fortemente e fracamente conectados
- Visualização gráfica estática com Matplotlib:
  - Destaque visual por métricas de centralidade (tamanho e cor dos nós)
  - Layout aleatório fixo para reprodutibilidade
  - Rótulos para os 5 nós mais centrais por métrica
- Visualização gráfica interativa da rede com **PyVis**, incluindo:
  - Cores e tamanhos dos nós baseados em graus de entrada e saída
  - Visualização das arestas com pesos (se presentes)
  - Layout ForceAtlas2 com física ativada
- Filtros para subgrafo com base em:
  - Grau mínimo e máximo de entrada ou de saída dos nós
  - Maior componente fortemente conectado (SCC)
  - Maior componente fracamente conectado (WCC)
- Visualização das distribuições de grau (entrada e saída) com histogramas usando Matplotlib e Seaborn
- Análise de centralidade com múltiplas métricas (degree, closeness, betweenness e eigenvector)
- Visualização dos nós mais centrais ordenados por métrica selecionada

---

## 🛠 Tecnologias utilizadas

- [Python 3](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [NetworkX](https://networkx.org/)
- [PyVis](https://pyvis.readthedocs.io/en/latest/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Numpy](https://numpy.org/)

---

## 📥 Como usar

1. Clone este repositório:

   ```bash
   git clone https://github.com/Croncl/streamlit-wiki-network.git
   cd streamlit-wiki-network
   ```

2. (Opcional) Crie e ative um ambiente virtual:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

4. Execute a aplicação:

   ```bash
   streamlit run app.py
   ```

5. Acesse no navegador: [http://localhost:8501](http://localhost:8501)

6. Faça upload de um arquivo CSV com colunas `source` e `target` para começar a análise.

---

## 🗂 Formato esperado do arquivo CSV

| source | target |
|--------|--------|
| A      | B      |
| B      | C      |
| C      | A      |

* As colunas indicam uma aresta dirigida de `source` para `target`.
* Pesos são opcionais.

---

## 📈 Métricas calculadas

- **Densidade:** Quão conectada está a rede (0 a 1).
- **Assortatividade:** Tendência de nós com grau similar se conectarem.
- **Coeficiente de Clustering:** Grau de formação de triângulos no grafo.
- **Componentes Fortemente Conectados (SCC):** Subgrafos com nós mutuamente alcançáveis via caminhos dirigidos.
- **Componentes Fracamente Conectados (WCC):** Subgrafos considerando o grafo como não direcionado.

---
