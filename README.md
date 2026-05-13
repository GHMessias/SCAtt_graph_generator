# synthetic_graphgen

Implementacao experimental do **SCAtt (Synthetic Community-Aware Attributed Graph Generator)**, um gerador de grafos atribuidos voltado para benchmarking de Graph Neural Networks, clustering e deteccao de comunidades.

O projeto acompanha o artigo em PDF presente neste diretorio: `SCAtt: Synthetic Community-Aware Attributed Graph Generator for Graph Neural Network Benchmarking`.

## O que este codigo gera

O pacote gera um grafo atribuido:

```text
G = (V, E, X, Y)
```

onde:

- `V`: conjunto de nos;
- `E`: conjunto de arestas;
- `X`: matriz de atributos/features dos nos;
- `Y`: rotulos/classes/comunidades dos nos.

O gerador principal e o `SCAttGenerator`, definido em `models/SCatt.py`. Ele possui dois modos:

- `generate(...)`: cria um grafo sintetico a partir de parametros definidos pelo usuario.
- `mimic(...)`: clona/mimetiza um grafo atribuido existente e, opcionalmente, aumenta seu numero de nos.

## Estrutura do repositorio

```text
core/
  attributed_graph.py        # Estrutura comum para grafo + features + labels
  base_graph_generator.py    # Classe base dos geradores

models/
  SCatt.py                   # Implementacao principal do SCAtt
  GenCAT.py                  # Baseline de geracao/mimicagem
  SkyMap.py                  # Baseline de geracao/mimicagem
  chunglu.py                 # Baseline Chung-Lu para mimicagem
  BTER.py, Cabam.py, XMark.py

experiments/
  experiments.py             # Experimentos principais
  func.py                    # Argumentos, treino GAE+KMeans e serializacao

analysis/
  sfanalysis.py              # Analise scale-free/power-law

DGCluster/
  Codigo usado para benchmark com DGCluster

test/
  Notebooks e scripts exploratorios
```

## Entradas do SCAtt

No modo `generate`, os principais parametros sao:

| Parametro | Descricao |
| --- | --- |
| `num_nodes` | Numero total de nos. Deve ser igual a `sum(y)`. |
| `k` | Numero de comunidades/classes. Deve ser igual a `len(y)`. |
| `y` | Quantidade de nos por comunidade. Ex.: `[40, 40, 40]`. |
| `e` | Numero de arestas homogeneas por comunidade. |
| `d` | Distribuicao de grau por comunidade: `"power_law"`, `"normal"` ou `"uniform"`. |
| `M` | Vetores com a proporcao de nos em cada subcomunidade de cada classe. |
| `C` | Matrizes de interacao entre subcomunidades dentro de cada classe. |
| `N` | Matriz de interacao entre comunidades diferentes. A diagonal deve ser zero. |
| `rho` | Densidade alvo usada para adicionar arestas heterogeneas/ruido topologico. |
| `dimensions` | Dimensao da matriz de atributos `X`. |
| `sigma` | Intensidade do ruido gaussiano nos atributos. No artigo, corresponde ao parametro lambda. |
| `alpha` | Peso entre atributos de classe e suavizacao topologica. |
| `alpha_powerlaw` | Expoente usado quando `d="power_law"`. |

### Interpretacao de `M`, `C` e `N`

`M` define o tamanho relativo das subcomunidades. Por exemplo:

```python
M = [
    torch.tensor([1.0]),              # classe 0: uma subcomunidade
    torch.tensor([0.4, 0.3, 0.3]),    # classe 1: tres subcomunidades
    torch.tensor([0.9, 0.1]),         # classe 2: duas subcomunidades
]
```

`C` define como essas subcomunidades se conectam dentro da mesma classe. Cada `C[c]` deve ser uma matriz quadrada com tamanho igual ao numero de subcomunidades de `M[c]`.

```python
C = [
    torch.tensor([[1.0]]),
    torch.tensor([
        [0.6, 0.3, 0.1],
        [0.1, 0.7, 0.2],
        [0.0, 0.3, 0.7],
    ]),
    torch.tensor([
        [0.9, 0.1],
        [0.1, 0.9],
    ]),
]
```

Na geracao de arestas homogeneas, o SCAtt agora usa:

1. `M[c]` para escolher a subcomunidade de origem;
2. a linha `C[c][origem, :]` para escolher a subcomunidade de destino;
3. `d[c]` para escolher os nos dentro das subcomunidades.

`N` controla o ruido entre comunidades. A entrada `N[i, j]` indica a tendencia de criar uma aresta heterogenea entre as classes `i` e `j`.

## Exemplo minimo

```python
import torch

from models.SCatt import SCAttGenerator

y = [40, 40, 40]
e = [150, 200, 100]
k = len(y)

M = [
    torch.tensor([1.0]),
    torch.tensor([0.4, 0.3, 0.3]),
    torch.tensor([0.9, 0.1]),
]

C = [
    torch.tensor([[1.0]]),
    torch.tensor([
        [0.6, 0.3, 0.1],
        [0.1, 0.7, 0.2],
        [0.0, 0.3, 0.7],
    ]),
    torch.tensor([
        [0.9, 0.1],
        [0.1, 0.9],
    ]),
]

N = torch.tensor([
    [0.0, 0.2, 0.8],
    [0.3, 0.0, 0.7],
    [0.5, 0.5, 0.0],
])

graph = SCAttGenerator(seed=2026).generate(
    num_nodes=sum(y),
    y=y,
    k=k,
    e=e,
    C=C,
    d=["power_law", "normal", "uniform"],
    rho=0.08,
    N=N,
    M=M,
    dimensions=60,
    sigma=0.3,
    alpha=0.5,
)

print(graph)
data = graph.to_data_pytorch()
```

## Plotando um grafo SCAtt

Depois de gerar um `AttributedGraph`, use `graphPlotter.plot_scatt_graph`:

```python
from plot.plot import graphPlotter

plotter = graphPlotter()
fig, ax = plotter.plot_scatt_graph(
    graph,
    layout="community",
    keep="all",
    community_halos=True,
    highlight_interclass_edges=True,
    vertex_size_range=(180, 620),
    save_path="scatt_graph.png",
)
```

O `layout="community"` usa os rotulos `y` para iniciar o desenho de forma organizada, mas ainda roda um layout de forca global para que comunidades conectadas fiquem mais proximas. Para um layout sem usar rotulos, use `layout="spring"`. Para o desenho antigo em blocos separados, use `layout="community_blocks"`. Para comparar com o render antigo baseado em igraph, use `backend="igraph"`.

Para uma figura mais apropriada para artigo, use o backend `netgraph`:

```python
fig, ax = plotter.plot_scatt_graph(
    graph,
    backend="netgraph",
    layout="community",
    netgraph_edge_layout="bundled",
    netgraph_bundle_k=2000,
    vertex_size_range=(120, 360),
    node_edge_width=0,
    edge_alpha=0.10,
    figsize=(7.2, 6.2),
    save_path="scatt_publication.png",
)
```

Esse estilo segue o exemplo do Netgraph: cores por comunidade, borda do vertice removida e arestas agrupadas. Se ficar lento em grafos maiores, troque `netgraph_edge_layout="bundled"` por `"curved"`. O backend `netgraph` requer:

```bash
pip install netgraph
```

Tambem e possivel gerar e plotar em uma chamada:

```python
from plot.plot import graphPlotter

graph, fig, ax = graphPlotter().generate_and_plot_scatt(
    seed=2026,
    num_nodes=sum(y),
    y=y,
    k=k,
    e=e,
    C=C,
    d=["power_law", "normal", "uniform"],
    rho=0.08,
    N=N,
    M=M,
    dimensions=60,
    sigma=0.3,
    alpha=0.5,
    plot_kwargs={
        "save_path": "scatt_graph.png",
        "layout": "community",
        "community_halos": True,
        "highlight_interclass_edges": True,
    },
)
```

## Mimicagem e aumento de dados

Para mimetizar um grafo existente, primeiro crie um `AttributedGraph`:

```python
from core.attributed_graph import AttributedGraph
from models.SCatt import SCAttGenerator

mimicked = SCAttGenerator(seed=2026).mimic(
    base_graph=base_graph,
    num_nodes=None,  # ou um valor maior que base_graph.num_nodes()
)
```

No modo `mimic`, o algoritmo:

1. usa os rotulos `Y` do grafo original;
2. detecta subcomunidades dentro de cada classe com PLM/Louvain;
3. reconstrui arestas usando graus e matriz empirica de subcomunidades;
4. adiciona arestas heterogeneas de acordo com o padrao interclasse observado;
5. transfere atributos por rank de grau dentro de cada classe;
6. se `num_nodes` for maior que o original, adiciona novos nos preservando proporcoes de classe e subcomunidade.

## Experimentos

Os experimentos principais ficam em `experiments/experiments.py`.

```bash
python experiments/experiments.py --experiment topology_rho
python experiments/experiments.py --experiment sfanalysis
python experiments/experiments.py --experiment DGCluster
```

O experimento `topology_rho` gera grafos SCAtt e avalia embeddings com GAE + KMeans usando NMI. O experimento `sfanalysis` compara mimicagem com GenCAT, SCAtt e Chung-Lu usando analise power-law/scale-free. O experimento `DGCluster` gera datasets sinteticos para avaliar o algoritmo DGCluster.

## Dependencias principais

Este repositorio nao possui um arquivo de dependencias fixado. Pelos imports do codigo, as bibliotecas principais sao:

- `torch`
- `torch_geometric`
- `networkit`
- `numpy`
- `scipy`
- `scikit-learn`
- `pandas`
- `matplotlib`
- `seaborn`
- `powerlaw` para `analysis/sfanalysis.py`

## Observacoes

- O grafo e nao direcionado e nao ponderado.
- Nos isolados podem ser removidos no modo `generate`, seguindo o fluxo descrito no artigo antes da reindexacao.
- A diagonal de `N` deve ser zero, pois arestas homogeneas sao controladas por `e`, `M`, `C` e `d`.
- `sigma` no codigo corresponde ao parametro lambda de dispersao dos atributos descrito no artigo.
