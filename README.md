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
| `n` | Numero total de nos. Deve ser igual a `sum(y)`. |
| `k` | Numero de comunidades/classes. Deve ser igual a `len(y)`. |
| `y` | Quantidade de nos por comunidade. Ex.: `[40, 40, 40]`. |
| `e` | Numero de arestas homogeneas por comunidade. |
| `dst` | Distribuicao de grau por comunidade: `"power_law"`, `"normal"` ou `"uniform"`. |
| `S` | Vetores com a proporcao de nos em cada subcomunidade de cada classe. |
| `A_in` | Matrizes de interacao entre subcomunidades dentro de cada classe. |
| `A_out` | Matriz de interacao entre comunidades diferentes. A diagonal deve ser zero. |
| `rho` | Numero total de arestas do grafo. Deve ser um inteiro maior ou igual a `sum(e)`. |
| `d` | Dimensao da matriz de atributos `X`. |
| `alpha_feat` | Intensidade do ruido gaussiano nos atributos. |
| `alpha_topo` | Peso entre atributos de classe e suavizacao topologica. |
| `alpha_powerlaw` | Expoente usado quando `dst="power_law"`. |

### Interpretacao de `S`, `A_in` e `A_out`

`S` define o tamanho relativo das subcomunidades. Por exemplo:

```python
S = [
    torch.tensor([1.0]),              # classe 0: uma subcomunidade
    torch.tensor([0.4, 0.3, 0.3]),    # classe 1: tres subcomunidades
    torch.tensor([0.9, 0.1]),         # classe 2: duas subcomunidades
]
```

`A_in` define como essas subcomunidades se conectam dentro da mesma classe. Cada `A_in[c]` deve ser uma matriz quadrada com tamanho igual ao numero de subcomunidades de `S[c]`.

```python
A_in = [
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

1. `S[c]` para escolher a subcomunidade de origem;
2. a linha `A_in[c][origem, :]` para escolher a subcomunidade de destino;
3. `dst[c]` para escolher os nos dentro das subcomunidades.

`A_out` controla o ruido entre comunidades. A entrada `A_out[i, j]` indica a tendencia ou quantidade relativa de arestas heterogeneas entre as classes `i` e `j`.

O numero de arestas heterogeneas e calculado por `rho - sum(e)`. Assim, `e` controla as arestas internas de cada comunidade, enquanto `A_out` controla como as arestas entre comunidades sao distribuidas.

## Exemplo minimo

```python
import torch

from models.SCatt import SCAttGenerator

y = [40, 40, 40]
e = [150, 200, 100]
k = len(y)

S = [
    torch.tensor([1.0]),
    torch.tensor([0.4, 0.3, 0.3]),
    torch.tensor([0.9, 0.1]),
]

A_in = [
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

A_out = torch.tensor([
    [0.0, 0.2, 0.8],
    [0.3, 0.0, 0.7],
    [0.5, 0.5, 0.0],
])

graph = SCAttGenerator(seed=2026).generate(
    n=sum(y),
    y=y,
    k=k,
    e=e,
    A_in=A_in,
    dst=["power_law", "normal", "uniform"],
    rho=600,
    A_out=A_out,
    S=S,
    d=60,
    alpha_feat=0.3,
    alpha_topo=0.5,
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
    edge_layout="bundled",
    edge_alpha=0.08,
    community_distance=1.35,
    save_path="scatt_graph.png",
)
```

Para plotar somente uma classe:

```python
fig, ax = plotter.plot_class(
    graph,
    class_id=1,
    save_path="scatt_class_1.png",
)
```

O plot novo usa `netgraph` para renderizar uma visualizacao limpa para artigo. Por padrao, o grafo completo usa `layout="community"` para organizar os vertices por rotulo e `edge_layout="bundled"` para agrupar visualmente as arestas. Arestas dentro da mesma classe recebem a cor da classe; arestas entre classes ficam pretas. Para visualizacoes sem usar rotulos, use `layout="spring"`.

Para ajustar o estilo da figura:

```python
from plot.plot import PublicationPlotStyle, graphPlotter

style = PublicationPlotStyle(
    figsize=(7.2, 6.2),
    edge_alpha=0.12,
    community_distance=1.35,
    edge_bundle_k=2000,
)

fig, ax = graphPlotter(style).plot_scatt_graph(
    graph,
    layout="community",
    edge_layout="bundled",
    save_path="scatt_publication.png",
)
```

Para comparar varios grafos mantendo os vertices no mesmo lugar, calcule as posicoes uma vez e reutilize em todos os paineis:

```python
import matplotlib.pyplot as plt

from plot.plot import PublicationPlotStyle, graphPlotter

graphs = [graph_600_100, graph_450_250, graph_300_400, graph_150_550]
titles = ["600 intra / 100 inter", "450 intra / 250 inter", "300 intra / 400 inter", "150 intra / 550 inter"]

plotter = graphPlotter(PublicationPlotStyle(node_size=0.8, edge_alpha=0.06))
node_positions = plotter.get_node_positions(
    graphs[0],
    layout="community",
    community_distance=1.35,
)

fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), dpi=300)
for ax, graph_i, title in zip(axes, graphs, titles):
    plotter.plot_scatt_graph(
        graph_i,
        ax=ax,
        node_positions=node_positions,
        edge_layout="bundled",
        title=title,
    )

fig.tight_layout()
fig.savefig("scatt_heterogeneous_control.png", bbox_inches="tight")
```

Tambem e possivel gerar e plotar em uma chamada:

```python
from plot.plot import graphPlotter

graph, fig, ax = graphPlotter().generate_and_plot_scatt(
    seed=2026,
    n=sum(y),
    y=y,
    k=k,
    e=e,
    A_in=A_in,
    dst=["power_law", "normal", "uniform"],
    rho=600,
    A_out=A_out,
    S=S,
    d=60,
    alpha_feat=0.3,
    alpha_topo=0.5,
    plot_kwargs={
        "save_path": "scatt_graph.png",
        "layout": "community",
        "edge_layout": "bundled",
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
- `networkx`
- `netgraph`
- `seaborn`
- `powerlaw` para `analysis/sfanalysis.py`

## Observacoes

- O grafo e nao direcionado e nao ponderado.
- A diagonal de `A_out` deve ser zero, pois arestas homogeneas sao controladas por `e`, `S`, `A_in` e `dst`.
