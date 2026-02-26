from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union, Iterable, Any
import math

import igraph as ig
import networkit as nk
import matplotlib.pyplot as plt

from core.attributed_graph import AttributedGraph

import torch
import numpy as np
import pandas as pd


def nk_to_igraph_remap(g_nk: nk.Graph) -> Tuple[ig.Graph, Dict[int, int], Dict[int, int]]:
    nodes = list(g_nk.iterNodes())
    old_to_new = {old: new for new, old in enumerate(nodes)}
    new_to_old = {new: old for old, new in old_to_new.items()}

    g_ig = ig.Graph(n=len(nodes), directed=g_nk.isDirected())
    edges = [(old_to_new[u], old_to_new[v]) for u, v in g_nk.iterEdges()]
    g_ig.add_edges(edges)
    return g_ig, old_to_new, new_to_old


@dataclass
class PlotStyle:
    layout: Union[str, ig.Layout] = "fr"
    dpi: int = 300
    bbox: Tuple[int, int] = (900, 900)
    margin: int = 35
    edge_width: float = 0.6

    vertex_size_range: Tuple[float, float] = (6.0, 28.0)
    degree_size_sqrt: bool = True

    default_vertex_color: str = "#4C78A8"
    subgraph_colors: Sequence[str] = field(default_factory=lambda: [
        "#4C78A8", "#F58518", "#54A24B", "#E45756",
        "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC"
    ])

    show_labels: bool = False
    label_size: float = 9.0

    add_titles: bool = True
    title_prefix: str = "Subgrafo"
    figsize: Optional[Tuple[float, float]] = None


class graphPlotter:
    def __init__(self, style: Optional[PlotStyle] = None):
        self.style = style or PlotStyle()

    # -------- helpers --------
    def _resolve_layout(self, g_ig: ig.Graph, layout: Optional[Union[str, ig.Layout]] = None) -> ig.Layout:
        lay_spec = layout if layout is not None else self.style.layout
        if isinstance(lay_spec, ig.Layout):
            return lay_spec
        return g_ig.layout(lay_spec)

    def _sizes_from_degree(self, g_ig: ig.Graph, size_range: Optional[Tuple[float, float]] = None) -> List[float]:
        vmin, vmax = size_range if size_range is not None else self.style.vertex_size_range
        deg = g_ig.degree()
        if len(deg) == 0:
            return []
        dmin, dmax = min(deg), max(deg)
        if dmin == dmax:
            return [0.5 * (vmin + vmax)] * g_ig.vcount()

        sizes: List[float] = []
        for d in deg:
            t = (d - dmin) / (dmax - dmin)
            if self.style.degree_size_sqrt:
                t = math.sqrt(t)
            sizes.append(vmin + t * (vmax - vmin))
        return sizes

    def _labels_from_mapping(
        self,
        g_ig: ig.Graph,
        new_to_old: Dict[int, int],
        show_labels: Optional[bool] = None,
    ) -> Optional[List[str]]:
        show = self.style.show_labels if show_labels is None else show_labels
        if not show:
            return None
        return [str(new_to_old[v.index]) for v in g_ig.vs]

    def _plot_one(
        self,
        ax,
        g_ig: ig.Graph,
        layout: ig.Layout,
        vertex_color: Union[str, Sequence[str]],
        vertex_sizes: List[float],
        vertex_labels: Optional[List[str]],
        title: Optional[str],
    ):
        ax.axis("off")
        ax.set_aspect("equal")
        ig.plot(
            g_ig,
            target=ax,
            layout=layout,
            bbox=self.style.bbox,
            margin=self.style.margin,
            vertex_color=vertex_color,
            vertex_size=vertex_sizes,
            vertex_label=vertex_labels,
            vertex_label_size=self.style.label_size,
            edge_width=self.style.edge_width,
        )
        if title and self.style.add_titles:
            ax.set_title(title, fontsize=11)


    def plot_full_graph(
        self,
        attGraph: AttributedGraph,
        *,
        layout: Optional[Union[str, ig.Layout]] = "drl",   # "drl" costuma espalhar bem
        show_labels: Optional[bool] = None,
        color_by_label: bool = False,
        only_linked: bool = True,          # remove nós com grau 0
        keep: str = "lcc",                  # "lcc" | "all"
        save_path: Optional[str] = None,
        dpi: Optional[int] = None,
        figsize: Optional[Tuple[float, float]] = None,
        title: str = "Grafo completo (filtrado)",
        bbox: Optional[Tuple[int, int]] = None,
        margin: Optional[int] = None,
    ):
        g_nk = attGraph.graph

        # -----------------------------
        # 1) Filtrar nós sem ligação
        # -----------------------------
        nodes = list(g_nk.iterNodes())

        if only_linked:
            nodes = [u for u in nodes if g_nk.degree(u) > 0]

        if len(nodes) == 0:
            raise ValueError("Após filtrar nós sem ligação (grau 0), não sobrou nenhum nó para plotar.")

        # subgrafo induzido nesses nós
        g_f = nk.graphtools.subgraphFromNodes(g_nk, nodes)

        # -----------------------------
        # 2) Opcional: manter apenas a LCC
        # -----------------------------
        keep = keep.lower().strip()
        if keep not in {"lcc", "all"}:
            raise ValueError("keep deve ser 'lcc' ou 'all'.")

        if keep == "lcc":
            cc = nk.components.ConnectedComponents(g_f)
            cc.run()
            comps = cc.getComponents()
            if len(comps) == 0:
                raise ValueError("Nenhuma componente encontrada no grafo filtrado.")
            lcc_nodes = max(comps, key=len)
            g_f = nk.graphtools.subgraphFromNodes(g_f, lcc_nodes)

        # -----------------------------
        # 3) Converter para igraph (já remapeando)
        # -----------------------------
        g_ig, _, new_to_old = nk_to_igraph_remap(g_f)

        # -----------------------------
        # 4) Layout + ajustes para espalhar melhor
        # -----------------------------
        lay = self._resolve_layout(g_ig, layout=layout)

        # Ajustes visuais recomendados para grafos maiores:
        # - aumentar bbox e margin
        # - diminuir tamanho dos nós
        n = g_ig.vcount()
        if bbox is None:
            # bbox cresce com n (regra simples)
            side = int(900 + min(2500, 15 * math.sqrt(max(n, 1)) * 20))
            bbox = (side, side)
        if margin is None:
            margin = 60

        # sobrescreve temporariamente bbox/margin do style no plot
        old_bbox = self.style.bbox
        old_margin = self.style.margin
        self.style.bbox = bbox
        self.style.margin = margin

        # tamanhos de nós menores para não virar “bolinha” gigante no centro
        old_vs = self.style.vertex_size_range
        self.style.vertex_size_range = (2.0, 8.0)

        sizes = self._sizes_from_degree(g_ig)
        labels = self._labels_from_mapping(g_ig, new_to_old, show_labels=show_labels)

        # cores
        if color_by_label and hasattr(attGraph, "y") and attGraph.y is not None:
            y_cpu = attGraph.y.detach().cpu().numpy()
            palette = list(self.style.subgraph_colors) if self.style.subgraph_colors else [self.style.default_vertex_color]
            colors = []
            for v in g_ig.vs:
                old_id = new_to_old[v.index]
                c = int(y_cpu[old_id])
                colors.append(palette[c % len(palette)])
            vertex_color = colors
        else:
            vertex_color = self.style.default_vertex_color

        # figura
        fig_dpi = dpi if dpi is not None else self.style.dpi
        if figsize is None:
            fig_size = self.style.figsize or (9, 9)
        else:
            fig_size = figsize

        fig, ax = plt.subplots(1, 1, figsize=fig_size, dpi=fig_dpi)
        self._plot_one(
            ax=ax,
            g_ig=g_ig,
            layout=lay,
            vertex_color=vertex_color,
            vertex_sizes=sizes,
            vertex_labels=labels,
            title=title,
        )

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=fig_dpi, bbox_inches="tight")

        # restaura style
        self.style.bbox = old_bbox
        self.style.margin = old_margin
        self.style.vertex_size_range = old_vs

        return fig, ax

    # ----------------------------
    # Compare subgrafo a subgrafo entre vários grafos
    # ----------------------------
    def compare_graphs_subgraphs(
        self,
        graphs: Sequence[AttributedGraph],
        *,
        which: str = "all",                 # "all" | "one" | "subset"
        graph_index: int = 0,               # usado quando which="one"
        graph_indices: Optional[Sequence[int]] = None,  # usado quando which="subset"
        subgraph_ids: Optional[Sequence[int]] = None,   # colunas específicas (classes específicas)
        layout: Optional[Union[str, ig.Layout]] = None,
        colors: Optional[Sequence[str]] = None,         # cores por COLUNA (subgrafo)
        show_labels: Optional[bool] = None,
        save_path: Optional[str] = None,
        dpi: Optional[int] = None,
        figsize: Optional[Tuple[float, float]] = None,
        titles: Optional[Sequence[str]] = None,         # títulos por linha (grafo)
        empty_cell_text: str = "vazio",
    ):
        """
        Compara subgrafos entre grafos:
        - Linhas = grafos
        - Colunas = subgraph_id (classe)

        which:
          - "all": usa todos os grafos na lista
          - "one": usa somente graphs[graph_index]
          - "subset": usa apenas graphs[graph_indices]
        """

        if len(graphs) == 0:
            raise ValueError("Lista 'graphs' vazia.")

        which = which.lower().strip()
        if which not in {"all", "one", "subset"}:
            raise ValueError("which deve ser 'all', 'one' ou 'subset'.")

        # Seleciona quais grafos entram
        if which == "all":
            selected = list(range(len(graphs)))
        elif which == "one":
            if not (0 <= graph_index < len(graphs)):
                raise IndexError(f"graph_index={graph_index} fora do intervalo 0..{len(graphs)-1}")
            selected = [graph_index]
        else:  # subset
            if not graph_indices:
                raise ValueError("Para which='subset', forneça graph_indices=[...].")
            selected = list(graph_indices)
            for gi in selected:
                if not (0 <= gi < len(graphs)):
                    raise IndexError(f"graph_indices contém {gi} fora do intervalo 0..{len(graphs)-1}")

        selected_graphs = [graphs[i] for i in selected]
        print([i.graph.numberOfNodes() for i in selected_graphs])

        # Define quais subgraph_ids (colunas) entram:
        # - se o usuário não passou, pega a união das chaves de subgraphs de todos os grafos selecionados
        if subgraph_ids is None:
            union_ids = set()
            for g in selected_graphs:
                if not hasattr(g, "subgraphs"):
                    raise ValueError("Algum AttributedGraph não tem atributo 'subgraphs'. Rode create_subgraphs().")
                union_ids.update(g.subgraphs.keys())
            col_ids = sorted(union_ids)
        else:
            col_ids = list(subgraph_ids)

        nrows = len(selected_graphs)
        ncols = len(col_ids)
        if ncols == 0:
            raise ValueError("Nenhum subgraph_id para plotar (col_ids está vazio).")

        # Cores por COLUNA (subgrafo)
        palette = list(colors) if colors is not None else list(self.style.subgraph_colors)
        if len(palette) == 0:
            palette = [self.style.default_vertex_color]

        fig_dpi = dpi if dpi is not None else self.style.dpi
        if figsize is None:
            # uma regra simples: ~4.2 por coluna, ~4.2 por linha
            fig_size = self.style.figsize or (4.2 * ncols, 4.2 * nrows)
        else:
            fig_size = figsize

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size, dpi=fig_dpi)
        if nrows == 1 and ncols == 1:
            axes_grid = [[axes]]
        elif nrows == 1:
            axes_grid = [list(axes)]
        elif ncols == 1:
            axes_grid = [[ax] for ax in axes]
        else:
            axes_grid = [list(row) for row in axes]

        # Plot célula por célula: (grafo i, subgrafo j)
        for r, att in enumerate(selected_graphs):
            # título da linha (grafo)
            row_title = None
            if titles is not None and r < len(titles):
                row_title = titles[r]
            else:
                row_title = f"Grafo {selected[r]}"

            for c, sub_id in enumerate(col_ids):
                ax = axes_grid[r][c]

                if not hasattr(att, "subgraphs") or att.subgraphs is None:
                    raise ValueError("Algum AttributedGraph não tem 'subgraphs'. Rode create_subgraphs().")

                if sub_id not in att.subgraphs:
                    # célula vazia: não existe esse subgrafo nesse grafo
                    ax.axis("off")
                    ax.text(
                        0.5, 0.5, empty_cell_text,
                        ha="center", va="center", fontsize=11
                    )
                    # coloca título na coluna mesmo assim, se for primeira linha
                    if self.style.add_titles and r == 0:
                        ax.set_title(f"{self.style.title_prefix} {sub_id}", fontsize=11)
                    continue

                sg_nk = att.subgraphs[sub_id]
                g_ig, _, new_to_old = nk_to_igraph_remap(sg_nk)

                lay = self._resolve_layout(g_ig, layout=layout)
                sizes = self._sizes_from_degree(g_ig)
                labels = self._labels_from_mapping(g_ig, new_to_old, show_labels=show_labels)

                col_color = palette[c % len(palette)]
                # Título: na primeira linha, título da coluna; na primeira coluna, “marca” a linha
                title = None
                if self.style.add_titles and r == 0:
                    title = f"{self.style.title_prefix} {sub_id}"

                self._plot_one(ax, g_ig, lay, col_color, sizes, labels, title)

                # “rótulo” da linha na primeira coluna (fica na lateral esquerda)
                if c == 0:
                    ax.text(
                        -0.02, 0.5, row_title,
                        transform=ax.transAxes,
                        rotation=90,
                        va="center", ha="right",
                        fontsize=11
                    )

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=fig_dpi, bbox_inches="tight")

        return fig, axes

    def plot_degree_distributions(
        self,
        attGraph: AttributedGraph,
        *,
        include_full: bool = True,
        subgraph_ids: Optional[Sequence[int]] = None,
        which: str = "all",          # "all" | "one"
        subgraph_id: Optional[int] = None,  # usado quando which="one"
        pmf: bool = True,            # True: P(k); False: contagem
        bins: Optional[int] = None,  # se None, usa bins automáticos por grau inteiro
        loglog: bool = False,
        density: Optional[bool] = None,  # se None, segue pmf; se bool, força
        save_path: Optional[str] = None,
        dpi: int = 300,
        figsize: Optional[Tuple[float, float]] = None,
        title_full: str = "Grafo completo",
        title_prefix: str = "Subgrafo",
        grid: Optional[Tuple[int, int]] = None,
        linewidth: float = 1.8,
        alpha: float = 0.9,
    ):
        """
        Plota distribuição de grau (grafo completo + subgrafos) usando NetworKit para graus.

        - pmf=True: plota P(k) (probabilidade)
        - pmf=False: plota contagem de nós com grau k

        loglog=True plota em escala log-log (muito útil p/ caudas).
        """

        if density is None:
            density = pmf

        # --------- Seleciona quais subgrafos vão entrar ---------
        if not hasattr(attGraph, "subgraphs") or attGraph.subgraphs is None:
            raise ValueError("attGraph.subgraphs não existe. Rode create_subgraphs() antes.")

        all_sub_ids = sorted(attGraph.subgraphs.keys())

        if which.lower().strip() == "one":
            if subgraph_id is None:
                raise ValueError("which='one' exige subgraph_id=...")
            if subgraph_id not in attGraph.subgraphs:
                raise ValueError(f"subgraph_id={subgraph_id} não existe em attGraph.subgraphs.")
            chosen_sub_ids = [subgraph_id]
        else:
            if subgraph_ids is None:
                chosen_sub_ids = all_sub_ids
            else:
                chosen_sub_ids = [sid for sid in subgraph_ids if sid in attGraph.subgraphs]

        # Lista de grafos para plotar
        items: List[Tuple[str, nk.Graph]] = []
        if include_full:
            items.append((title_full, attGraph.graph))

        for sid in chosen_sub_ids:
            items.append((f"{title_prefix} {sid}", attGraph.subgraphs[sid]))

        if len(items) == 0:
            raise ValueError("Nada para plotar: include_full=False e nenhum subgrafo selecionado.")

        # --------- Funções auxiliares ---------
        def degree_array(g: nk.Graph) -> np.ndarray:
            # graus via NetworKit
            return np.array([g.degree(u) for u in g.iterNodes()], dtype=int)

        def degree_hist(deg: np.ndarray):
            """
            Retorna (k_vals, y_vals) onde:
            - k_vals = graus
            - y_vals = contagem ou pmf
            """
            if deg.size == 0:
                return np.array([], dtype=int), np.array([], dtype=float)

            kmax = int(deg.max())
            counts = np.bincount(deg, minlength=kmax + 1)

            k_vals = np.arange(len(counts), dtype=int)
            y_vals = counts.astype(float)

            if density:  # normaliza para probabilidade
                total = y_vals.sum()
                if total > 0:
                    y_vals = y_vals / total

            # remove k onde y = 0 (melhor para loglog)
            mask = y_vals > 0
            return k_vals[mask], y_vals[mask]

        # --------- Layout de subplots ---------
        n_plots = len(items)
        if grid is None:
            ncols = math.ceil(math.sqrt(n_plots))
            nrows = math.ceil(n_plots / ncols)
        else:
            nrows, ncols = grid

        if figsize is None:
            figsize = (4.8 * ncols, 4.2 * nrows)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
        if isinstance(axes, plt.Axes):
            axes_list = [axes]
        else:
            axes_list = list(axes.flatten())

        # desliga eixos extras
        for ax in axes_list[n_plots:]:
            ax.axis("off")

        # --------- Plota cada distribuição ---------
        for i, (title, g_nk) in enumerate(items):
            ax = axes_list[i]
            deg = degree_array(g_nk)
            k, y = degree_hist(deg)

            # caso grafo vazio
            if k.size == 0:
                ax.axis("off")
                ax.set_title(title)
                ax.text(0.5, 0.5, "vazio", ha="center", va="center")
                continue

            # Desenho como “linha com marcadores” (bom para comparar caudas)
            ax.plot(k, y, marker="o", linewidth=linewidth, alpha=alpha)

            ax.set_title(title)
            ax.set_xlabel("grau k")
            ax.set_ylabel("P(k)" if density else "contagem")

            if loglog:
                ax.set_xscale("log")
                ax.set_yscale("log")

            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return fig, axes
    

    def compare_graph_metrics(
        self,
        graphs: Sequence["AttributedGraph"],
        *,
        names: Optional[Sequence[str]] = None,
        reference_index: int = 0,
        compute_modularity: bool = True,
        community_iters: int = 10,
        compute_ks_degree: bool = True,
    ) -> pd.DataFrame:
        """
        Compara N grafos (AttributedGraph) e retorna um DataFrame com métricas.

        graphs: lista de AttributedGraph
        names: nomes opcionais (mesmo tamanho de graphs). Se None: "G0", "G1", ...
        reference_index: índice do grafo de referência para KS da NDD (default 0)
        compute_modularity: roda PLM e computa modularidade
        community_iters: iterações do PLM (tradeoff qualidade/tempo)
        compute_ks_degree: computa KS da distribuição de grau vs. referência (PMF acumulada)
        """

        if len(graphs) == 0:
            raise ValueError("graphs está vazio.")

        if names is None:
            names = [f"G{i}" for i in range(len(graphs))]
        if len(names) != len(graphs):
            raise ValueError("names deve ter o mesmo tamanho que graphs.")

        if not (0 <= reference_index < len(graphs)):
            raise IndexError("reference_index fora do intervalo.")

        # ---------- helpers ----------
        def density(n: int, m: int) -> float:
            if n <= 1:
                return 0.0
            return (2.0 * m) / (n * (n - 1))

        def degree_array(g: nk.Graph) -> np.ndarray:
            return np.fromiter((g.degree(u) for u in g.iterNodes()), dtype=np.int64, count=g.numberOfNodes())

        def degree_entropy(deg: np.ndarray) -> float:
            # entropia de P(k)
            if deg.size == 0:
                return 0.0
            counts = np.bincount(deg)
            p = counts[counts > 0].astype(float)
            p /= p.sum()
            return float(-(p * np.log(p)).sum())

        def homophily_fraction(g: nk.Graph, y: torch.Tensor) -> Tuple[float, float]:
            # fração de arestas intra-classe vs inter-classe
            if y is None:
                return (float("nan"), float("nan"))
            if y.numel() == 0 or g.numberOfEdges() == 0:
                return (0.0, 0.0)

            same = 0
            total = 0
            # y pode estar em GPU; traga pra CPU p/ indexação rápida
            y_cpu = y.detach().cpu()
            for u, v in g.iterEdges():
                total += 1
                if y_cpu[u].item() == y_cpu[v].item():
                    same += 1
            if total == 0:
                return (0.0, 0.0)
            homo = same / total
            hetero = 1.0 - homo
            return (homo, hetero)

        def largest_cc_fraction(g: nk.Graph) -> Tuple[int, float]:
            # n_components e fração LCC
            if g.numberOfNodes() == 0:
                return 0, 0.0
            cc = nk.components.ConnectedComponents(g)
            cc.run()
            comps = cc.getComponents()
            n_comp = len(comps)
            lcc = max((len(c) for c in comps), default=0)
            return n_comp, (lcc / g.numberOfNodes())

        def avg_local_clustering(g: nk.Graph) -> float:
            if g.numberOfNodes() == 0:
                return 0.0
            lcc = nk.centrality.LocalClusteringCoefficient(g)
            lcc.run()
            scores = lcc.scores()
            return float(np.mean(scores)) if len(scores) else 0.0

        def degree_assortativity(g: nk.Graph) -> float:
            """
            Assortatividade por grau (Newman) para grafo não-direcionado.
            Implementação manual para evitar dependência de API específica do NetworKit.
            """
            m = g.numberOfEdges()
            if m == 0:
                return float("nan")

            deg = degree_array(g).astype(float)

            # Para cada aresta, pegue (j,k) = graus dos endpoints
            js = []
            ks = []
            for u, v in g.iterEdges():
                js.append(deg[u])
                ks.append(deg[v])

            j = np.array(js, dtype=float)
            k = np.array(ks, dtype=float)

            # Newman (2002): r = [E[jk] - E[(j+k)/2]^2] / [E[(j^2+k^2)/2] - E[(j+k)/2]^2]
            ejk = np.mean(j * k)
            e1 = np.mean((j + k) / 2.0)
            e2 = np.mean((j**2 + k**2) / 2.0)

            num = ejk - e1**2
            den = e2 - e1**2
            if den == 0:
                return float("nan")
            return float(num / den)

        def community_metrics_plm(g: nk.Graph) -> Dict[str, Any]:
            if g.numberOfNodes() == 0 or g.numberOfEdges() == 0:
                return {
                    "modularity": float("nan"),
                    "n_communities": 0,
                    "community_size_mean": 0.0,
                    "community_size_std": 0.0,
                    "community_size_entropy": 0.0,
                }

            # PLM: bom custo/benefício
            plm = nk.community.PLM(g, refine=True)
            plm.run()
            part = plm.getPartition()

            # modularidade
            mod = nk.community.Modularity().getQuality(part, g)

            # tamanhos das comunidades
            sizes = part.subsetSizeMap()  # dict {comm_id: size}
            sz = np.array(list(sizes.values()), dtype=float)
            if sz.size == 0:
                return {
                    "modularity": float(mod),
                    "n_communities": 0,
                    "community_size_mean": 0.0,
                    "community_size_std": 0.0,
                    "community_size_entropy": 0.0,
                }

            p = sz / sz.sum()
            ent = float(-(p * np.log(p)).sum()) if p.size else 0.0

            return {
                "modularity": float(mod),
                "n_communities": int(sz.size),
                "community_size_mean": float(sz.mean()),
                "community_size_std": float(sz.std(ddof=0)),
                "community_size_entropy": ent,
            }

        def cdf_from_degree_pmf(deg: np.ndarray) -> np.ndarray:
            if deg.size == 0:
                return np.array([1.0], dtype=float)
            counts = np.bincount(deg)
            pmf = counts.astype(float)
            s = pmf.sum()
            if s > 0:
                pmf /= s
            return np.cumsum(pmf)

        def ks_distance_degree(deg_a: np.ndarray, deg_b: np.ndarray) -> float:
            cdf_a = cdf_from_degree_pmf(deg_a)
            cdf_b = cdf_from_degree_pmf(deg_b)
            L = max(len(cdf_a), len(cdf_b))
            if len(cdf_a) < L:
                cdf_a = np.pad(cdf_a, (0, L - len(cdf_a)), mode="edge")
            if len(cdf_b) < L:
                cdf_b = np.pad(cdf_b, (0, L - len(cdf_b)), mode="edge")
            return float(np.max(np.abs(cdf_a - cdf_b)))

        # ---------- referência para KS ----------
        ref_deg = degree_array(graphs[reference_index].graph) if compute_ks_degree else None

        rows: List[Dict[str, Any]] = []

        for idx, (name, ag) in enumerate(zip(names, graphs)):
            g = ag.graph
            n = g.numberOfNodes()
            m = g.numberOfEdges()

            deg = degree_array(g) if n > 0 else np.array([], dtype=np.int64)

            avg_deg = float((2.0 * m / n) if n > 0 else 0.0)
            var_deg = float(np.var(deg)) if deg.size else 0.0
            ent_deg = degree_entropy(deg)

            n_comp, lcc_frac = largest_cc_fraction(g)
            clust = avg_local_clustering(g)
            assort = degree_assortativity(g)

            # homofilia por rótulo (se tiver)
            if ag.y is not None and isinstance(ag.y, torch.Tensor):
                homo, hetero = homophily_fraction(g, ag.y)
            else:
                homo, hetero = (float("nan"), float("nan"))

            # modularidade e comunidades
            comm = {}
            if compute_modularity:
                comm = community_metrics_plm(g)
            else:
                comm = {
                    "modularity": float("nan"),
                    "n_communities": 0,
                    "community_size_mean": float("nan"),
                    "community_size_std": float("nan"),
                    "community_size_entropy": float("nan"),
                }

            # KS vs referência
            ks_deg = float("nan")
            if compute_ks_degree and ref_deg is not None:
                ks_deg = ks_distance_degree(deg, ref_deg)

            rows.append({
                "name": name,
                "n_nodes": int(n),
                "n_edges": int(m),
                "density": float(density(n, m)),
                "avg_degree": avg_deg,
                "var_degree": var_deg,
                "degree_entropy": ent_deg,
                "n_components": int(n_comp),
                "lcc_fraction": float(lcc_frac),
                "avg_local_clustering": float(clust),
                "degree_assortativity": float(assort),
                "label_homophily": float(homo),
                "label_heterophily": float(hetero),
                "ks_degree_to_ref": float(ks_deg),
                **comm,
            })

        df = pd.DataFrame(rows)

        # Coloca "name" como primeira coluna e ordena colunas principais
        col_order = [
            "name",
            "n_nodes", "n_edges", "density",
            "avg_degree", "var_degree", "degree_entropy",
            "n_components", "lcc_fraction",
            "avg_local_clustering", "degree_assortativity",
            "modularity", "n_communities", "community_size_mean", "community_size_std", "community_size_entropy",
            "label_homophily", "label_heterophily",
            "ks_degree_to_ref",
        ]
        # mantém extras caso algo apareça no futuro
        df = df[[c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order]]

        return df
    
    def plot_full_graph_with_subgraph_layout(
    self,
    attGraph: AttributedGraph,
    *,
    subgraph_ids: Optional[Sequence[int]] = None,
    layout: Optional[Union[str, ig.Layout]] = "fr",   # layout interno de cada subgrafo
    grid: Optional[Tuple[int, int]] = None,
    block_gap: float = 3.0,
    normalize_each_block: bool = True,
    color_by_label: bool = True,
    show_labels: Optional[bool] = False,

    # ---- NOVOS CONTROLES VISUAIS ----
    interclass_edge_color: str = "#D9D9D9",  # bem claro
    interclass_edge_width: float = 0.35,
    intraclass_edge_width: float = 0.9,
    vertex_size_range: Tuple[float, float] = (10.0, 36.0),  # vértices maiores
    # ---------------------------------

    save_path: Optional[str] = None,
    dpi: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Grafo completo (layout reaproveitado dos subgrafos)",
    ):
        if not hasattr(attGraph, "subgraphs") or attGraph.subgraphs is None:
            raise ValueError("attGraph.subgraphs não existe. Rode create_subgraphs() antes.")

        # quais subgrafos entram
        all_ids = sorted(attGraph.subgraphs.keys())
        if subgraph_ids is None:
            sub_ids = all_ids
        else:
            sub_ids = [sid for sid in subgraph_ids if sid in attGraph.subgraphs]
        if len(sub_ids) == 0:
            raise ValueError("Nenhum subgrafo selecionado para compor o layout.")

        # ---------- converter grafo completo ----------
        g_full_nk = attGraph.graph
        g_full_ig, old_to_new_full, new_to_old_full = nk_to_igraph_remap(g_full_nk)
        n_full = g_full_ig.vcount()

        # ---------- define grid ----------
        k = len(sub_ids)
        if grid is None:
            ncols = math.ceil(math.sqrt(k))
            nrows = math.ceil(k / ncols)
        else:
            nrows, ncols = grid

        coords_full = np.zeros((n_full, 2), dtype=float)
        positioned = np.zeros(n_full, dtype=bool)

        palette = list(self.style.subgraph_colors) if self.style.subgraph_colors else [self.style.default_vertex_color]

        # ---------- layout por bloco ----------
        block_index = 0
        for sid in sub_ids:
            sg_nk = attGraph.subgraphs[sid]
            sg_ig, _, new_to_old_sg = nk_to_igraph_remap(sg_nk)

            lay_sg = self._resolve_layout(sg_ig, layout=layout)
            xy = np.array(lay_sg.coords, dtype=float)

            if normalize_each_block and xy.shape[0] > 0:
                xy = xy - xy.mean(axis=0, keepdims=True)
                max_abs = np.max(np.abs(xy))
                if max_abs > 0:
                    xy = xy / max_abs  # ~[-1,1]

            r = block_index // ncols
            c = block_index % ncols
            offset = np.array([c * block_gap, -r * block_gap], dtype=float)
            xy = xy + offset

            for v in sg_ig.vs:
                old_id = new_to_old_sg[v.index]
                new_id_full = old_to_new_full[old_id]
                coords_full[new_id_full] = xy[v.index]
                positioned[new_id_full] = True

            block_index += 1

        # nós não posicionados (se existirem)
        if not positioned.all():
            last_r = (block_index - 1) // ncols
            last_c = (block_index - 1) % ncols
            base = np.array([last_c * block_gap, -last_r * block_gap], dtype=float) + np.array([block_gap, 0.0])
            missing = np.where(~positioned)[0]
            for t, vid in enumerate(missing):
                coords_full[vid] = base + np.array([0.25 * (t % 10), -0.25 * (t // 10)])

        layout_full = ig.Layout(coords_full.tolist())

        # ---------- cores por nó ----------
        if color_by_label and hasattr(attGraph, "y") and attGraph.y is not None:
            y_cpu = attGraph.y.detach().cpu().numpy()
            vcolors = []
            for v in g_full_ig.vs:
                old_id = new_to_old_full[v.index]
                c = int(y_cpu[old_id])
                vcolors.append(palette[c % len(palette)])
        else:
            vcolors = self.style.default_vertex_color
            y_cpu = None  # para não usar nas arestas

        # ---------- cores e larguras por aresta ----------
        # intraclasse = cor da classe; interclasse = cinza claro
        edge_colors = []
        edge_widths = []

        if y_cpu is not None:
            for e in g_full_ig.es:
                u_new, v_new = e.tuple
                u_old = new_to_old_full[u_new]
                v_old = new_to_old_full[v_new]

                cu = int(y_cpu[u_old])
                cv = int(y_cpu[v_old])

                if cu == cv:
                    edge_colors.append(palette[cu % len(palette)])
                    edge_widths.append(intraclass_edge_width)
                else:
                    edge_colors.append(interclass_edge_color)
                    edge_widths.append(interclass_edge_width)
        else:
            # sem y: tudo igual
            edge_colors = ["#C0C0C0"] * g_full_ig.ecount()
            edge_widths = [interclass_edge_width] * g_full_ig.ecount()

        # ---------- tamanhos dos nós (maiores) ----------
        old_vs = self.style.vertex_size_range
        self.style.vertex_size_range = vertex_size_range
        sizes = self._sizes_from_degree(g_full_ig)
        self.style.vertex_size_range = old_vs

        labels = self._labels_from_mapping(g_full_ig, new_to_old_full, show_labels=show_labels)

        # ---------- plot ----------
        fig_dpi = dpi if dpi is not None else self.style.dpi
        if figsize is None:
            figsize = (12, 12)

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=fig_dpi)
        ax.axis("off")
        ax.set_aspect("equal")

        ig.plot(
            g_full_ig,
            target=ax,
            layout=layout_full,
            bbox=self.style.bbox,
            margin=self.style.margin,
            vertex_color=vcolors,
            vertex_size=sizes,
            vertex_label=labels,
            vertex_label_size=self.style.label_size,
            edge_color=edge_colors,
            edge_width=edge_widths,
        )

        if self.style.add_titles:
            ax.set_title(title, fontsize=11)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=fig_dpi, bbox_inches="tight")

        return fig, ax