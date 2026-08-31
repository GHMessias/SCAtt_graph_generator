from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkit as nk
import networkx as nx
import numpy as np
import torch
from matplotlib.lines import Line2D

try:
    from netgraph import Graph
except ImportError as exc:
    Graph = None
    _NETGRAPH_IMPORT_ERROR = exc
else:
    _NETGRAPH_IMPORT_ERROR = None

from core.attributed_graph import AttributedGraph


@dataclass
class PublicationPlotStyle:
    """Defaults chosen for clean, publication-oriented graph figures."""

    figsize: Tuple[float, float] = (6.4, 5.6)
    dpi: int = 300
    seed: int = 2026
    node_layout: str = "community"
    edge_layout: str = "bundled"
    edge_bundle_k: int = 2000
    community_distance: float = 1.0
    node_layout_pad_by: float = 0.05
    node_size: Optional[float] = None
    node_edge_width: float = 0.0
    edge_width: float = 0.45
    edge_alpha: float = 0.14
    class_edge_alpha: Optional[float] = None
    interclass_edge_alpha: Optional[float] = None
    font_size: float = 9.0
    title_size: float = 11.0
    show_legend: bool = True
    legend_title: str = "Class"
    background: str = "white"
    edge_color: str = "#9AA1A9"
    color_intraclass_edges: bool = True
    interclass_edge_color: str = "#000000"
    node_edge_color: str = "white"
    palette: Sequence[str] = field(
        default_factory=lambda: (
            "#0072B2",
            "#D55E00",
            "#009E73",
            "#CC79A7",
            "#E69F00",
            "#56B4E9",
            "#F0E442",
            "#6F4E7C",
            "#4D4D4D",
        )
    )


def _as_label_array(y: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if y is None:
        return None
    if isinstance(y, torch.Tensor):
        return y.detach().cpu().numpy()
    return np.asarray(y)


def _networkit_to_networkx(
    g: nk.Graph,
    nodes: Optional[Iterable[int]] = None,
) -> nx.Graph:
    selected = None if nodes is None else {int(node) for node in nodes}
    graph_cls = nx.DiGraph if g.isDirected() else nx.Graph
    graph_nx = graph_cls()

    for node in g.iterNodes():
        node = int(node)
        if selected is None or node in selected:
            graph_nx.add_node(node)

    for u, v in g.iterEdges():
        u = int(u)
        v = int(v)
        if u in graph_nx and v in graph_nx:
            graph_nx.add_edge(u, v)

    return graph_nx


class graphPlotter:
    def __init__(self, style: Optional[PublicationPlotStyle] = None):
        self.style = style or PublicationPlotStyle()

    def _class_nodes(self, graph: AttributedGraph, class_id: int) -> list[int]:
        labels = _as_label_array(graph.y)
        if labels is None:
            raise ValueError("Para plotar uma classe especifica, o grafo precisa ter labels em graph.y.")
        return [int(node) for node in graph.graph.iterNodes() if int(labels[int(node)]) == int(class_id)]

    def _class_colors(self, labels: Optional[np.ndarray]) -> Dict[int, str]:
        if labels is None:
            return {}

        class_values = [int(value) for value in sorted(np.unique(labels[labels >= 0]))]
        return {
            class_value: self.style.palette[idx % len(self.style.palette)]
            for idx, class_value in enumerate(class_values)
        }

    def _node_to_community(self, graph_nx: nx.Graph, labels: Optional[np.ndarray]) -> Dict[int, int]:
        if labels is None:
            return {int(node): 0 for node in graph_nx.nodes}
        return {int(node): int(labels[int(node)]) for node in graph_nx.nodes}

    def _node_colors(
        self,
        graph_nx: nx.Graph,
        labels: Optional[np.ndarray],
        color_map: Dict[int, str],
        class_id: Optional[int],
    ) -> Dict[int, str]:
        if labels is None:
            return {int(node): self.style.palette[0] for node in graph_nx.nodes}

        colors = {}
        for node in graph_nx.nodes:
            label = int(class_id) if class_id is not None else int(labels[int(node)])
            colors[int(node)] = color_map.get(label, "#8A8F98")
        return colors

    def _edge_colors(
        self,
        graph_nx: nx.Graph,
        labels: Optional[np.ndarray],
        color_map: Dict[int, str],
        *,
        class_only: bool,
        show_interclass_edges: bool,
    ) -> Dict[tuple[int, int], str]:
        colors = {}

        for u, v in graph_nx.edges:
            edge = (int(u), int(v))
            if labels is None:
                colors[edge] = self.style.edge_color
                continue

            u_label = int(labels[int(u)])
            v_label = int(labels[int(v)])
            same_class = u_label == v_label

            if same_class:
                color = color_map.get(u_label, self.style.edge_color)
                if not self.style.color_intraclass_edges:
                    color = self.style.edge_color
                colors[edge] = color
            elif show_interclass_edges:
                colors[edge] = self.style.interclass_edge_color
            else:
                colors[edge] = self.style.interclass_edge_color

        return colors

    def _edge_alphas(
        self,
        graph_nx: nx.Graph,
        labels: Optional[np.ndarray],
        *,
        edge_alpha: float,
        show_interclass_edges: bool,
    ) -> Dict[tuple[int, int], float]:
        class_edge_alpha = self.style.class_edge_alpha if self.style.class_edge_alpha is not None else edge_alpha
        interclass_edge_alpha = (
            self.style.interclass_edge_alpha if self.style.interclass_edge_alpha is not None else edge_alpha
        )
        alphas = {}

        for u, v in graph_nx.edges:
            edge = (int(u), int(v))
            if labels is None:
                alphas[edge] = edge_alpha
                continue

            same_class = int(labels[int(u)]) == int(labels[int(v)])
            if same_class:
                alphas[edge] = class_edge_alpha
            elif show_interclass_edges:
                alphas[edge] = interclass_edge_alpha
            else:
                alphas[edge] = edge_alpha

        return alphas

    def _remove_interclass_edges(self, graph_nx: nx.Graph, labels: Optional[np.ndarray]) -> nx.Graph:
        if labels is None:
            return graph_nx

        filtered = graph_nx.copy()
        interclass_edges = [
            (u, v)
            for u, v in filtered.edges
            if int(labels[int(u)]) != int(labels[int(v)])
        ]
        filtered.remove_edges_from(interclass_edges)
        return filtered

    def _fit_positions_to_unit_box(
        self,
        positions: Dict[int, np.ndarray],
    ) -> Dict[int, np.ndarray]:
        if not positions:
            return positions

        nodes = list(positions)
        xy = np.vstack([positions[node] for node in nodes]).astype(float)
        xy_min = xy.min(axis=0)
        xy_max = xy.max(axis=0)
        span = xy_max - xy_min
        span[span == 0.0] = 1.0

        pad = self.style.node_layout_pad_by
        xy = (xy - xy_min) / span
        xy = pad + xy * (1.0 - 2.0 * pad)
        return {node: xy[idx] for idx, node in enumerate(nodes)}

    def _add_missing_positions(
        self,
        graph_nx: nx.Graph,
        positions: Dict[int, np.ndarray],
        node_to_community: Dict[int, int],
    ) -> Dict[int, np.ndarray]:
        missing_nodes = [int(node) for node in graph_nx.nodes if int(node) not in positions]
        if not missing_nodes:
            return positions

        community_to_positions: Dict[int, list[np.ndarray]] = {}
        for node, position in positions.items():
            community_to_positions.setdefault(node_to_community[node], []).append(position)

        global_center = np.mean(list(positions.values()), axis=0) if positions else np.array([0.5, 0.5])
        rng = np.random.default_rng(self.style.seed)
        for node in missing_nodes:
            community = node_to_community[node]
            if community in community_to_positions:
                center = np.mean(community_to_positions[community], axis=0)
            else:
                center = global_center
            positions[node] = center + 0.015 * rng.normal(size=2)

        return positions

    def _complete_node_positions(
        self,
        graph_nx: nx.Graph,
        positions: Dict[int, np.ndarray],
        labels: Optional[np.ndarray],
    ) -> Dict[int, np.ndarray]:
        missing_nodes = [int(node) for node in graph_nx.nodes if int(node) not in positions]
        if not missing_nodes:
            return positions

        rng = np.random.default_rng(self.style.seed)
        if labels is None or not positions:
            center = np.mean(list(positions.values()), axis=0) if positions else np.array([0.5, 0.5])
            for node in missing_nodes:
                positions[node] = center + 0.015 * rng.normal(size=2)
            return positions

        community_to_positions: Dict[int, list[np.ndarray]] = {}
        for node, position in positions.items():
            community_to_positions.setdefault(int(labels[int(node)]), []).append(position)

        global_center = np.mean(list(positions.values()), axis=0)
        for node in missing_nodes:
            community = int(labels[int(node)])
            if community in community_to_positions:
                center = np.mean(community_to_positions[community], axis=0)
            else:
                center = global_center
            positions[node] = center + 0.015 * rng.normal(size=2)

        return positions

    def _spaced_community_layout(
        self,
        graph_nx: nx.Graph,
        labels: Optional[np.ndarray],
        community_distance: float,
    ) -> Optional[Dict[int, np.ndarray]]:
        if labels is None:
            return None

        node_to_community = self._node_to_community(graph_nx, labels)
        communities = sorted(set(node_to_community.values()))
        if len(communities) < 2:
            return None

        community_to_nodes: Dict[int, list[int]] = {}
        for node, community in node_to_community.items():
            community_to_nodes.setdefault(community, []).append(node)

        center_radius = 0.34 * community_distance
        local_scale = 0.11 / max(community_distance, 0.25)
        community_centers = {}
        for idx, community in enumerate(communities):
            angle = 2.0 * np.pi * idx / len(communities)
            community_centers[community] = np.array(
                [
                    0.5 + center_radius * np.cos(angle),
                    0.5 + center_radius * np.sin(angle),
                ]
            )

        spaced_positions = {}
        rng = np.random.default_rng(self.style.seed)
        for community in communities:
            nodes = sorted(community_to_nodes[community])
            center = community_centers[community]
            if len(nodes) == 1:
                spaced_positions[nodes[0]] = center
                continue

            subgraph = graph_nx.subgraph(nodes)
            try:
                local_positions = nx.spring_layout(
                    subgraph,
                    seed=self.style.seed + int(community),
                    iterations=80,
                )
            except ValueError:
                local_positions = {}

            if len(local_positions) != len(nodes):
                angles = np.linspace(0.0, 2.0 * np.pi, len(nodes), endpoint=False)
                rng.shuffle(angles)
                local_positions = {
                    node: np.array([np.cos(angle), np.sin(angle)])
                    for node, angle in zip(nodes, angles)
                }

            local_xy = np.vstack([local_positions[node] for node in nodes]).astype(float)
            local_xy -= local_xy.mean(axis=0)
            max_abs = np.abs(local_xy).max()
            if max_abs > 0.0:
                local_xy /= max_abs

            scale = local_scale * np.sqrt(len(nodes) / max(len(graph_nx), 1))
            scale = max(scale, 0.035)
            for idx, node in enumerate(nodes):
                spaced_positions[node] = center + scale * local_xy[idx]

        return self._fit_positions_to_unit_box(spaced_positions)

    def _normalize_node_layout(self, layout: str) -> str:
        layout = layout.lower().strip()
        aliases = {
            "fr": "spring",
            "force": "spring",
            "fruchterman_reingold": "spring",
            "circle": "circular",
        }
        return aliases.get(layout, layout)

    def _normalize_edge_layout(self, layout: str) -> str:
        layout = layout.lower().strip()
        aliases = {
            "bundle": "bundled",
            "curves": "curved",
        }
        return aliases.get(layout, layout)

    def _graph_kwargs(
        self,
        graph_nx: nx.Graph,
        labels: Optional[np.ndarray],
        *,
        node_layout: str,
        edge_layout: str,
        class_id: Optional[int],
        show_interclass_edges: bool,
        edge_alpha: float,
        community_distance: float,
        node_positions: Optional[Dict[int, Sequence[float]]],
        ax,
    ) -> Dict[str, Any]:
        color_map = self._class_colors(labels)
        active_node_layout: str | Dict[int, np.ndarray] = node_layout
        if node_positions is not None:
            active_node_layout = {
                int(node): np.asarray(position, dtype=float)
                for node, position in node_positions.items()
                if int(node) in graph_nx
            }
            active_node_layout = self._complete_node_positions(graph_nx, active_node_layout, labels)
        elif node_layout == "community":
            spaced_layout = self._spaced_community_layout(graph_nx, labels, community_distance)
            if spaced_layout is not None:
                active_node_layout = spaced_layout

        kwargs: Dict[str, Any] = {
            "node_color": self._node_colors(graph_nx, labels, color_map, class_id),
            "node_edge_width": self.style.node_edge_width,
            "edge_color": self._edge_colors(
                graph_nx,
                labels,
                color_map,
                class_only=class_id is not None,
                show_interclass_edges=show_interclass_edges,
            ),
            "edge_width": self.style.edge_width,
            "edge_alpha": self._edge_alphas(
                graph_nx,
                labels,
                edge_alpha=edge_alpha,
                show_interclass_edges=show_interclass_edges,
            ),
            "node_layout": active_node_layout,
            "edge_layout": edge_layout,
            "ax": ax,
        }

        if self.style.node_size is not None:
            kwargs["node_size"] = self.style.node_size

        if node_layout == "community" and active_node_layout == node_layout:
            kwargs["node_layout_kwargs"] = {
                "node_to_community": self._node_to_community(graph_nx, labels),
                "pad_by": self.style.node_layout_pad_by,
            }

        if edge_layout == "bundled":
            kwargs["edge_layout_kwargs"] = {
                "k": self.style.edge_bundle_k,
            }

        return kwargs

    def get_node_positions(
        self,
        graph: AttributedGraph,
        *,
        class_id: Optional[int] = None,
        layout: Optional[str] = None,
        node_layout: Optional[str] = None,
        community_distance: Optional[float] = None,
    ) -> Dict[int, np.ndarray]:
        """Compute reusable node positions for repeated plots of comparable graphs."""
        if not isinstance(graph, AttributedGraph):
            raise TypeError("get_node_positions espera um objeto AttributedGraph.")

        labels = _as_label_array(graph.y)
        selected_nodes = None if class_id is None else self._class_nodes(graph, class_id)
        graph_nx = _networkit_to_networkx(graph.graph, selected_nodes)
        if graph_nx.number_of_nodes() == 0:
            raise ValueError(f"Nenhum no encontrado para class_id={class_id}.")

        active_node_layout = self._normalize_node_layout(
            node_layout or layout or ("spring" if class_id is not None else self.style.node_layout)
        )
        effective_community_distance = (
            community_distance if community_distance is not None else self.style.community_distance
        )

        if active_node_layout == "community":
            positions = self._spaced_community_layout(graph_nx, labels, effective_community_distance)
            if positions is not None:
                return positions

        if active_node_layout in {"spring", "fr"}:
            return {
                int(node): np.asarray(position, dtype=float)
                for node, position in nx.spring_layout(
                    graph_nx,
                    seed=self.style.seed,
                    iterations=100,
                ).items()
            }

        if active_node_layout == "circular":
            return {
                int(node): np.asarray(position, dtype=float)
                for node, position in nx.circular_layout(graph_nx).items()
            }

        if active_node_layout == "spectral":
            return {
                int(node): np.asarray(position, dtype=float)
                for node, position in nx.spectral_layout(graph_nx).items()
            }

        raise ValueError("layout deve ser 'community', 'spring', 'circular' ou 'spectral'.")

    def _finish_axes(self, ax, title: Optional[str]) -> None:
        ax.set_facecolor(self.style.background)
        ax.set_aspect("equal")
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=self.style.title_size, pad=8)

    def _add_legend(self, ax, color_map: Dict[int, str], labels: Optional[np.ndarray], class_id: Optional[int]) -> None:
        if not self.style.show_legend or labels is None:
            return

        class_values = [int(class_id)] if class_id is not None else sorted(color_map)
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color_map[value],
                markeredgecolor=self.style.node_edge_color,
                markersize=6,
                label=str(value),
            )
            for value in class_values
            if value in color_map
        ]
        if not handles:
            return

        ax.legend(
            handles=handles,
            title=self.style.legend_title,
            frameon=False,
            fontsize=self.style.font_size,
            title_fontsize=self.style.font_size,
            loc="upper right",
            bbox_to_anchor=(1.02, 1.02),
        )

    def plot_scatt_graph(
        self,
        graph: AttributedGraph,
        *,
        class_id: Optional[int] = None,
        layout: Optional[str] = None,
        node_layout: Optional[str] = None,
        edge_layout: Optional[str] = None,
        title: Optional[str] = None,
        show_interclass_edges: bool = True,
        edge_alpha: Optional[float] = None,
        community_distance: Optional[float] = None,
        node_positions: Optional[Dict[int, Sequence[float]]] = None,
        ax=None,
        save_path: Optional[str | Path] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
    ):
        """
        Plot a complete SCAtt graph or an induced subgraph for one class using netgraph.

        ``layout`` is kept as a short alias for ``node_layout``. The default full
        graph view uses netgraph's community node layout and bundled edge layout.
        """
        if not isinstance(graph, AttributedGraph):
            raise TypeError("plot_scatt_graph espera um objeto AttributedGraph.")
        if Graph is None:
            raise ImportError("O plot agora usa netgraph. Instale com `pip install netgraph`.") from _NETGRAPH_IMPORT_ERROR

        labels = _as_label_array(graph.y)
        selected_nodes = None if class_id is None else self._class_nodes(graph, class_id)
        graph_nx = _networkit_to_networkx(graph.graph, selected_nodes)
        if class_id is None and not show_interclass_edges:
            graph_nx = self._remove_interclass_edges(graph_nx, labels)

        if graph_nx.number_of_nodes() == 0:
            raise ValueError(f"Nenhum no encontrado para class_id={class_id}.")

        active_node_layout = self._normalize_node_layout(
            node_layout or layout or ("spring" if class_id is not None else self.style.node_layout)
        )
        active_edge_layout = self._normalize_edge_layout(
            edge_layout or ("straight" if class_id is not None else self.style.edge_layout)
        )

        owns_figure = ax is None
        if owns_figure:
            fig, ax = plt.subplots(
                figsize=figsize or self.style.figsize,
                dpi=dpi or self.style.dpi,
                facecolor=self.style.background,
            )
        else:
            fig = ax.figure
            fig.set_facecolor(self.style.background)

        color_map = self._class_colors(labels)
        effective_edge_alpha = (
            edge_alpha
            if edge_alpha is not None
            else self.style.class_edge_alpha
            if class_id is not None and self.style.class_edge_alpha is not None
            else self.style.edge_alpha
        )
        effective_community_distance = (
            community_distance if community_distance is not None else self.style.community_distance
        )
        Graph(
            graph_nx,
            **self._graph_kwargs(
                graph_nx,
                labels,
                node_layout=active_node_layout,
                edge_layout=active_edge_layout,
                class_id=class_id,
                show_interclass_edges=show_interclass_edges and class_id is None,
                edge_alpha=effective_edge_alpha,
                community_distance=effective_community_distance,
                node_positions=node_positions,
                ax=ax,
            ),
        )

        self._add_legend(ax, color_map, labels, class_id)
        default_title = None if class_id is None else f"Class {class_id}"
        self._finish_axes(ax, title if title is not None else default_title)
        if owns_figure:
            fig.tight_layout(pad=0.2)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi or self.style.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())

        return fig, ax

    def plot_class(self, graph: AttributedGraph, class_id: int, **kwargs):
        return self.plot_scatt_graph(graph, class_id=class_id, **kwargs)

    def generate_and_plot_scatt(
        self,
        *,
        seed: Optional[int] = None,
        plot_kwargs: Optional[Dict[str, Any]] = None,
        **generate_kwargs,
    ):
        from models.SCatt import SCAttGenerator

        graph = SCAttGenerator(seed=seed).generate(**generate_kwargs)
        fig, ax = self.plot_scatt_graph(graph, **(plot_kwargs or {}))
        return graph, fig, ax
