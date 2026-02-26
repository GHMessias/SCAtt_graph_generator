import functools as ft
import itertools as it
import math
import random
from typing import Literal
from dataclasses import dataclass

import networkx as nx
import networkit as nk
import numpy as np
import pandas as pd


# from graphgen_models.SkyMap.metrics import SkyMapMetrics
# from graphgen_models.SkyMap.utils import (
#     distance,
#     gen_beta_moments,
#     imbalance_distribution,
#     pmf_disc_beta,
#     pmf_discrete_log_logistic,
# )

def networkit_to_networkx(g_nk):
    g_nx = nx.Graph()

    # adiciona nós
    g_nx.add_nodes_from(range(g_nk.numberOfNodes()))

    # adiciona arestas
    for u, v in g_nk.iterEdges():
        g_nx.add_edge(u, v)

    return g_nx

from typing import Dict, Hashable, Tuple


def networkx_to_networkit(
    g_nx: nx.Graph,
) -> Tuple[nk.Graph, Dict[Hashable, int], Dict[int, Hashable]]:
    """
    Converte um grafo não-direcionado e não-ponderado de NetworkX
    para um grafo Networkit.

    Retorna:
        g_nk      : nk.Graph
        node2id   : dict rótulo_do_no_original -> índice inteiro (0..n-1)
        id2node   : dict índice inteiro -> rótulo_do_no_original
    """
    # 1) Criar mapeamento de nós
    nodes = list(g_nx.nodes())
    node2id: Dict[Hashable, int] = {node: i for i, node in enumerate(nodes)}
    id2node: Dict[int, Hashable] = {i: node for node, i in node2id.items()}

    # 2) Instanciar grafo Networkit
    g_nk = nk.Graph(len(nodes), weighted=False, directed=False)

    # 3) Adicionar arestas
    for u, v in g_nx.edges():
        u_id = node2id[u]
        v_id = node2id[v]
        if u_id != v_id:  # evita self-loop por via das dúvidas
            g_nk.addEdge(u_id, v_id)

    return g_nk, node2id, id2node

@dataclass
class SkyMapMetrics:
    num_nodes: int
    density: float

    # Class distribution
    num_classes: int
    class_imbalance_ratio: float
    log_logistic_delta: float
    log_logistic_lambda: float

    # Mixing Matrix params.
    dist_x_mean: float
    dist_y_mean: float
    dist_affinity_mean: float
    self_affinity_imbalance_ratio: float
    interclass_affinity_imbalance_ratio: float
    homophily: float

    # Feature distribution metrics
    num_features: float
    zero_gen_means_mean: float
    zero_gen_means_var: float
    zero_gen_vars_mean: float
    zero_gen_vars_var: float

    # Other
    perc_edges_conc_mean: float
    perc_edges_conc_var: float
    perc_nodes_conc_mean: float
    perc_nodes_conc_var: float
    deg_mult_mean: float
    deg_mult_var: float

    @classmethod
    def from_graph(cls, g: nx.Graph):

        print('computing metrics')

        all_metrics = {
            **compute_basic_metrics(g),
            **compute_class_distribution_metrics(g),
            **compute_degree_distribution_metrics(g),
            **compute_feature_distribution_metrics(g),
            **compute_laplacian_matrix_metrics(g),
            **compute_mixing_matrix_metrics(g),
        }

        # Clean metrics to use only the ones defined in dataclass
        required = cls.__annotations__.keys()
        metrics = {k: v for k, v in all_metrics.items() if k in required}
        # print('metrics', metrics)
        return cls(**metrics)

    @property
    def num_edges(self):
        # density = (2 * num_edges) / (num_nodes*(num_nodes-1))
        return self.density * self.num_nodes * (self.num_nodes - 1) / 2

class SkyMap:

    def __init__(self, subgraphs_gen_method: Literal["search", "random"] = "search"):
        self.subgraphs_gen_method = subgraphs_gen_method

    def mimic_graph(self, g: nx.Graph, num_nodes: int = None) -> nx.Graph:
        metrics = SkyMapMetrics.from_graph(g)
        # print('metrics created')
        if num_nodes is not None:
            metrics.num_nodes = num_nodes
        return self.generate_graph(metrics)

    def generate_random_subgraphs(
        self, metrics: SkyMapMetrics, mixing_matrix: np.ndarray
    ) -> list[nx.Graph]:

        nodes_per_class = imbalance_distribution(
            metrics.num_classes, metrics.class_imbalance_ratio
        )
        nodes_per_class = nodes_per_class[::-1]
        classes = range(metrics.num_classes)

        subgraphs: list[nx.Graph] = []
        for c in classes:
            num_nodes = math.ceil(nodes_per_class[c] * metrics.num_nodes)
            num_edges = math.ceil(metrics.num_edges * mixing_matrix[c, c])
            subgraph = nx.gnm_random_graph(num_nodes, num_edges)
            nx.set_node_attributes(subgraph, c, name="y")
            subgraphs.append(subgraph)

        return subgraphs

    def search_subgraphs(
        self, metrics: SkyMapMetrics, mixing_matrix: np.ndarray
    ) -> list[nx.Graph]:

        nodes_per_class = imbalance_distribution(
            metrics.num_classes, metrics.class_imbalance_ratio
        )
        nodes_per_class = nodes_per_class[::-1]
        classes = range(metrics.num_classes)

        subgraphs: list[nx.Graph] = []
        for c in classes:
            num_nodes = math.ceil(nodes_per_class[c] * metrics.num_nodes)
            num_edges = math.ceil(metrics.num_edges * mixing_matrix[c, c])

            log_logistic = ft.partial(
                pmf_discrete_log_logistic,
                d=metrics.log_logistic_delta,
                l=metrics.log_logistic_lambda,
            )
            ps = [log_logistic(i) for i in range(0, metrics.num_nodes)]
            # print('print metrics')
            # print(metrics.num_nodes)
            # print(ps)
            exp_degrees = np.array(
                random.choices(range(0, metrics.num_nodes), weights=ps, k=metrics.num_nodes),
                dtype=float,
            )
            exp_degrees[exp_degrees == 0] = 1e-1
            subgraph = nx.Graph()
            subgraph.add_nodes_from(range(metrics.num_nodes))

            posible_edges = list(it.combinations(range(metrics.num_nodes), 2))
            num_edges = min(num_edges, len(posible_edges))
            w = np.array([exp_degrees[n1] * exp_degrees[n2] for n1, n2 in posible_edges])
            w = w / w.sum()
            edges_to_add = np.random.choice(
                range(len(posible_edges)), size=num_edges, replace=False, p=w
            )
            subgraph.add_edges_from(np.array(posible_edges)[edges_to_add])

            nx.set_node_attributes(subgraph, c, name="y")
            subgraphs.append(subgraph)

        return subgraphs

    def reconstruct_mixing_matrix(
        self, metrics: SkyMapMetrics, make_symmetric: bool = True, max_num_iters: int = 10_000
    ) -> np.ndarray:

        metrics_objective = [metrics.dist_x_mean, metrics.dist_y_mean, metrics.dist_affinity_mean]
        diag = imbalance_distribution(metrics.num_classes, metrics.self_affinity_imbalance_ratio)
        diag = metrics.homophily * (diag / sum(diag))

        X = np.zeros((metrics.num_classes, metrics.num_classes), float)
        np.fill_diagonal(X, diag)
        idx = np.flip(np.argsort(np.diag(X)))
        X = X[idx, :][:, idx]

        other = imbalance_distribution(
            (metrics.num_classes - 1) * metrics.num_classes / 2,
            metrics.interclass_affinity_imbalance_ratio,
        ) * (1 - metrics.homophily)
        dist = math.inf
        for i in range(max_num_iters):

            random.shuffle(other)
            other_proposed = other.copy()

            dist_affinity_array = [0] * metrics.num_classes
            X_i = np.zeros((metrics.num_classes, metrics.num_classes), float)

            index = len(other_proposed) - 1
            for i in range(metrics.num_classes):
                for j in reversed(range(i + 1, metrics.num_classes)):
                    X_i[i, j] = other_proposed[index]
                    index -= 1

            dist_x_list = [0] * metrics.num_classes
            dist_y_list = [0] * (metrics.num_classes - 1)
            for c1 in range(metrics.num_classes):
                for c2 in range(c1 + 1, metrics.num_classes):
                    dist_affinity_array[c2 - c1] += X_i[c1, c2]
                    dist_x_list[c1] += X_i[c1, c2]
                    dist_y_list[c2 - 1] += X_i[c1, c2]

            dist_affinity_array = np.array(dist_affinity_array) / np.array(
                np.arange(metrics.num_classes, 0, -1)
            )
            dist_affinity_array = dist_affinity_array / sum(dist_affinity_array)
            dist_affinity_mean = 0
            for i in range(len(dist_affinity_array)):
                dist_affinity_mean += dist_affinity_array[i] * i

            dist_x_list = np.array(dist_x_list) / np.array(np.arange(metrics.num_classes, 0, -1))
            dist_x_list = dist_x_list / sum(dist_x_list)
            dist_x_mean = 0
            for i in range(len(dist_x_list)):
                dist_x_mean += dist_x_list[i] * i

            dist_y_list = np.array(dist_y_list) / np.array(np.arange(1, metrics.num_classes))
            dist_y_list = dist_y_list / sum(dist_y_list)
            dist_y_mean = 0
            for i in range(len(dist_y_list)):
                dist_y_mean += dist_y_list[i] * i
            metrics_proposed = [dist_x_mean, dist_y_mean, dist_affinity_mean]
            dist_proposed = distance(metrics_proposed, metrics_objective)
            if dist_proposed < dist:
                dist = dist_proposed
                other = other_proposed

        index = len(other) - 1
        for i in range(metrics.num_classes):
            for j in reversed(range(i + 1, metrics.num_classes)):
                X[i, j] = other[index]
                index -= 1

        if make_symmetric:
            X = X + X.T - np.diag(np.diag(X))

        return X

    def assign_properties(
        self,
        graph: nx.Graph,
        metrics: SkyMapMetrics,
        rho_mean: float = None,
        rho_var: float = None,
    ) -> nx.Graph:
        ...
        classes = range(metrics.num_classes)
        means = gen_beta_moments(
            metrics.zero_gen_means_mean, metrics.zero_gen_means_var, metrics.num_classes
        )
        stds = gen_beta_moments(
            metrics.zero_gen_vars_mean, metrics.zero_gen_vars_mean, metrics.num_classes
        )
        A_c = np.array(
            [gen_beta_moments(e[0], e[1], metrics.num_features) for e in zip(means, stds)]
        )

        # return assign_properties_matrixes(G, num_features, A_c, rho_by_class = rho_by_class)

        H = nx.Graph()
        H.add_nodes_from(sorted(graph.nodes(data=True)))
        H.add_edges_from(graph.edges(data=True))

        node_data = pd.DataFrame.from_dict(
            [{"node": i[0], "y": i[1]["y"]} for i in H.nodes.items()]
        )
        # classes = list(set(node_data["y"]))
        num_classes = len(classes)

        deg_data = pd.DataFrame.from_dict(
            [
                {
                    "node": d[0],
                    "k": d[1],
                    "neigh_by_class": [
                        node_data[node_data["node"].isin(graph.neighbors(d[0]))]["y"].eq(c).sum()
                        for c in classes
                    ],
                }
                for d in H.degree()
            ]
        )

        deg_node_data = pd.merge(node_data, deg_data, on="node")
        deg_node_data["neigh_k_by_class"] = [
            [
                deg_node_data[
                    deg_node_data["node"].isin(
                        [e for e in graph.neighbors(e["node"])] + [e["node"]]
                    )
                    & deg_node_data["y"].eq(c)
                ]["k"].sum()
                for c in classes
            ]
            for e in deg_node_data.iloc
        ]
        deg_node_data["neigh_k_internal"] = [
            e["neigh_k_by_class"][e["y"]] for e in deg_node_data.iloc
        ]

        node_classes = np.array([i[1]["y"] for i in H.nodes.items()])

        # A = gen_properties(A_c, classes, H.number_of_nodes(), metrics.num_features, None, None, rho_by_class) #waights)
        rho = None
        w = None
        if w is None:
            w = np.ones((metrics.num_features, nx.number_of_nodes(graph)))
        props = np.array([[1] * metrics.num_features] * nx.number_of_nodes(graph))
        for c in classes:
            idx_all = np.where(node_classes == c)[0]
            for f in range(metrics.num_features):
                p = A_c[c, f]
                class_w = w[f][idx_all]
                class_w /= sum(class_w)
                idx = np.random.choice(
                    len(idx_all), size=int(len(idx_all) * p), replace=False, p=class_w
                )
                props[idx_all[idx], f] = 0
        A = np.array(props)

        for n in H.nodes:
            H.nodes[n]["x"] = A[int(n)].tolist()
        return H

    def add_inter_edges(
        self, graph: nx.Graph, metrics: SkyMapMetrics, mixing_matrix: np.ndarray
    ) -> nx.Graph:
        node_data = pd.DataFrame.from_dict(
            [{"node": i[0], "y": i[1]["y"]} for i in graph.nodes.items()]
        )
        deg_data = pd.DataFrame.from_dict([{"node": d[0], "k": d[1]} for d in graph.degree()])
        deg_node_data = pd.merge(node_data, deg_data, on="node")
        classes = range(metrics.num_classes)
        for c1 in classes:
            for c2 in range(c1 + 1, metrics.num_classes):
                num_edges = math.ceil(metrics.num_edges * mixing_matrix[c1, c2])
                c1_data = deg_node_data[deg_node_data["y"] == classes[c1]]
                c2_data = deg_node_data[deg_node_data["y"] == classes[c2]]

                edges_to_add = set()
                perc_edges_conc = gen_beta_moments(
                    metrics.perc_edges_conc_mean, metrics.perc_edges_conc_var, 2
                )
                perc_nodes_conc = gen_beta_moments(
                    metrics.perc_nodes_conc_mean, metrics.perc_nodes_conc_var, 2
                )

                num_edges_conc_c1 = math.floor(num_edges * perc_edges_conc[0])
                num_edges_conc_c2 = math.floor(num_edges * perc_edges_conc[1])
                num_edges_deg = num_edges - num_edges_conc_c1 - num_edges_conc_c2
                edges_conc_1 = set(
                    random.choices(
                        list(c1_data["node"]),
                        k=math.ceil(len(c1_data["node"]) * perc_nodes_conc[0]),
                    )
                )
                edges_conc_2 = set(
                    random.choices(
                        list(c2_data["node"]),
                        k=math.ceil(len(c2_data["node"]) * perc_nodes_conc[1]),
                    )
                )

                posible_edges_conc1 = list(it.product(edges_conc_1, set(c2_data["node"])))
                edges_to_add.update(random.choices(posible_edges_conc1, k=num_edges_conc_c1))

                posible_edges_conc2 = list(
                    set(it.product(edges_conc_2, set(c1_data["node"]))) - edges_to_add
                )
                edges_to_add.update(random.choices(posible_edges_conc2, k=num_edges_conc_c2))

                posible_edges = list(set(it.product(set(c1_data["node"]), set(c2_data["node"]))))

                deg_mult_mean = metrics.deg_mult_mean
                deg_mult_var = metrics.deg_mult_var

                c1_max_degree = np.array(c1_data["k"]).max() + 1
                c2_max_degree = np.array(c2_data["k"]).max() + 1

                c1_deg_counts = c1_data["k"].value_counts()
                c2_deg_counts = c2_data["k"].value_counts()

                if num_edges_deg > 0:
                    w = np.array(
                        pmf_disc_beta(
                            np.array(
                                [
                                    np.sqrt(c1_k * c2_k / (c1_max_degree * c2_max_degree))
                                    for c1_k, c2_k in it.product(c1_data["k"], c2_data["k"])
                                ]
                            ),
                            deg_mult_mean,
                            deg_mult_var,
                        )
                    ) / np.array(
                        [
                            c1_deg_counts[c1_k] * c2_deg_counts[c2_k]
                            for c1_k, c2_k in it.product(c1_data["k"], c2_data["k"])
                        ]
                    )
                    w = w / w.sum()
                    edges_to_add.update(random.choices(posible_edges, k=num_edges_deg, weights=w))
                graph.add_edges_from(edges_to_add)
        return graph

    def generate_graph(self, metrics: SkyMapMetrics) -> nx.Graph:

        # Reconstruct Mixing Matrix
        mixing_matrix = self.reconstruct_mixing_matrix(metrics)

        # Generate subgraphs
        match self.subgraphs_gen_method:
            case "search":
                subgraphs = self.search_subgraphs(metrics, mixing_matrix)
            case "random":
                subgraphs = self.generate_random_subgraphs(metrics, mixing_matrix)
            case _:
                raise AttributeError(f"Invalid {self.subgraphs_gen_method=}")

        # Combine subgraphs
        G = nx.disjoint_union_all(subgraphs)

        # Assign Properties
        G = self.assign_properties(G, metrics)

        # Add edges to combine subgraphs
        G = self.add_inter_edges(G, metrics, mixing_matrix)

        return G

import sys
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 20)        # número máximo de linhas a mostrar
pd.set_option("display.max_columns", 10)    # número máximo de colunas
pd.set_option("display.width", 120)         # largura da tela (em caracteres)
pd.set_option("display.colheader_justify", "left")  # alinhamento
pd.set_option("display.precision", 3)       # casas decimais

from networkx.algorithms.approximation.clustering_coefficient import (
    average_clustering as average_clustering_approx,
)
from networkx.algorithms.cluster import average_clustering
from scipy.sparse.linalg import eigsh

# from graphgen_models.SkyMap.utils import (
#     fit_degree_distribution,
#     get_mixing_matrix,
#     imb_distr_fit,
#     imb_distr_fit_likelihood,
#     moments_joint_prob,
# )

MAX_NUM_NODES_CLUSTERING_APPROX = 10_000
MAX_CLUSTER_COEFF_TRIALS = 1_000


def compute_basic_metrics(g: nx.Graph) -> dict[str, float]:
    # print('computing basic metrics')
    degrees = [x[1] for x in nx.degree(g)]
    largest_component = g.subgraph(max(nx.connected_components(g), key=len))
    num_nodes = nx.number_of_nodes(g)
    if num_nodes > MAX_NUM_NODES_CLUSTERING_APPROX:
        clustering_coeff = average_clustering_approx(g)
    else:
        clustering_coeff = average_clustering(g)
    return dict(
        num_nodes=num_nodes,
        density=nx.density(g),
        mean_degree=np.mean(degrees),
        max_degree=np.max(degrees),
        max_degree_per=np.mean(degrees) / num_nodes,
        clustering=clustering_coeff,
        degree_assortativity=nx.degree_assortativity_coefficient(g),
        largest_component_perc=nx.number_of_nodes(largest_component) / num_nodes,
    )


def compute_degree_distribution_metrics(g: nx.Graph) -> dict[str, float]:
    # print('computing degree distribution metrics')
    delta, lamda = fit_degree_distribution(g)
    return dict(log_logistic_delta=delta, log_logistic_lambda=lamda)


def compute_laplacian_matrix_metrics(g: nx.Graph) -> dict[str, float]:
    # print('computing laplacian matrix metrics')
    largest_component = g.subgraph(max(nx.connected_components(g), key=len))
    assert len(largest_component) > 1  # Largest comp has at least one node
    laplacian = nx.normalized_laplacian_matrix(largest_component)
    eigenvals = eigsh(laplacian, return_eigenvectors=False, k=1, which="SM")
    return dict(eigen1=eigenvals[0])


def compute_class_distribution_metrics(g: nx.Graph) -> dict[str, float]:
    # print('computing class distribution metrics')
    # print('computing class distribution metrics')
    data_dict = [{"node": i[0], "y": i[1]["y"], "x": i[1]["x"]} for i in g.nodes.items()]
    # print('data ditc', data_dict)
    node_data = pd.DataFrame.from_dict(data_dict)
    # print(node_data)
    node_data.to_csv('a.csv')
    num_classes = len(node_data.y.unique())

    nodes_per_class = node_data["y"].value_counts().to_numpy()
    perc_nodes = nodes_per_class / nodes_per_class.sum()
    k, class_imbalance_ratio, class_imbalance_ratio_tendency = imb_distr_fit(perc_nodes)

    return dict(
        num_classes=num_classes,
        class_imbalance_ratio=class_imbalance_ratio,
    )


def compute_feature_distribution_metrics(g: nx.Graph) -> dict[str, float]:
    # print('computing feature distribution metrics')

    data_dict = [{"node": i[0], "y": i[1]["y"], "x": i[1]["x"]} for i in g.nodes.items()]
    node_data = pd.DataFrame.from_dict(data_dict)
    data_dict = [
        {
            "node": d[0],
            "k": d[1],
            "k_internal": node_data[node_data["node"].isin(g.neighbors(d[0]))]["y"]
            .eq(node_data[node_data["node"].eq(d[0])]["y"].iloc[0])
            .sum(),
        }
        for d in g.degree()
    ]
    deg_data = pd.DataFrame.from_dict(data_dict)
    # print('deg_data', deg_data) # OK
    deg_node_data = pd.merge(node_data, deg_data, on="node")
    # print('deg_node_data \n', deg_node_data)
    deg_node_data["x_per"] = deg_node_data.apply(lambda x: sum(x["x"]) / len(x["x"]), axis=1)

    num_classes = len(node_data.y.unique())
    num_features = len(node_data.x.loc[0])

    # TODO: Refactor
    class_feature_data = node_data
    class_feature_data["idx"] = class_feature_data.index
    class_feature_data = class_feature_data.explode("x", ignore_index=True)
    class_feature_data = class_feature_data.rename(columns={"x": "value", "y": "class"})
    class_feature_data["feature"] = class_feature_data.groupby("idx").cumcount()

    class_feature_data_norm = class_feature_data

    class_feature_data_norm["value"] = (
        class_feature_data_norm["value"]
        / class_feature_data_norm[abs(class_feature_data_norm["value"]) > sys.float_info.epsilon][
            "value"
        ]
        .abs()
        .mean()
    )

    # print('class_feature_data_norm', class_feature_data_norm)

    feature_vec = class_feature_data_norm["value"]
    zero_mask = np.array(abs(feature_vec) < sys.float_info.epsilon)
    non_cero_features = feature_vec[~zero_mask]
    feature_vec_abs = np.abs(non_cero_features)

    num_features = len(node_data["x"].iloc[0])
    perc_0 = sum(zero_mask) / len(feature_vec)

    class_feature_matrix_zeros = np.zeros((num_classes, num_features))
    classes = list(set(node_data["y"]))
    for c in classes:
        feature_vec_c = class_feature_data_norm[(class_feature_data_norm["class"] == c)]
        # print('feature_vec_c', feature_vec_c)
        c_i = classes.index(c)
        means = feature_vec_c.groupby("feature")["value"].mean()
        # print('means', means)
        for k, v in means.items():
            # print('v', v)
            class_feature_matrix_zeros[c_i, k] = 1 - v

    # print('class feature matrix zeros', class_feature_matrix_zeros)
    zero_gen_means = np.array([row.mean() for row in class_feature_matrix_zeros])
    # print('zero_gen_means', zero_gen_means)
    zero_gen_vars = np.array([row.var() for row in class_feature_matrix_zeros])
    zero_gen_means_mean = zero_gen_means.mean()
    zero_gen_means_var = zero_gen_means.var()
    zero_gen_vars_mean = zero_gen_vars.mean()
    zero_gen_vars_var = zero_gen_vars.var()

    list_intdeg = []
    list_num_edges_concentrated_links = []
    list_perc_edges_concentrated_links = []
    mixing_matrix_abs = np.zeros((num_classes, num_classes))
    for c1 in classes:
        neigbour_classes = []
        nodes = deg_node_data[deg_node_data["y"] == c1]
        neigh_ids = []
        neigh_deg_internal = []
        deg_x_internal = []
        for _, d in nodes.iterrows():
            neighs = list(g.neighbors(d["node"]))  # + [d['node']]
            neighbours = deg_node_data[deg_node_data["node"].isin(neighs)]
            neigh_classes = np.array(neighbours["y"])
            neigh_ids.extend(np.array(neighbours["node"]))
            neigh_deg_internal.extend(np.array(neighbours["k_internal"]))
            deg_x_internal.extend([d["k_internal"]] * len(np.array(neighbours["k_internal"])))
            if len(neigh_classes) > 0:
                neigbour_classes.extend(neigh_classes)
        for c2 in classes:
            mixing_matrix_abs[classes.index(c1)][classes.index(c2)] = neigbour_classes.count(c2)
            if c1 != c2:
                raw_interk = np.array(neigh_ids)[np.array(neigbour_classes) == c2]
                data_inerk = np.unique(raw_interk, return_counts=True)
                indexes = data_inerk[1] > data_inerk[1].mean() + 3 * data_inerk[1].std()
                x = np.array(deg_x_internal)[np.array(neigbour_classes) == c2] / (
                    deg_node_data[deg_node_data["y"] == c1]["k_internal"].max() + 1
                )
                y = np.array(neigh_deg_internal)[np.array(neigbour_classes) == c2] / (
                    deg_node_data[deg_node_data["y"] == c2]["k_internal"].max() + 1
                )
                list_intdeg.extend(np.sqrt(x * y))
                if len(set(data_inerk[0])) > 0:
                    list_num_edges_concentrated_links.append(
                        (indexes).sum() / len(set(data_inerk[0]))
                    )
                if sum(data_inerk[1]) > 0:
                    list_perc_edges_concentrated_links.append(
                        sum(data_inerk[1][indexes]) / sum(data_inerk[1])
                    )

    for c in range(num_classes):
        mixing_matrix_abs[c, c] = mixing_matrix_abs[c, c] / 2

    corrs = []
    for c in classes:
        class_nodes = node_data[node_data["y"].eq(c)]
        assert len(class_nodes) > 1  # At least 2 nodes per class
        _, probs = moments_joint_prob(class_nodes)
        # np.savetxt(Path(out_path ,f"cov_{dataset.lower()}_{c}.txt"), cov)
        corrs.extend(probs)

    return dict(
        num_features=num_features,
        perc_0=perc_0,
        zero_gen_means_mean=zero_gen_means_mean,
        zero_gen_means_var=zero_gen_means_var,
        zero_gen_vars_mean=zero_gen_vars_mean,
        zero_gen_vars_var=zero_gen_vars_var,
        perc_nodes_conc_mean=np.mean(list_num_edges_concentrated_links),
        perc_nodes_conc_var=np.var(list_num_edges_concentrated_links),
        perc_edges_conc_mean=np.mean(list_perc_edges_concentrated_links),
        perc_edges_conc_var=np.var(list_perc_edges_concentrated_links),
        deg_mult_mean=np.mean(list_intdeg),
        deg_mult_var=np.var(list_intdeg),
        corr_mean=np.mean(corrs),
        corr_var=np.var(corrs),
    )


def compute_mixing_matrix_metrics(g: nx.Graph) -> dict[str, float]:
    # print('computing mixing matrix metrics')
    mixing_matrix = get_mixing_matrix(g)

    num_classes = len(mixing_matrix)

    self_affinity_list = []
    interclass_affinity_list = []
    dist_affinity = [0] * num_classes

    for c1 in range(num_classes):
        self_affinity_list.append(mixing_matrix[c1, c1])
        for c2 in range(c1 + 1, num_classes):
            interclass_affinity_list.append(mixing_matrix[c1, c2])
            dist_affinity[c2 - c1] += mixing_matrix[c1, c2]

    self_affinity_list = np.array(self_affinity_list)
    interclass_affinity_list = np.array(interclass_affinity_list)

    homophily = sum(self_affinity_list)
    self_affinity_list_norm = self_affinity_list / sum(self_affinity_list)
    self_affinity_imbalance_ratio = imb_distr_fit_likelihood(self_affinity_list_norm)
    interclass_affinity_list_norm = interclass_affinity_list / sum(interclass_affinity_list)
    interclass_affinity_imbalance_ratio = imb_distr_fit_likelihood(interclass_affinity_list_norm)

    dist_affinity_array = [0] * num_classes
    dist_x_list = [0] * num_classes
    dist_y_list = [0] * (num_classes - 1)
    for c1 in range(num_classes):
        for c2 in range(c1 + 1, num_classes):
            dist_affinity_array[c2 - c1] += mixing_matrix[c1, c2]
            dist_x_list[c1] += mixing_matrix[c1, c2]
            dist_y_list[c2 - 1] += mixing_matrix[c1, c2]
    dist_affinity_array = np.array(dist_affinity_array) / np.array(np.arange(num_classes, 0, -1))
    dist_affinity_array = dist_affinity_array / sum(dist_affinity_array)
    dist_affinity_mean = 0
    for i in range(len(dist_affinity_array)):
        dist_affinity_mean += dist_affinity_array[i] * i
    dist_affinity_std = np.sqrt(
        np.cov(range(len(dist_affinity_array)), aweights=dist_affinity_array)
    )

    dist_x_list = np.array(dist_x_list) / np.array(np.arange(num_classes, 0, -1))
    dist_x_list = dist_x_list / sum(dist_x_list)
    dist_x_mean = 0
    for i in range(len(dist_x_list)):
        dist_x_mean += dist_x_list[i] * i

    dist_y_list = np.array(dist_y_list) / np.array(np.arange(1, num_classes))
    dist_y_list = dist_y_list / sum(dist_y_list)
    dist_y_mean = 0
    for i in range(len(dist_y_list)):
        dist_y_mean += dist_y_list[i] * i

    return dict(
        num_classes=num_classes,
        homophily=homophily,
        self_affinity_imbalance_ratio=self_affinity_imbalance_ratio,
        interclass_affinity_imbalance_ratio=interclass_affinity_imbalance_ratio,
        dist_affinity_mean=dist_affinity_mean,
        dist_affinity_std=dist_affinity_std,
        dist_x_mean=dist_x_mean,
        dist_y_mean=dist_y_mean,
    )


import math
import random
import sys

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta


def cmf_discrete_log_logistic(x, d, l):
    return 1 / (1 + ((x + 1) / l) ** -d)


def pmf_discrete_log_logistic(x, d, l):
    return cmf_discrete_log_logistic(x + 1, d, l) - cmf_discrete_log_logistic(x, d, l)


def log_pmf_discrete_log_logistic(x, d, l):
    return np.log(pmf_discrete_log_logistic(x, d, l))


def fit_degree_distribution(G):
    try:
        num_nodes = len(G.nodes)
        histogram = np.array(nx.degree_histogram(G)) / num_nodes
        degree_list = np.array([val for (_, val) in G.degree()])

        def neg_loglike(params):
            d, l = params
            res = -np.sum(log_pmf_discrete_log_logistic(degree_list, d, l))
            return res

        def least_square(params):
            d, l = params
            degrees = np.array(range(len(histogram)))
            res = np.sqrt(np.sum((pmf_discrete_log_logistic(degrees, d, l) - histogram) ** 2))
            return res

        mean_degree = len(G.edges) / len(G.nodes)

        res = minimize(
            neg_loglike, [3, mean_degree], method="Nelder-Mead", bounds=[(1, 1000), (0, num_nodes)]
        )
        return res["x"]
    except:
        raise ValueError("Could not fig log-logistic distribution to degrees")
        # return [0, 0]


def normalize_rho(marginals, corr_matrix):
    n_features = len(marginals)
    corrnorm_matrix = np.zeros([n_features, n_features])
    for i in range(n_features):
        for j in range(n_features):
            p = marginals[i]
            tita_1 = p * (1 - p)
            q = marginals[j]
            tita_2 = q * (1 - q)
            if i != j and p * q > 0:

                corr_min = (max(0, p + q - 1) - p * q) / math.sqrt(tita_1 * tita_2)

                corr_max = (min(p, q) - p * q) / math.sqrt(tita_1 * tita_2)
                corrnorm_matrix[i][j] = (corr_matrix[i][j] - corr_min) / (corr_max - corr_min)
    return corrnorm_matrix


def calculate_joint_prob(node_data, num_features):
    matrix = np.zeros([num_features, num_features])
    for i in range(num_features):
        for j in range(num_features):
            matrix[i][j] = (
                sum(np.array(node_data["x"].iloc[i]) * np.array(node_data["x"].iloc[j]))
                / num_features
            )
    return matrix


def moments_joint_prob(node_data):
    features = np.array(list(node_data["x"].iloc))
    probs = []
    corr_matrix = np.nan_to_num(np.corrcoef(features, rowvar=False))
    marginals = np.sum(features, axis=0) / len(features)
    zero_vars = [i for i, p in enumerate(marginals) if p == 0]
    normcorr_matrix = normalize_rho(marginals, corr_matrix)
    non_zero_corr_matrix = np.delete(
        np.delete(normcorr_matrix, zero_vars, axis=0), zero_vars, axis=1
    )

    probs = non_zero_corr_matrix[np.triu_indices_from(non_zero_corr_matrix, 1)]
    return corr_matrix, probs


def imbalance_distribution(k, ir):
    if k == 1:
        return np.array([1])
    p = 1 / k + (1 - 1 / k) * ir
    dstr = np.append(imbalance_distribution(k - 1, ir) * (1 - p), p)
    return dstr


def imb_distr_fit(x):
    elements = len(x)
    if elements == 1:
        return (1, 0, 0)

    x_sorted = sorted(x, reverse=True)
    irs = np.empty(elements - 1)
    for i in range(elements - 1):
        k = elements - i
        rest = sum(x_sorted[i:])
        m = x_sorted[i] / rest if rest > 0 else 1
        irs[i] = (m - 1 / k) / (1 - 1 / k)
    importance = np.linspace(1, 0, len(irs))
    irs_mean = sum(irs * importance) / sum(importance)
    return (
        elements,
        irs_mean,
        0 if elements == 2 else np.sqrt(np.sum((irs - [irs_mean] * (elements - 1)) ** 2)),
        # 0 if elements == 2 else mean(irs[:-1] - irs[1:])
    )


def imb_distr_fit_likelihood(x):
    elements = len(x)
    if elements == 1:
        return 0

    x_sorted = sorted(x, reverse=True)
    generated = random.choices(range(elements), k=1000, weights=x_sorted)

    def neg_loglike(ir):
        distr = list(reversed(imbalance_distribution(elements, ir[0])))
        logs = np.log([distr[e] for e in generated])
        res = -np.sum(logs)
        return res

    res = minimize(neg_loglike, [0.1], method="Nelder-Mead", bounds=[(0, 1)])
    return res["x"][0]


# Mixing matrix procs


def get_mixing_matrix(g: nx.Graph) -> np.ndarray:

    data_dict = [{"node": i[0], "y": i[1]["y"], "x": i[1]["x"]} for i in g.nodes.items()]
    node_data = pd.DataFrame.from_dict(data_dict)
    data_dict = [
        {
            "node": d[0],
            "k": d[1],
            "k_internal": node_data[node_data["node"].isin(g.neighbors(d[0]))]["y"]
            .eq(node_data[node_data["node"].eq(d[0])]["y"].iloc[0])
            .sum(),
        }
        for d in g.degree()
    ]
    deg_data = pd.DataFrame.from_dict(data_dict)
    deg_node_data = pd.merge(node_data, deg_data, on="node")
    deg_node_data["x_per"] = deg_node_data.apply(lambda x: sum(x["x"]) / len(x["x"]), axis=1)

    classes = list(set(node_data["y"]))
    num_classes = len(classes)

    list_intdeg = []
    list_num_edges_concentrated_links = []
    list_perc_edges_concentrated_links = []
    mixing_matrix_abs = np.zeros((num_classes, num_classes))
    for c1 in classes:
        neigbour_classes = []
        nodes = deg_node_data[deg_node_data["y"] == c1]
        neigh_ids = []
        neigh_deg_internal = []
        deg_x_internal = []
        for _, d in nodes.iterrows():
            neighs = list(g.neighbors(d["node"]))  # + [d['node']]
            neighbours = deg_node_data[deg_node_data["node"].isin(neighs)]
            neigh_classes = np.array(neighbours["y"])
            neigh_ids.extend(np.array(neighbours["node"]))
            neigh_deg_internal.extend(np.array(neighbours["k_internal"]))
            deg_x_internal.extend([d["k_internal"]] * len(np.array(neighbours["k_internal"])))
            if len(neigh_classes) > 0:
                neigbour_classes.extend(neigh_classes)
        for c2 in classes:
            mixing_matrix_abs[classes.index(c1)][classes.index(c2)] = neigbour_classes.count(c2)
            if c1 != c2:
                raw_interk = np.array(neigh_ids)[np.array(neigbour_classes) == c2]
                data_inerk = np.unique(raw_interk, return_counts=True)
                indexes = data_inerk[1] > data_inerk[1].mean() + 3 * data_inerk[1].std()
                x = np.array(deg_x_internal)[np.array(neigbour_classes) == c2] / (
                    deg_node_data[deg_node_data["y"] == c1]["k_internal"].max() + 1
                )
                y = np.array(neigh_deg_internal)[np.array(neigbour_classes) == c2] / (
                    deg_node_data[deg_node_data["y"] == c2]["k_internal"].max() + 1
                )
                list_intdeg.extend(np.sqrt(x * y))
                if len(set(data_inerk[0])) > 0:
                    list_num_edges_concentrated_links.append(
                        (indexes).sum() / len(set(data_inerk[0]))
                    )
                if sum(data_inerk[1]) > 0:
                    list_perc_edges_concentrated_links.append(
                        sum(data_inerk[1][indexes]) / sum(data_inerk[1])
                    )

    for c in range(num_classes):
        mixing_matrix_abs[c, c] = mixing_matrix_abs[c, c] / 2

    idx = np.flip(np.argsort(np.diag(mixing_matrix_abs)))
    mixing_matrix_abs = mixing_matrix_abs[idx, :][:, idx]

    mixing_matrix = mixing_matrix_abs / nx.number_of_edges(g)
    return mixing_matrix


def decompose_mixin_matrix(mixing_matrix):
    num_classes = len(mixing_matrix)

    self_affinity_list = []
    interclass_affinity_list = []
    dist_affinity = [0] * num_classes

    for c1 in range(num_classes):
        self_affinity_list.append(mixing_matrix[c1, c1])
        for c2 in range(c1 + 1, num_classes):
            interclass_affinity_list.append(mixing_matrix[c1, c2])
            dist_affinity[c2 - c1] += mixing_matrix[c1, c2]

    self_affinity_list = np.array(self_affinity_list)
    interclass_affinity_list = np.array(interclass_affinity_list)

    homophility = sum(self_affinity_list)
    self_affinity_list_norm = self_affinity_list / sum(self_affinity_list)
    self_affinity_imbalance_ratio = imb_distr_fit_likelihood(self_affinity_list_norm)
    interclass_affinity_list_norm = interclass_affinity_list / sum(interclass_affinity_list)
    interclass_affinity_imbalance_ratio = imb_distr_fit_likelihood(interclass_affinity_list_norm)

    dist_affinity_array = [0] * num_classes
    dist_x_list = [0] * num_classes
    dist_y_list = [0] * (num_classes - 1)
    for c1 in range(num_classes):
        for c2 in range(c1 + 1, num_classes):
            dist_affinity_array[c2 - c1] += mixing_matrix[c1, c2]
            dist_x_list[c1] += mixing_matrix[c1, c2]
            dist_y_list[c2 - 1] += mixing_matrix[c1, c2]
    dist_affinity_array = np.array(dist_affinity_array) / np.array(np.arange(num_classes, 0, -1))
    dist_affinity_array = dist_affinity_array / sum(dist_affinity_array)
    dist_affinity_mean = 0
    for i in range(len(dist_affinity_array)):
        dist_affinity_mean += dist_affinity_array[i] * i
    dist_affinity_std = np.sqrt(
        np.cov(range(len(dist_affinity_array)), aweights=dist_affinity_array)
    )

    dist_x_list = np.array(dist_x_list) / np.array(np.arange(num_classes, 0, -1))
    dist_x_list = dist_x_list / sum(dist_x_list)
    dist_x_mean = 0
    for i in range(len(dist_x_list)):
        dist_x_mean += dist_x_list[i] * i

    dist_y_list = np.array(dist_y_list) / np.array(np.arange(1, num_classes))
    dist_y_list = dist_y_list / sum(dist_y_list)
    dist_y_mean = 0
    for i in range(len(dist_y_list)):
        dist_y_mean += dist_y_list[i] * i

    return {
        "num_classes": num_classes,
        "homophility": homophility,
        "self_affinity_imbalance_ratio": self_affinity_imbalance_ratio,
        "interclass_affinity_imbalance_ratio": interclass_affinity_imbalance_ratio,
        "dist_affinity_mean": dist_affinity_mean,
        "dist_affinity_std": dist_affinity_std,
        "dist_x_mean": dist_x_mean,
        "dist_y_mean": dist_y_mean,
    }


def distance(a, b):
    values = zip(a, b)
    percential_vector = np.array([abs(v[0] - v[1]) / max(abs(v[0]), abs(v[1])) for v in values])
    percential_vector = np.nan_to_num(percential_vector)
    return np.linalg.norm(percential_vector, ord=2)


# Feuature/property generator


def gen_beta_moments(mean, var, n):
    # print('mean var n', mean, var, n)
    if mean == 1:
        return np.asarray([1] * n)
    elif mean == 0:
        return np.asarray([0] * n)
    elif var < sys.float_info.epsilon:
        return np.asarray([mean] * n)
    else:
        # return np.asarray(np.random.normal(mean, sqrt(var), n)).clip(0,1)
        # a=mean*(mean*(1-mean)/var-1)
        # b=a*(1-mean)/mean
        # a = ((1-mean)/var - 1/mean)* mean * mean
        # b = ((1-mean)/var - 1/mean) * mean * (1-mean)
        if var >= mean * (1 - mean):
            c = 1e-8
        else:
            c = (mean * (1 - mean)) / var - 1
        a = mean * c
        b = (1 - mean) * c

        return np.asarray(beta.rvs(a, b, loc=0, scale=1, size=n))


# Other


def pmf_disc_beta(x, mean, var):
    available = sorted(list(set(x)))
    available.append(np.nan)
    x_1 = x + np.array(available)[[available.index(x_i) + 1 for x_i in x]]
    c = (mean * (1 - mean)) / var - 1
    a = mean * c
    b = (1 - mean) * c
    return np.nan_to_num(beta.cdf(x_1, a, b), nan=1) - beta.cdf(x, a, b)


##############################
####### SkyMap Class #########
##############################

from core.attributed_graph import AttributedGraph
from core.base_graph_generator import BaseGenerator
import torch
import networkit as nk
import io
import numpy as np
from typing import Optional, Literal

class SkyMapGenerator(BaseGenerator):
    def __init__(self):
        super().__init__(name='SkyMap', supports_augment=True, supports_mimic=True)

    def _run_skymap(self, num_nodes, num_classes, density, class_imbalance_ratio=0.5,
    log_logistic_delta=0.1, log_logistic_lambda=0.2, dist_x_mean=0.2, dist_y_mean=0.2, 
    dist_affinity_mean=0.3, self_affinity_imbalance_ratio=0.4, interclass_affinity_imbalance_ratio=0.1,
    homophily=0.7, num_features=20, zero_gen_means_mean=0.5, zero_gen_means_var=0.3, zero_gen_vars_mean=0.2,
    zero_gen_vars_var=0.4, perc_edges_conc_mean=0.1, perc_edges_conc_var=0.005, perc_nodes_conc_mean=0.5, perc_nodes_conc_var=0.005,
    deg_mult_mean=0.5, deg_mult_var=0.05):
        
        metrics = SkyMapMetrics(num_nodes = num_nodes, num_classes = num_classes, density = density, class_imbalance_ratio = class_imbalance_ratio,
                                log_logistic_delta = log_logistic_delta, log_logistic_lambda = log_logistic_lambda, dist_x_mean = dist_x_mean,
                                dist_y_mean = dist_y_mean, dist_affinity_mean = dist_affinity_mean, self_affinity_imbalance_ratio = self_affinity_imbalance_ratio, interclass_affinity_imbalance_ratio=interclass_affinity_imbalance_ratio, homophily=homophily, num_features=num_features, zero_gen_means_mean=zero_gen_means_mean, zero_gen_means_var=zero_gen_means_var, zero_gen_vars_mean=zero_gen_vars_mean, zero_gen_vars_var=zero_gen_vars_var, perc_edges_conc_mean=perc_edges_conc_mean, perc_edges_conc_var=perc_edges_conc_var, perc_nodes_conc_mean=perc_nodes_conc_mean, perc_nodes_conc_var=perc_nodes_conc_var, deg_mult_mean=deg_mult_mean, deg_mult_var=deg_mult_var
                                )
        skynet = SkyMap()
        return skynet.generate_graph(metrics = metrics)
    
    def generate(self, num_nodes, num_classes, density, class_imbalance_ratio=0.5,
    log_logistic_delta=0.1, log_logistic_lambda=0.2, dist_x_mean=0.2, dist_y_mean=0.2, 
    dist_affinity_mean=0.3, self_affinity_imbalance_ratio=0.4, interclass_affinity_imbalance_ratio=0.1,
    homophily=0.7, num_features=20, zero_gen_means_mean=0.5, zero_gen_means_var=0.3, zero_gen_vars_mean=0.2,
    zero_gen_vars_var=0.4, perc_edges_conc_mean=0.1, perc_edges_conc_var=0.005, perc_nodes_conc_mean=0.5, perc_nodes_conc_var=0.005,
    deg_mult_mean=0.5, deg_mult_var=0.05):
        
        # Ainda precisa retornar um objeto AttributedGraph
        return self._run_skymap(num_nodes = num_nodes, num_classes = num_classes, density = density, class_imbalance_ratio = class_imbalance_ratio,
                                log_logistic_delta = log_logistic_delta, log_logistic_lambda = log_logistic_lambda, dist_x_mean = dist_x_mean,
                                dist_y_mean = dist_y_mean, dist_affinity_mean = dist_affinity_mean, self_affinity_imbalance_ratio = self_affinity_imbalance_ratio, interclass_affinity_imbalance_ratio=interclass_affinity_imbalance_ratio, homophily=homophily, num_features=num_features, zero_gen_means_mean=zero_gen_means_mean, zero_gen_means_var=zero_gen_means_var, zero_gen_vars_mean=zero_gen_vars_mean, zero_gen_vars_var=zero_gen_vars_var, perc_edges_conc_mean=perc_edges_conc_mean, perc_edges_conc_var=perc_edges_conc_var, perc_nodes_conc_mean=perc_nodes_conc_mean, perc_nodes_conc_var=perc_nodes_conc_var, deg_mult_mean=deg_mult_mean, deg_mult_var=deg_mult_var)
    
    def mimic(self, data_graph, num_nodes:Optional[int] = None):
        if num_nodes == None:
            num_nodes = data_graph.num_nodes()
        
        # Mudança do grafo networkit de data_graph.graph para networkx (input do skymap)

        tmp_nx_graph = networkit_to_networkx(data_graph.graph)
        # nx.set_node_attributes(tmp_nx_graph, dict(zip(tmp_nx_graph.nodes(), data_graph.y)), "y")
        # nx.set_node_attributes(tmp_nx_graph, dict(zip(tmp_nx_graph.nodes(), data_graph.x)), "x")
        nx.set_node_attributes(tmp_nx_graph, dict(zip(data_graph.graph.iterNodes(), data_graph.y.tolist())), "y")
        nx.set_node_attributes(tmp_nx_graph, dict(zip(data_graph.graph.iterNodes(), data_graph.x.tolist())), "x")

        print(
            data_graph.num_nodes(),
            len(tmp_nx_graph.nodes()),
            len([tmp_nx_graph.nodes[n]["y"] for n in tmp_nx_graph.nodes()])
        )

        skymap = SkyMap()
        gen_graph = skymap.mimic_graph(tmp_nx_graph, num_nodes=num_nodes)

        tmp_nk_graph,_,_ = networkx_to_networkit(gen_graph)
        y = torch.tensor([tmp_nx_graph.nodes[n]["y"] for n in tmp_nx_graph.nodes()], dtype = torch.long)
        x = torch.tensor([tmp_nx_graph.nodes[n]["x"] for n in tmp_nx_graph.nodes()], dtype = torch.float)

        return AttributedGraph(graph = tmp_nk_graph, x = x, y = y)