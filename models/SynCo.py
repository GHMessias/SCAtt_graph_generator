import sys

sys.path.append('')

import numpy as np
import torch
from core.base_graph_generator import BaseGenerator
from core.attributed_graph import AttributedGraph
import networkit as nk
from scipy.stats import rankdata

from typing import List, Sequence, Any, Optional, Dict
import math
import warnings

def compact_graph_ids(g: nk.Graph):
    """
    Retorna:
      g2: novo grafo com nós 0..n-1
      old2new: dict {old_id -> new_id} apenas para nós existentes
      new2old: list onde new2old[new_id] = old_id
    """
    # nós que "existem" (iterNodes() já pula os removidos)
    nodes = list(g.iterNodes())
    if nodes == list(range(len(nodes))):
        return g

    old2new = {old: new for new, old in enumerate(nodes)}
    new2old = nodes

    g2 = nk.Graph(len(nodes), weighted=g.isWeighted(), directed=g.isDirected())

    # copia arestas
    for u, v in g.iterEdges():
        g2.addEdge(old2new[u], old2new[v])

    return g2

def split_by_class_and_partitions(
    n: int,
    y: List[int],
    parts: List[List[float]],
    items: Optional[Sequence[Any]] = None,  # se None, trabalha com índices 0..n-1
    method: str = "largest_remainder",       # como arredondar
):
    """
    Separa n elementos em classes (y) e, dentro de cada classe, em partições (parts).

    - n: número total de elementos
    - y: contagem de elementos por classe (ex.: [300, 500, 200])
    - parts: proporções por classe (ex.: [[0.5,0.5],[1],[0.3,0.7]])
    - items: se você passar uma lista/array com n itens, o retorno será com itens;
             se None, o retorno será com índices.
    - method: "largest_remainder" (recomendado) garante somas exatas.

    Retorna:
      result[class_idx][part_idx] = lista de índices (ou itens) naquela subpartição.
    """
    if sum(y) != n:
        raise ValueError(f"sum(y) deve ser igual a n. sum(y)={sum(y)} e n={n}")

    if len(y) != len(parts):
        raise ValueError("y e parts precisam ter o mesmo número de classes.")

    if items is None:
        items = list(range(n))
    else:
        if len(items) != n:
            raise ValueError("items precisa ter tamanho n.")

    result = []
    start = 0

    for c, (class_count, p) in enumerate(zip(y, parts)):
        if not p:
            raise ValueError(f"Classe {c}: lista de partições vazia.")
        if any(pi < 0 for pi in p):
            raise ValueError(f"Classe {c}: proporções negativas não são permitidas.")
        s = sum(p)
        if s <= 0:
            raise ValueError(f"Classe {c}: soma das proporções deve ser > 0.")
        # normaliza (se já soma 1, não muda)
        p = [pi / s for pi in p]

        # pega os elementos dessa classe
        class_items = items[start:start + class_count]
        start += class_count

        if method == "largest_remainder":
            # alocação inteira mantendo soma exata
            raw = [class_count * pi for pi in p]
            base = [int(math.floor(v)) for v in raw]
            missing = class_count - sum(base)

            # distribui o que faltou para as maiores partes fracionárias
            frac = [(raw[i] - base[i], i) for i in range(len(p))]
            frac.sort(reverse=True)
            for k in range(missing):
                base[frac[k][1]] += 1

            sizes = base
        else:
            # arredonda simples (pode dar soma errada)
            sizes = [round(class_count * pi) for pi in p]
            diff = class_count - sum(sizes)
            sizes[0] += diff  # "conserta" jogando tudo na primeira partição

        # fatia em partições
        subparts = []
        s2 = 0
        for sz in sizes:
            subparts.append(class_items[s2:s2 + sz])
            s2 += sz

        result.append(subparts)

    return result

def distribution_transform(dst: str, n: int, powerlaw_exponent: float = 2.0, seed=None,
                           mu=None, normal_std=None, eps: float = 1e-6):

    ranks = torch.arange(1, n + 1, dtype=torch.float64)

    if dst == 'power_law':
        w = ranks ** (-powerlaw_exponent)

    elif dst == 'normal':
        # centro e dispersão "compatíveis" com ranks 1..n
        if mu is None:
            mu = (n + 1) / 2.0
        if normal_std is None:
            normal_std = max(n / 6.0, 1.0)  # regra prática: cobre bem o intervalo

        w = torch.exp(- (ranks - mu) ** 2 / (2 * normal_std ** 2))

    elif dst == 'uniform':
        w = torch.ones(n, dtype=torch.float64)

    else:
        raise ValueError(f"Distribuição desconhecida: {dst}")

    # suavização para evitar concentração absurda / underflow
    w = w + eps
    p = w / w.sum()
    return p


def _to_int_list(values, name: str) -> list[int]:
    """Convert common array-like inputs to a plain list of Python ints."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().view(-1).tolist()

    try:
        return [int(v) for v in values]
    except TypeError as exc:
        raise TypeError(f"{name} deve ser uma sequencia de inteiros.") from exc


def _to_float_tensor(values, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float)
    if torch.any(tensor < 0):
        raise ValueError(f"{name} nao pode conter valores negativos.")
    return tensor


def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    weights = weights.float()
    if weights.numel() == 0:
        raise ValueError("Nao e possivel amostrar de uma distribuicao vazia.")
    total = weights.sum()
    if total <= 0:
        return torch.ones_like(weights) / len(weights)
    return weights / total


def _sample_index(weights: torch.Tensor) -> int:
    weights = _normalize_weights(weights)
    return torch.multinomial(weights, num_samples=1).item()


def _numpy_probabilities(weights: torch.Tensor) -> np.ndarray:
    probabilities = _normalize_weights(weights).detach().cpu().numpy().astype(np.float64)
    probabilities[probabilities < 0] = 0.0
    total = probabilities.sum(dtype=np.float64)
    if total <= 0:
        probabilities = np.ones_like(probabilities, dtype=np.float64) / len(probabilities)
    else:
        probabilities = probabilities / total
    return probabilities


def _candidate_batch_size(missing_edges: int) -> int:
    return min(max(int(missing_edges) * 3, 4_096), 200_000)


class _CategoricalSampler:
    """Buffered categorical sampler for hot loops with fixed weights."""

    def __init__(self, weights: torch.Tensor, batch_size: int = 4096):
        self.weights = _normalize_weights(weights)
        self.batch_size = max(1, int(batch_size))
        self._buffer: list[int] = []

    def sample(self) -> int:
        if not self._buffer:
            self._buffer = torch.multinomial(
                self.weights,
                num_samples=self.batch_size,
                replacement=True,
            ).tolist()
        return self._buffer.pop()


def _edge_key(u: int, v: int) -> tuple[int, int]:
    u = int(u)
    v = int(v)
    return (u, v) if u < v else (v, u)


def _has_edge(graph: nk.Graph, u: int, v: int, edge_set: Optional[set[tuple[int, int]]] = None) -> bool:
    if edge_set is not None:
        return _edge_key(u, v) in edge_set
    return graph.hasEdge(u, v)


def _add_edge(graph: nk.Graph, u: int, v: int, edge_set: Optional[set[tuple[int, int]]] = None):
    graph.addEdge(u, v)
    if edge_set is not None:
        edge_set.add(_edge_key(u, v))


def _prepare_partition_matrix(matrix, num_partitions: int, label: int) -> torch.Tensor:
    matrix = _to_float_tensor(matrix, f"A_in[{label}]")

    if matrix.ndim == 1 and num_partitions == 1 and matrix.numel() == 1:
        matrix = matrix.reshape(1, 1)

    if matrix.ndim != 2 or tuple(matrix.shape) != (num_partitions, num_partitions):
        raise ValueError(
            f"A_in[{label}] deve ter shape ({num_partitions}, {num_partitions}); "
            f"recebido {tuple(matrix.shape)}."
        )

    return matrix

class graphPartition:
    """
    graphPartition recebe um subgrafo homogêneo (onde todos os vértices contém a mesma classe) e retorna a partição daquele subgrafo. A partição
    é detectada utilizando o algoritmo base_community_detector, definido como nk.community.PLM por padrão.
    """

    def __init__(self, subgraph: nk.Graph, base_community_detector):
        self.subgraph = subgraph
        self.base_community_detector = base_community_detector

        self._generate_partitions()
        self.compute_partition_heterogeneity()
        self.compute_partition_probability()

        # print(self.graph_partitions)
        return

    def _get_partition(self, node):
        """
        Return the partition of the node
        
        :param self: Description
        :param node: int
        """
        return self.node_to_partition.get(int(node))
            
    def _number_of_partitions(self):
        return max(list(self.graph_partitions.keys())) + 1

    def _generate_partitions(self):
        base_detector = self.base_community_detector(self.subgraph, refine = True)
        base_detector.run()
        partition = base_detector.getPartition()

        # pega os ids reais das comunidades (não vazias)
        subset_ids = partition.getSubsetIds()  # list de ints

        # mapeia para 0,1,2,... mas preservando os membros
        self.graph_partitions = {
            new_id: list(partition.getMembers(old_id))
            for new_id, old_id in enumerate(subset_ids)
        }
        self.node_to_partition = {
            int(node): partition_id
            for partition_id, nodes in self.graph_partitions.items()
            for node in nodes
        }

    def compute_partition_heterogeneity(self):
        '''
        Essa função calcula a probabilidade de uma aresta ser intra-partição e inter-partição, para cada uma das partições. A saída dela é uma matriz A_in onde A_in_{i,j} representa a probabilidade da partição i ser ligada na partição j
        
        :param self: Description
        '''

        self.A_in = torch.zeros((self._number_of_partitions(), self._number_of_partitions()))

        for u,v in self.subgraph.iterEdges():
            # print(self._get_partition(u), self._get_partition(v))
            u_partition = self._get_partition(u)
            v_partition = self._get_partition(v)
            self.A_in[u_partition, v_partition] += 1
            self.A_in[v_partition, u_partition] += 1
        return

    def compute_partition_probability(self):
        self.partition_probabilities = torch.tensor([len(partition) for partition in self.graph_partitions.values()])/self.subgraph.numberOfNodes()
        return
    
class SynCoGenerator(BaseGenerator):
    def __init__(self, seed: int | None = None):
        super().__init__(name='SynCo', supports_mimic=True, supports_augment=True)

        self.seed = seed
        if seed is not None:
            self._set_seed(seed)

    def _set_seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _validate_generate_inputs(
        self,
        n,
        y,
        k,
        e,
        A_in,
        dst,
        rho,
        A_out,
        S,
    ):
        y = _to_int_list(y, "y")
        e = _to_int_list(e, "e")

        if n != sum(y):
            raise ValueError(f"n deve ser igual a sum(y). n={n}, sum(y)={sum(y)}")
        if k != len(y):
            raise ValueError(f"k deve ser igual a len(y). k={k}, len(y)={len(y)}")
        if len(e) != k:
            raise ValueError(f"e deve ter tamanho k. len(e)={len(e)}, k={k}")
        if dst is None:
            raise ValueError("dst deve ser informado.")
        if len(dst) != k:
            raise ValueError(f"dst deve ter tamanho k. len(dst)={len(dst)}, k={k}")
        if A_in is None:
            raise ValueError("A_in deve ser informado.")
        if S is None:
            raise ValueError("S deve ser informado.")
        if A_out is None:
            raise ValueError("A_out deve ser informado.")
        if len(S) != k:
            raise ValueError(f"S deve ter tamanho k. len(S)={len(S)}, k={k}")
        if len(A_in) != k:
            raise ValueError(f"A_in deve ter tamanho k. len(A_in)={len(A_in)}, k={k}")
        if rho is None:
            raise ValueError("rho deve ser informado.")
        if isinstance(rho, torch.Tensor):
            if rho.numel() != 1:
                raise ValueError("rho deve ser um unico valor inteiro.")
            rho = rho.item()
        if isinstance(rho, bool):
            raise ValueError("rho deve ser um inteiro representando o numero total de arestas.")

        try:
            rho_value = float(rho)
        except (TypeError, ValueError) as exc:
            raise ValueError("rho deve ser um inteiro representando o numero total de arestas.") from exc
        if not rho_value.is_integer():
            raise ValueError("rho deve ser um inteiro representando o numero total de arestas.")

        rho = int(rho_value)
        if rho < 0:
            raise ValueError("rho deve ser maior ou igual a zero.")

        max_total_edges = n * (n - 1) // 2
        if rho > max_total_edges:
            raise ValueError(
                f"rho={rho} excede o maximo de arestas simples para {n} nos ({max_total_edges})."
            )
        if rho < sum(e):
            raise ValueError(f"rho={rho} deve ser maior ou igual a sum(e)={sum(e)}.")

        S = [_to_float_tensor(s, f"S[{label}]").view(-1) for label, s in enumerate(S)]
        for label, s in enumerate(S):
            if s.numel() == 0 or s.sum() <= 0:
                raise ValueError(f"S[{label}] deve conter pelo menos um peso positivo.")

        A_in = [
            _prepare_partition_matrix(A_in[label], num_partitions=len(S[label]), label=label)
            for label in range(k)
        ]

        A_out = _to_float_tensor(A_out, "A_out")
        if tuple(A_out.shape) != (k, k):
            raise ValueError(f"A_out deve ter shape ({k}, {k}); recebido {tuple(A_out.shape)}.")
        if not torch.all(torch.diagonal(A_out) == 0):
            raise ValueError("A diagonal da matriz A_out deve ser zero.")

        return y, e, A_in, A_out, S, rho

    def _sample_target_partition(self, A_in: torch.Tensor, src_partition: int) -> int:
        row = A_in[src_partition].float()
        if row.sum() <= 0:
            return src_partition
        return _sample_index(row)

    def _connect_isolated_nodes(
        self,
        graph: nk.Graph,
        y: torch.Tensor,
        edge_set: Optional[set[tuple[int, int]]] = None,
    ):
        isolated_nodes = [u for u in graph.iterNodes() if graph.degree(u) == 0]
        all_nodes = torch.tensor(list(graph.iterNodes()), dtype=torch.long)
        class_nodes = {
            int(label): all_nodes[y[all_nodes] == label]
            for label in torch.unique(y).tolist()
        }

        for u in isolated_nodes:
            if graph.degree(u) > 0:
                continue

            same_class = class_nodes.get(int(y[u]), torch.empty(0, dtype=torch.long))
            same_class = same_class[same_class != u]

            if len(same_class) > 0:
                degrees = torch.tensor([graph.degree(nd.item()) for nd in same_class]).float()
                v = same_class[_sample_index(degrees)].item()
            else:
                other_nodes = all_nodes[all_nodes != u]
                if len(other_nodes) == 0:
                    continue
                v = other_nodes[torch.randint(len(other_nodes), (1,))].item()

            if not _has_edge(graph, u, v, edge_set):
                _add_edge(graph, u, v, edge_set)

    def _estimate_interclass_matrix(self, graph: nk.Graph, y: torch.Tensor) -> torch.Tensor:
        valid_labels = y[y >= 0]
        if valid_labels.numel() == 0:
            return torch.zeros((0, 0), dtype=torch.float)

        matrix_size = int(valid_labels.max().item()) + 1
        A_out = torch.zeros((matrix_size, matrix_size), dtype=torch.float)

        for u, v in graph.iterEdges():
            cu = int(y[u])
            cv = int(y[v])
            if cu >= 0 and cv >= 0 and cu != cv:
                A_out[cu, cv] += 1
                A_out[cv, cu] += 1

        return A_out

    def _subgraphs_by_label(self, attributed_graph: AttributedGraph) -> dict[int, nk.Graph]:
        subgraphs = {}
        for label in torch.unique(attributed_graph.y).tolist():
            label = int(label)
            if label < 0:
                continue
            nodes = torch.where(attributed_graph.y == label)[0].tolist()
            subgraphs[label] = nk.graphtools.subgraphAndNeighborsFromNodes(attributed_graph.graph, nodes=nodes)
        return subgraphs

    def _count_heterogeneous_edges(self, graph: nk.Graph, y: torch.Tensor) -> int:
        return sum(1 for u, v in graph.iterEdges() if int(y[u]) >= 0 and int(y[v]) >= 0 and int(y[u]) != int(y[v]))

    def _interclass_sampling_context(self, graph: nk.Graph, y: torch.Tensor, A_out: torch.Tensor):
        valid_nodes = [u for u in graph.iterNodes() if int(y[u]) >= 0]
        labels = sorted({int(y[u]) for u in valid_nodes})
        if len(labels) < 2:
            return labels, {}, torch.zeros((0, 0), dtype=torch.float), 0

        class_nodes = {label: [u for u in valid_nodes if int(y[u]) == label] for label in labels}
        matrix_size = max(max(labels) + 1, A_out.shape[0])
        A_out_eff = torch.zeros((matrix_size, matrix_size), dtype=torch.float)
        A_out_eff[: A_out.shape[0], : A_out.shape[1]] = A_out.float()
        A_out_eff.fill_diagonal_(0)

        if A_out_eff.sum() <= 0:
            for src in labels:
                for tgt in labels:
                    if src != tgt:
                        A_out_eff[src, tgt] = 1

        return labels, class_nodes, A_out_eff, matrix_size

    def _add_partition_edges_until_count(
        self,
        graph: nk.Graph,
        partitions: list[list[int]],
        src_partition_weights: torch.Tensor,
        target_partition_weights: list[torch.Tensor],
        node_distributions: list[torch.Tensor],
        target_edges: int,
        edge_set: set[tuple[int, int]],
        max_attempts_factor: int = 200,
        error_context: str = "arestas homogeneas",
    ):
        if target_edges <= 0:
            return

        partition_arrays = [np.asarray(partition, dtype=int) for partition in partitions]
        src_probs = _numpy_probabilities(src_partition_weights)
        target_probs = [_numpy_probabilities(weights) for weights in target_partition_weights]
        node_probs = [
            _numpy_probabilities(weights) if len(partition_arrays[idx]) > 0 else None
            for idx, weights in enumerate(node_distributions)
        ]

        num_partitions = len(partition_arrays)
        added_edges = 0
        attempts = 0
        max_attempts = max(1_000, target_edges * max_attempts_factor)

        while added_edges < target_edges:
            missing_edges = target_edges - added_edges
            batch_size = min(_candidate_batch_size(missing_edges), max_attempts - attempts)
            if batch_size <= 0:
                raise RuntimeError(
                    f"Nao foi possivel criar {target_edges} {error_context}. "
                    "Verifique A_in, S, distribuicoes e densidade solicitada."
                )
            attempts += batch_size

            src_partitions = np.random.choice(num_partitions, size=batch_size, p=src_probs)
            tgt_partitions = np.empty(batch_size, dtype=int)
            for src_partition in np.unique(src_partitions):
                positions = np.flatnonzero(src_partitions == src_partition)
                tgt_partitions[positions] = np.random.choice(
                    num_partitions,
                    size=len(positions),
                    p=target_probs[src_partition],
                )

            u_nodes = np.empty(batch_size, dtype=int)
            v_nodes = np.empty(batch_size, dtype=int)
            for partition_id in np.unique(src_partitions):
                positions = np.flatnonzero(src_partitions == partition_id)
                nodes = partition_arrays[partition_id]
                u_nodes[positions] = np.random.choice(
                    nodes,
                    size=len(positions),
                    p=node_probs[partition_id],
                )

            for partition_id in np.unique(tgt_partitions):
                positions = np.flatnonzero(tgt_partitions == partition_id)
                nodes = partition_arrays[partition_id]
                v_nodes[positions] = np.random.choice(
                    nodes,
                    size=len(positions),
                    p=node_probs[partition_id],
                )

            for u, v in zip(u_nodes, v_nodes):
                if added_edges >= target_edges:
                    break
                if u == v or _has_edge(graph, u, v, edge_set):
                    continue
                _add_edge(graph, u, v, edge_set)
                added_edges += 1

    def _add_heterogeneous_edges_until_count(
        self,
        graph: nk.Graph,
        y: torch.Tensor,
        target_hetero_edges: int,
        A_out: torch.Tensor,
        max_attempts_factor: int = 200,
        error_context: str = "rho",
        edge_set: Optional[set[tuple[int, int]]] = None,
    ):
        current_hetero_edges = self._count_heterogeneous_edges(graph, y)
        if current_hetero_edges >= target_hetero_edges:
            return

        labels, class_nodes, A_out_eff, matrix_size = self._interclass_sampling_context(graph, y, A_out)
        if len(labels) < 2:
            return

        max_hetero_edges = 0
        for src_pos, src in enumerate(labels):
            for tgt in labels[src_pos + 1:]:
                max_hetero_edges += len(class_nodes[src]) * len(class_nodes[tgt])

        if target_hetero_edges > max_hetero_edges:
            raise ValueError(
                f"{error_context} exige {target_hetero_edges} arestas heterogeneas, "
                f"mas a capacidade maxima com os nos atuais e {max_hetero_edges}."
            )

        attempts = 0
        missing_edges = target_hetero_edges - current_hetero_edges
        max_attempts = max(1_000, missing_edges * max_attempts_factor)
        local_edge_set = edge_set
        if local_edge_set is None:
            local_edge_set = {_edge_key(u, v) for u, v in graph.iterEdges()}

        source_weights = torch.zeros(matrix_size, dtype=torch.float)
        for label in labels:
            source_weights[label] = A_out_eff[label, labels].sum()

        source_probs = _numpy_probabilities(source_weights)
        target_probs = {}
        for src in labels:
            target_weights = torch.zeros(matrix_size, dtype=torch.float)
            for label in labels:
                target_weights[label] = A_out_eff[src, label]
            if target_weights.sum() > 0:
                target_probs[src] = _numpy_probabilities(target_weights)

        class_node_arrays = {
            label: np.asarray(nodes, dtype=int)
            for label, nodes in class_nodes.items()
        }

        while current_hetero_edges < target_hetero_edges:
            missing_edges = target_hetero_edges - current_hetero_edges
            batch_size = min(_candidate_batch_size(missing_edges), max_attempts - attempts)
            if batch_size <= 0:
                raise RuntimeError(
                    f"Nao foi possivel atingir {error_context} com a matriz A_out informada. "
                    "Verifique se ha pares de classes disponiveis para novas arestas heterogeneas."
                )
            attempts += batch_size

            communities_u = np.random.choice(matrix_size, size=batch_size, p=source_probs)
            communities_v = np.empty(batch_size, dtype=int)
            valid_positions = np.ones(batch_size, dtype=bool)
            for community_u in np.unique(communities_u):
                positions = np.flatnonzero(communities_u == community_u)
                probs = target_probs.get(int(community_u))
                if probs is None:
                    valid_positions[positions] = False
                    continue
                communities_v[positions] = np.random.choice(
                    matrix_size,
                    size=len(positions),
                    p=probs,
                )

            valid_positions &= communities_u != communities_v
            if not valid_positions.any():
                continue

            u_nodes = np.empty(batch_size, dtype=int)
            v_nodes = np.empty(batch_size, dtype=int)
            for community in np.unique(communities_u[valid_positions]):
                positions = np.flatnonzero(valid_positions & (communities_u == community))
                candidates = class_node_arrays.get(int(community))
                if candidates is None or len(candidates) == 0:
                    valid_positions[positions] = False
                    continue
                u_nodes[positions] = candidates[np.random.randint(len(candidates), size=len(positions))]

            for community in np.unique(communities_v[valid_positions]):
                positions = np.flatnonzero(valid_positions & (communities_v == community))
                candidates = class_node_arrays.get(int(community))
                if candidates is None or len(candidates) == 0:
                    valid_positions[positions] = False
                    continue
                v_nodes[positions] = candidates[np.random.randint(len(candidates), size=len(positions))]

            for u, v in zip(u_nodes[valid_positions], v_nodes[valid_positions]):
                if current_hetero_edges >= target_hetero_edges:
                    break
                if u == v or _has_edge(graph, u, v, local_edge_set):
                    continue
                _add_edge(graph, u, v, local_edge_set)
                current_hetero_edges += 1

    def _add_heterogeneous_edges_until_density(
        self,
        graph: nk.Graph,
        y: torch.Tensor,
        target_density: float,
        A_out: torch.Tensor,
        max_attempts_factor: int = 200,
        edge_set: Optional[set[tuple[int, int]]] = None,
    ):
        actual_density = nk.graphtools.density(graph)
        if actual_density >= target_density:
            return

        max_possible_edges = graph.numberOfNodes() * (graph.numberOfNodes() - 1) // 2
        target_total_edges = math.ceil(target_density * max_possible_edges)
        missing_edges = target_total_edges - graph.numberOfEdges()
        target_hetero_edges = self._count_heterogeneous_edges(graph, y) + missing_edges
        self._add_heterogeneous_edges_until_count(
            graph=graph,
            y=y,
            target_hetero_edges=target_hetero_edges,
            A_out=A_out,
            max_attempts_factor=max_attempts_factor,
            error_context="rho",
            edge_set=edge_set,
        )

    def _generate_attributes(self, graph: nk.Graph, y: torch.Tensor, d: int, alpha_feat: float, alpha_topo: float):
        labels = sorted(int(label) for label in torch.unique(y[y >= 0]).tolist())
        k = len(labels)
        if d < k:
            raise ValueError(f"d deve ser >= numero de classes. d={d}, classes={k}")

        A = np.random.randn(d, k)
        Q, _ = np.linalg.qr(A)
        prototypes = Q[:, :k].T

        y_np = y.detach().cpu().numpy().astype(int)
        label_lookup = np.full(max(labels) + 1, -1, dtype=int)
        label_lookup[labels] = np.arange(k)
        prototype_indices = label_lookup[y_np[: graph.numberOfNodes()]]
        if np.any(prototype_indices < 0):
            raise ValueError("Todos os nos devem possuir rotulos validos para gerar atributos.")

        x0 = prototypes[prototype_indices] + alpha_feat * np.random.normal(
            size=(graph.numberOfNodes(), d)
        )

        neighbor_sum = np.zeros_like(x0)
        edges = list(graph.iterEdges())
        if edges:
            edge_array = np.asarray(edges, dtype=int)
            np.add.at(neighbor_sum, edge_array[:, 0], x0[edge_array[:, 1]])
            np.add.at(neighbor_sum, edge_array[:, 1], x0[edge_array[:, 0]])

        degrees = np.array([graph.degree(nd) for nd in graph.iterNodes()], dtype=float)
        degree_scale = np.zeros_like(degrees)
        mask = degrees > 0
        degree_scale[mask] = degrees[mask] ** (-0.5)

        smoothed = degree_scale[:, None] * neighbor_sum
        x = alpha_topo * x0 + (1 - alpha_topo) * smoothed
        return torch.tensor(x)

    def rank_based_matching(self, base_graph: AttributedGraph, mimic_graph: nk.Graph, noise_mean: float = 0, noise_std: float = 1):
        x = torch.zeros((base_graph.graph.numberOfNodes(), base_graph.x.size(1)))

        for label in torch.unique(base_graph.y):
            # Aplicar a máscara para pegar apenas os nós com o rótulo (label) atual
            mask = base_graph.y == label

            # Filtrando os nós pelo label
            nodes_with_label = torch.nonzero(mask).view(-1).tolist()

            # Calculando os graus apenas para os nós do label específico
            original_degrees = {node: base_graph.graph.degree(node) for node in nodes_with_label}
            original_degrees = dict(sorted(original_degrees.items(), key=lambda item: item[1]))

            mimic_degrees = {node: mimic_graph.degree(node) for node in nodes_with_label}
            mimic_degrees = dict(sorted(mimic_degrees.items(), key=lambda item: item[1]))

            transform_dict = {node_src: node_tgt for node_src, node_tgt in zip(original_degrees.keys(), mimic_degrees.keys())}

            for node_src, node_tgt in transform_dict.items():
                # x[node_tgt] = base_graph.x[node_src] + torch.rand((x.size(1))) * noise_std + noise_mean
                x[node_tgt] = base_graph.x[node_src]
        return x
    
    
    def mimic(self, 
              base_graph: AttributedGraph,
              num_nodes = None,
              base_community_detector = nk.community.PLM
              ):
        """
        Mimic the community structure and class relationships of another attributed graph.

        Args:
            base_graph (AttributedGraph): The original graph whose structure is to be reproduced.
            num_nodes (Optional[int|None]): Number of nodes (can be set to be greather than base_graph)
            

        This method:
            1. Detects communities (named as partitions) in each subgraph of the original graph.
            2. Computes degree-based connection probabilities for intra-partition links.
            3. Reconstructs each subgraph by probabilistic edge sampling.
            4. Adds inter-class (heterogeneous) edges between subgraphs to reflect the
               original heterogeneity pattern observed in `base_graph`.
        """

        # ensure that base_graph.y exists
        if not hasattr(base_graph, 'y'):
            raise ValueError('base_graph has no attribute base_graph.y')
        
        if num_nodes is not None and num_nodes < base_graph.graph.numberOfNodes():
            raise ValueError('num_nodes is lower than the number of nodes in base_graph')
        
        ############################
        ### STAGE 1: MIMIC STAGE ###
        ############################

        # Adding nodes 
        graph = nk.Graph(base_graph.graph.numberOfNodes(), weighted = False, directed = False)
        edge_set: set[tuple[int, int]] = set()
        y = base_graph.y
        interclass_matrix = self._estimate_interclass_matrix(base_graph.graph, y)

        for index, sub_g in base_graph.subgraphs.items():
            prt = graphPartition(subgraph=sub_g, base_community_detector=base_community_detector)
            deg = {node: sub_g.degree(node) for node in sub_g.iterNodes()}

            keys = list(deg.keys())

            # Somando 1 para considerar vértices que são isolados
            # weights = torch.tensor(list(deg.values()), dtype=torch.float) + 1
            weights = torch.tensor(list(deg.values()), dtype=torch.float)
            node_sampler = _CategoricalSampler(weights)

            partition_node_samplers = {}
            for partition_id, partition_nodes in prt.graph_partitions.items():
                partition_weights = torch.tensor(
                    [sub_g.degree(node) for node in partition_nodes],
                    dtype=torch.float,
                )
                partition_node_samplers[partition_id] = (
                    partition_nodes,
                    _CategoricalSampler(partition_weights),
                )

            target_partition_samplers = []
            for partition_id in range(prt._number_of_partitions()):
                row = prt.A_in[partition_id].float()
                if row.sum() <= 0:
                    row = torch.zeros_like(row)
                    row[partition_id] = 1
                target_partition_samplers.append(_CategoricalSampler(row))

            for _ in range(sub_g.numberOfEdges()):
                # Escolho um vértice de acordo com sua node degree distribution
                tmp_idx = node_sampler.sample()
                u = keys[tmp_idx]

                # Partição 
                src_partition = prt._get_partition(u)

                # Escolho em A_in algum vértice para ligar u
                tgt_partition = target_partition_samplers[src_partition].sample()

                # Seleciono algum vértice v de acordo com a probabilidade de ligação dos vértices em tgt_partition
                tmp_keys, tmp_sampler = partition_node_samplers[tgt_partition]
                tmp_idx = tmp_sampler.sample()
                v = tmp_keys[tmp_idx]

                if u != v and not _has_edge(graph, u, v, edge_set):
                    _add_edge(graph, u, v, edge_set)

        # Adding noise to the graph

        aim_density = nk.graphtools.density(base_graph.graph)

        self._add_heterogeneous_edges_until_density(
            graph=graph,
            y=y,
            target_density=aim_density,
            A_out=interclass_matrix,
            edge_set=edge_set,
        )


        self._connect_isolated_nodes(graph, y, edge_set=edge_set)

        # Feature generation stage (Cluster-Conditional Sampling)
        # For every node x_v in class y_i
        # x'_v' = \mu_c + \lambda \times (x_sample - \mu_c)
        # lambda controls the dispersion of the data
        # x = self._clone_features(base_graph = base_graph, mimic_graph = graph, y = y, lambda_param=lambda_param, window_size = window_size)
        # x = torch.rand((graph.numberOfNodes(), 2))
        x = self.rank_based_matching(base_graph = base_graph, mimic_graph=graph)
            
        attGraph = AttributedGraph(graph = graph, x = x, y = y )

        ##############################
        ### STAGE 2: AUGMENT STAGE ###
        ##############################

        if num_nodes is not None:
            nodes_to_add = num_nodes - base_graph.graph.numberOfNodes()

            # Gerando novas partições para o grafo clonado
            _global_partitions = {
                label: graphPartition(subgraph, base_community_detector)
                for label, subgraph in self._subgraphs_by_label(attGraph).items()
            }
            
            for _ in range(nodes_to_add):
                # added node
                v_i = attGraph.graph.addNode()

                # class of the added node
                class_values, class_counts = torch.unique(attGraph.y, return_counts=True)
                sampled_class_index = _sample_index(class_counts.float())
                c_i = int(class_values[sampled_class_index].item())

                # adding the selected class to y
                attGraph.y = torch.cat((attGraph.y, torch.tensor([c_i])))

                # chose a partition to add v_i
                src_partition = torch.multinomial(_global_partitions[c_i].partition_probabilities, num_samples=1, replacement=False)[0].item()

                # O que eu quero fazer aqui é que as features do novo v_i tenham a média das features da partição ao qual ela faz parte, somada a um ruído gaussiano que já foi definido lá em cima
                # tmp_x = torch.mean(attGraph.x[_global_partitions[c_i].graph_partitions[src_partition]], dim = 0, keepdim=True)
                tmp_x = torch.mean(attGraph.x[_global_partitions[c_i].graph_partitions[src_partition]], dim = 0, keepdim=True)

                attGraph.x = torch.cat([attGraph.x, tmp_x])

                #adding v_i to the partition
                _global_partitions[c_i].graph_partitions[src_partition].append(v_i)

                # number of nodes to connect v_i
                tmp_graph = nk.graphtools.subgraphAndNeighborsFromNodes(attGraph.graph, _global_partitions[c_i].graph_partitions[src_partition], includeOutNeighbors = False)
                n_nodes_v_i = int((2 * tmp_graph.numberOfEdges()) / tmp_graph.numberOfNodes())

                if n_nodes_v_i == 0:
                    n_nodes_v_i = 1

                for _ in range(n_nodes_v_i):
                    row = _global_partitions[c_i].A_in[src_partition]
                    if torch.sum(row) == 0:
                        tgt_partition = src_partition
                    else:
                        tgt_partition = _sample_index(row)

                    continue_add = True
                    while continue_add:
                        # Seleciona o vértice a ser ligado a partir do degree
                        candidate_degrees = torch.tensor(
                            [attGraph.graph.degree(nd) for nd in _global_partitions[c_i].graph_partitions[tgt_partition]]
                        ).float()
                        v_j = _sample_index(candidate_degrees)
                        v_j = _global_partitions[c_i].graph_partitions[tgt_partition][v_j]
                        if v_i != v_j:
                            graph.addEdge(v_i,v_j)
                            _global_partitions[c_i].A_in[src_partition, tgt_partition] += 1
                            _global_partitions[c_i].A_in[tgt_partition, src_partition] += 1
                            continue_add = False

            # Adicionar Ruído
            self._add_heterogeneous_edges_until_density(
                graph=attGraph.graph,
                y=attGraph.y,
                target_density=nk.graphtools.density(base_graph.graph),
                A_out=interclass_matrix,
            )

        # REMOVE IF THERE WAS AN ERROR
        attGraph.create_subgraphs()
        return attGraph
    
    def _run_synco(
        self,
        n: int,
        y: list[int],
        k: int,
        e: list[int],
        A_in: list[torch.tensor],
        dst: list[str],
        rho: int,
        A_out: torch.tensor = None,
        S: list[torch.tensor] = None,
        alpha_feat=1,
        d: int = 10,
        alpha_topo=0.8,
        alpha_powerlaw=2,
        std_distribution=None,
        mu_distribution=None,
        eps=1e-6,
        build_subgraphs: bool = True,
    ):
        '''

        :param n: number of nodes for the graph
        :type n: int
        :param y: list of number of nodes for each class, i.e, y[1] is a int that represents the number of nodes in class 1
        :type y: list[int]
        :param k: number of classes. len(y) == k
        :type k: int
        :param e: list of number of homogeneous edges for each class.
        :type e: list[int]
        :param A_in: list of torch.tensor elements. Each element represents the connection probabilities between sub-communities inside one class.
        :type A_in: list[torch.tensor]
        :param dst: list of strings that represents the node degree distributions of each class.
        :type dst: list[str]
        :param rho: total number of edges in the graph.
        :type rho: int
        :param A_out: torch.tensor that represents the edge allocation weights between classes.
        :type A_out: torch.tensor
        :param S: list of tensors representing the relative size of each sub-community.
        :type S: list[torch.tensor]
        :param d: dimension of the feature matrix X.
        :type d: int
        :param alpha_feat: Gaussian feature perturbation strength.
        :type alpha_feat: float
        :param alpha_topo: trade-off between class-conditioned attributes and topological smoothing.
        :type alpha_topo: float
        :param build_subgraphs: if False, skips AttributedGraph subgraph construction for faster bulk generation.
        :type build_subgraphs: bool
        '''

        y, e, A_in, A_out, S, rho = self._validate_generate_inputs(
            n=n,
            y=y,
            k=k,
            e=e,
            A_in=A_in,
            dst=dst,
            rho=rho,
            A_out=A_out,
            S=S,
        )

        tmp_graph = nk.Graph(n, weighted = False, directed = False)
        tmp_y = torch.tensor([i for i, v in enumerate(y) for _ in range(v)], dtype=torch.long)
        edge_set: set[tuple[int, int]] = set()

        # Atribuir cada um dos vértices de cada grupo a cada partição

        out = split_by_class_and_partitions(n = n, y = y, parts=[x.tolist() for x in S])
        for label, label_num_nodes in enumerate(y):
            max_edges = label_num_nodes * (label_num_nodes - 1) // 2
            if e[label] > max_edges:
                raise ValueError(
                    f"e[{label}]={e[label]} excede o maximo de arestas simples na classe "
                    f"com {label_num_nodes} nos ({max_edges})."
                )

            distributions = []
            partition_sizes = torch.tensor([len(partition) for partition in out[label]], dtype=torch.float)
            for ptt in range(len(S[label])):
                if len(out[label][ptt]) == 0:
                    distributions.append(torch.empty(0))
                    continue
                distributions.append(
                    distribution_transform(
                        dst=dst[label],
                        n=len(out[label][ptt]),
                        powerlaw_exponent=alpha_powerlaw,
                        mu=mu_distribution,
                        normal_std=std_distribution,
                        eps=eps,
                    )
                )

            src_partition_weights = S[label].float().clone()
            src_partition_weights[partition_sizes == 0] = 0

            if src_partition_weights.sum() <= 0 and e[label] > 0:
                raise ValueError(f"A classe {label} nao possui subcomunidades com nos.")

            target_partition_weights = []
            for ptt in range(len(S[label])):
                target_weights = A_in[label][ptt].float().clone()
                target_weights[partition_sizes == 0] = 0
                if target_weights.sum() <= 0:
                    target_weights = src_partition_weights.clone()
                target_partition_weights.append(target_weights)

            self._add_partition_edges_until_count(
                graph=tmp_graph,
                partitions=out[label],
                src_partition_weights=src_partition_weights,
                target_partition_weights=target_partition_weights,
                node_distributions=distributions,
                target_edges=e[label],
                edge_set=edge_set,
                error_context=f"arestas homogeneas na classe {label}",
            )

        self._connect_isolated_nodes(tmp_graph, tmp_y, edge_set=edge_set)
        if tmp_graph.numberOfEdges() > rho:
            warnings.warn(
                f"Ao conectar vertices isolados, o grafo passou a ter {tmp_graph.numberOfEdges()} arestas, "
                f"mas rho={rho}. O grafo sera retornado com mais arestas que o valor solicitado.",
                RuntimeWarning,
            )

        missing_edges = max(0, rho - tmp_graph.numberOfEdges())
        target_hetero_edges = self._count_heterogeneous_edges(tmp_graph, tmp_y) + missing_edges
        self._add_heterogeneous_edges_until_count(
            graph=tmp_graph,
            y=tmp_y,
            target_hetero_edges=target_hetero_edges,
            A_out=A_out,
            edge_set=edge_set,
        )
        if tmp_graph.numberOfEdges() != rho:
            warnings.warn(
                f"O grafo gerado possui {tmp_graph.numberOfEdges()} arestas, mas rho={rho}.",
                RuntimeWarning,
            )

        # TESTE REMOVENDO OS VÉRTICES E REINDEXANDO
        tmp_graph = compact_graph_ids(tmp_graph)

        mask = torch.ones_like(tmp_y, dtype = torch.bool)
        mask[tmp_y == -1] = False
        tmp_y = tmp_y[mask]

        x = self._generate_attributes(
            graph=tmp_graph,
            y=tmp_y,
            d=d,
            alpha_feat=alpha_feat,
            alpha_topo=alpha_topo,
        )

        return  AttributedGraph(graph = tmp_graph, y = tmp_y, x = x, build_subgraphs=build_subgraphs)

    def generate(
        self,
        n: int,
        y: list[int],
        k: int,
        e: list[int],
        A_in: list[torch.tensor],
        dst: list[str],
        rho: int,
        A_out: torch.tensor = None,
        S: list[torch.tensor] = None,
        alpha_feat=1,
        d: int = 10,
        alpha_topo=0.8,
        alpha_powerlaw=2,
        std_distribution=None,
        mu_distribution=None,
        eps=1e-6,
        build_subgraphs: bool = True,
    ):
        return self._run_synco(
            n=n,
            y=y,
            k=k,
            e=e,
            A_in=A_in,
            dst=dst,
            rho=rho,
            A_out=A_out,
            S=S,
            alpha_feat=alpha_feat,
            d=d,
            alpha_topo=alpha_topo,
            alpha_powerlaw=alpha_powerlaw,
            std_distribution=std_distribution,
            mu_distribution=mu_distribution,
            eps=eps,
            build_subgraphs=build_subgraphs,
        )
    

    def prepare_inductive_mimic_data(
        self,
        base_graph: AttributedGraph,
        val_ratio: float = 0.2,
        stratified: bool = True,
        base_community_detector = nk.community.PLM,
        mimic_num_nodes: Optional[int] = None,
        seed: Optional[int] = None,
        use_subgraph_and_neighbors: bool = False,
        include_out_neighbors: bool = False
    ) -> Dict[str, Any]:
        """
        Prepara dados para avaliação indutiva de consistência/diversidade em mimicagem.

        O objetivo é:
          1) Splitar os nós do grafo original em treino/val (opcionalmente estratificado).
          2) Construir um grafo SOMENTE de treino, removendo nós de validação.
             - Como solicitado, usa nk.graphtools.subgraphAndNeighborsFromNodes (opcional),
               mas GARANTE que nós de validação não permaneçam no grafo final.
          3) Reindexar os nós do grafo de treino para 0..n_train-1 (compactação).
          4) Retornar:
             - train_graph: AttributedGraph apenas com treino
             - mimic_graph: AttributedGraph mimicado a partir de train_graph
             - split: índices/máscaras originais
             - y_val: rótulos da validação (originais)
             - mapas old<->new do treino (para rastrear ids se quiser)

        Parâmetros:
          base_graph: AttributedGraph original (deve ter .graph (nk.Graph), .x (Tensor), .y (Tensor))
          val_ratio: fração de nós para validação (0<val_ratio<1)
          stratified: se True, split por classe (mantém proporções em val)
          base_community_detector: detector usado no mimic (PLM por padrão)
          mimic_num_nodes: se definido, permite augment no mimic do train_graph
          seed: seed local do split (se None usa self.seed se existir)
          use_subgraph_and_neighbors: se True, usa subgraphAndNeighborsFromNodes como você pediu
          include_out_neighbors: parâmetro do NetworKit (mantive exposto)

        Retorna:
          dict com chaves:
            - "train_graph": AttributedGraph (somente treino, ids compactados)
            - "mimic_graph": AttributedGraph (mimicado a partir do train_graph)
            - "split": dict com train_idx, val_idx, train_mask, val_mask (no espaço original)
            - "y_val": Tensor com rótulos de validação (na ordem de val_idx)
            - "x_val": Tensor com features de validação (na ordem de val_idx) [útil no pipeline]
            - "id_maps": dict com old2new_train e new2old_train
        """

        # -----------------------------
        # 0) Validações e seed do split
        # -----------------------------
        if not hasattr(base_graph, "y") or base_graph.y is None:
            raise ValueError("base_graph precisa ter atributo .y (Tensor com rótulos).")
        if not hasattr(base_graph, "x") or base_graph.x is None:
            raise ValueError("base_graph precisa ter atributo .x (Tensor de features).")
        if not hasattr(base_graph, "graph") or base_graph.graph is None:
            raise ValueError("base_graph precisa ter atributo .graph (nk.Graph).")

        n = base_graph.graph.numberOfNodes()
        if n != base_graph.y.numel():
            raise ValueError(f"Inconsistência: n_nodes={n} mas y tem {base_graph.y.numel()} elementos.")
        if base_graph.x.size(0) != n:
            raise ValueError(f"Inconsistência: n_nodes={n} mas x tem {base_graph.x.size(0)} linhas.")

        if not (0.0 < val_ratio < 1.0):
            raise ValueError("val_ratio precisa estar entre 0 e 1 (ex.: 0.2).")

        split_seed = seed if seed is not None else getattr(self, "seed", None)
        g = torch.Generator()
        if split_seed is not None:
            g.manual_seed(int(split_seed))

        # ------------------------------------------
        # 1) Split treino/val no espaço ORIGINAL (nós)
        # ------------------------------------------
        y = base_graph.y.detach().cpu()
        all_idx = torch.arange(n, dtype=torch.long)

        if stratified:
            # Split estratificado: para cada classe, sorteia uma fração para validação
            val_indices = []
            train_indices = []

            classes = torch.unique(y)
            for c in classes:
                idx_c = all_idx[y == c]
                if idx_c.numel() == 0:
                    continue

                perm = idx_c[torch.randperm(idx_c.numel(), generator=g)]
                n_val_c = max(1, int(round(idx_c.numel() * val_ratio))) if idx_c.numel() > 1 else 0

                val_c = perm[:n_val_c]
                train_c = perm[n_val_c:]

                # Se por algum motivo train ficou vazio numa classe muito pequena,
                # empurra 1 elemento de volta pro treino (pra não sumir a classe)
                if train_c.numel() == 0 and val_c.numel() > 0:
                    train_c = val_c[:1]
                    val_c = val_c[1:]

                val_indices.append(val_c)
                train_indices.append(train_c)

            val_idx = torch.cat(val_indices) if len(val_indices) else torch.empty(0, dtype=torch.long)
            train_idx = torch.cat(train_indices) if len(train_indices) else torch.empty(0, dtype=torch.long)

            # embaralha a ordem final (não é obrigatório, mas deixa “neutro”)
            if val_idx.numel() > 0:
                val_idx = val_idx[torch.randperm(val_idx.numel(), generator=g)]
            if train_idx.numel() > 0:
                train_idx = train_idx[torch.randperm(train_idx.numel(), generator=g)]

        else:
            perm = all_idx[torch.randperm(n, generator=g)]
            n_val = int(round(n * val_ratio))
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]

        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True

        # Guardar y_val e x_val no formato que você vai usar depois no pipeline
        y_val = base_graph.y[val_idx].clone()
        x_val = base_graph.x[val_idx].clone()

        # ---------------------------------------------------------
        # 2) Construir grafo SOMENTE treino (removendo validação)
        #    (usando subgraphAndNeighborsFromNodes, mas garantindo separação)
        # ---------------------------------------------------------
        train_nodes_list = train_idx.detach().cpu().tolist()
        val_nodes_set = set(val_idx.detach().cpu().tolist())

        if use_subgraph_and_neighbors:
            # Isso pode trazer vizinhos que não estão em train_nodes_list.
            # Como você quer remover validação, faremos:
            #   a) gera subgrafo + vizinhos
            #   b) remove explicitamente qualquer nó que seja validação
            g_tmp = nk.graphtools.subgraphAndNeighborsFromNodes(
                base_graph.graph,
                train_nodes_list,
                includeOutNeighbors=include_out_neighbors
            )
        else:
            # alternativa mais “pura” seria subgraphFromNodes (sem vizinhos)
            g_tmp = nk.graphtools.subgraphFromNodes(base_graph.graph, train_nodes_list)

        # Remover nós de validação caso tenham entrado como vizinhos
        nodes_to_remove = [u for u in g_tmp.iterNodes() if u in val_nodes_set]
        for u in nodes_to_remove:
            g_tmp.removeNode(u)

        # Também é possível que tenham entrado nós que não são treino (ex.: vizinhos de treino)
        # Se você quer *estritamente* só treino, removemos qualquer nó que não esteja em train_mask.
        train_nodes_set = set(train_nodes_list)
        extra_nodes = [u for u in g_tmp.iterNodes() if u not in train_nodes_set]
        for u in extra_nodes:
            g_tmp.removeNode(u)

        # ---------------------------------------------------------
        # 3) Compactar ids do grafo de treino e alinhar x/y
        # ---------------------------------------------------------
        # Vamos criar mapas old<->new e um grafo novo com ids 0..n_train-1
        old_nodes = list(g_tmp.iterNodes())  # ids antigos (do grafo original) que permaneceram
        old2new = {old: new for new, old in enumerate(old_nodes)}
        new2old = old_nodes

        g_train = nk.Graph(len(old_nodes), weighted=g_tmp.isWeighted(), directed=g_tmp.isDirected())
        for u, v in g_tmp.iterEdges():
            g_train.addEdge(old2new[u], old2new[v])

        # Reindexa features e labels do treino usando new2old
        x_train = base_graph.x[torch.tensor(new2old, dtype=torch.long)].clone()
        y_train = base_graph.y[torch.tensor(new2old, dtype=torch.long)].clone()

        train_graph = AttributedGraph(graph=g_train, x=x_train, y=y_train)
        train_graph.create_subgraphs()  # garante subgraphs consistentes (se sua classe usa isso)

        # ---------------------------------------------------------
        # 4) Mimicar/clonar SOMENTE o grafo de treino
        # ---------------------------------------------------------

        final_num_nodes = (
            int(mimic_num_nodes * train_graph.graph.numberOfNodes())
            if mimic_num_nodes is not None
            else None
        )
        mimic_graph = self.mimic(
            base_graph=train_graph,
            num_nodes=final_num_nodes,
            base_community_detector=base_community_detector,
        )

        # ---------------------------------------------------------
        # 5) Empacotar retornos para seu pipeline de experimentos
        # ---------------------------------------------------------
        out = {
            "train_graph": train_graph,
            "mimic_graph": mimic_graph,
            "split": {
                "train_idx": train_idx.clone(),   # no espaço ORIGINAL do base_graph
                "val_idx": val_idx.clone(),       # no espaço ORIGINAL do base_graph
                "train_mask": train_mask.clone(),
                "val_mask": val_mask.clone(),
                "val_ratio": float(val_ratio),
                "stratified": bool(stratified),
                "seed": split_seed,
            },
            "y_val": y_val,
            "x_val": x_val,
            "id_maps": {
                "old2new_train": old2new,   # ids originais -> ids compactados no train_graph
                "new2old_train": new2old,   # ids compactados -> ids originais
            }
        }

        return out


SCAttGenerator = SynCoGenerator

__all__ = ["SynCoGenerator", "SCAttGenerator"]
