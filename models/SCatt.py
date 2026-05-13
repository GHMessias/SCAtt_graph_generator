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

def compact_graph_ids(g: nk.Graph):
    """
    Retorna:
      g2: novo grafo com nós 0..n-1
      old2new: dict {old_id -> new_id} apenas para nós existentes
      new2old: list onde new2old[new_id] = old_id
    """
    # nós que "existem" (iterNodes() já pula os removidos)
    nodes = list(g.iterNodes())
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

def distribution_transform(d: str, n: int, alpha: float = 2.0, seed=None,
                           mu=None, sigma=None, eps: float = 1e-6):

    ranks = torch.arange(1, n + 1, dtype=torch.float64)

    if d == 'power_law':
        w = ranks ** (-alpha)

    elif d == 'normal':
        # centro e dispersão "compatíveis" com ranks 1..n
        if mu is None:
            mu = (n + 1) / 2.0
        if sigma is None:
            sigma = max(n / 6.0, 1.0)  # regra prática: cobre bem o intervalo

        w = torch.exp(- (ranks - mu) ** 2 / (2 * sigma ** 2))

    elif d == 'uniform':
        w = torch.ones(n, dtype=torch.float64)

    else:
        raise ValueError(f"Distribuição desconhecida: {d}")

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


def _prepare_partition_matrix(matrix, num_partitions: int, label: int) -> torch.Tensor:
    matrix = _to_float_tensor(matrix, f"C[{label}]")

    if matrix.ndim == 1 and num_partitions == 1 and matrix.numel() == 1:
        matrix = matrix.reshape(1, 1)

    if matrix.ndim != 2 or tuple(matrix.shape) != (num_partitions, num_partitions):
        raise ValueError(
            f"C[{label}] deve ter shape ({num_partitions}, {num_partitions}); "
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
        for index,value in self.graph_partitions.items():
            if node in value:
                return index
            
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

    def compute_partition_heterogeneity(self):
        '''
        Essa função calcula a probabilidade de uma aresta ser intra-partição e inter-partição, para cada uma das partições. A saída dela é uma matriz C onde C_{i,j} representa a probabilidade da partição i ser ligada na partição j
        
        :param self: Description
        '''

        self.C = torch.zeros((self._number_of_partitions(), self._number_of_partitions()))

        for u,v in self.subgraph.iterEdges():
            # print(self._get_partition(u), self._get_partition(v))
            self.C[self._get_partition(u), self._get_partition(v)] += 1
            self.C[self._get_partition(v), self._get_partition(u)] += 1
        return

    def compute_partition_probability(self):
        self.partition_probabilities = torch.tensor([len(partition) for partition in self.graph_partitions.values()])/self.subgraph.numberOfNodes()
        return
    
class SCAttGenerator(BaseGenerator):
    def __init__(self, seed: int | None = None):
        super().__init__(name='SCAtt', supports_mimic=True, supports_augment=True)

        self.seed = seed
        if seed is not None:
            self._set_seed(seed)

    def _set_seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _validate_generate_inputs(self, num_nodes, y, k, e, C, d, rho, N, M):
        y = _to_int_list(y, "y")
        e = _to_int_list(e, "e")

        if num_nodes != sum(y):
            raise ValueError(f"num_nodes deve ser igual a sum(y). num_nodes={num_nodes}, sum(y)={sum(y)}")
        if k != len(y):
            raise ValueError(f"k deve ser igual a len(y). k={k}, len(y)={len(y)}")
        if len(e) != k:
            raise ValueError(f"e deve ter tamanho k. len(e)={len(e)}, k={k}")
        if len(d) != k:
            raise ValueError(f"d deve ter tamanho k. len(d)={len(d)}, k={k}")
        if len(M) != k:
            raise ValueError(f"M deve ter tamanho k. len(M)={len(M)}, k={k}")
        if len(C) != k:
            raise ValueError(f"C deve ter tamanho k. len(C)={len(C)}, k={k}")
        if rho < 0:
            raise ValueError("rho deve ser maior ou igual a zero.")

        M = [_to_float_tensor(m, f"M[{label}]").view(-1) for label, m in enumerate(M)]
        for label, m in enumerate(M):
            if m.numel() == 0 or m.sum() <= 0:
                raise ValueError(f"M[{label}] deve conter pelo menos um peso positivo.")

        C = [
            _prepare_partition_matrix(C[label], num_partitions=len(M[label]), label=label)
            for label in range(k)
        ]

        N = _to_float_tensor(N, "N")
        if tuple(N.shape) != (k, k):
            raise ValueError(f"N deve ter shape ({k}, {k}); recebido {tuple(N.shape)}.")
        if not torch.all(torch.diagonal(N) == 0):
            raise ValueError("A diagonal da matriz N deve ser zero.")

        return y, e, C, N, M

    def _sample_target_partition(self, C: torch.Tensor, src_partition: int) -> int:
        row = C[src_partition].float()
        if row.sum() <= 0:
            return src_partition
        return _sample_index(row)

    def _connect_isolated_nodes(self, graph: nk.Graph, y: torch.Tensor):
        isolated_nodes = [u for u in graph.iterNodes() if graph.degree(u) == 0]

        for u in isolated_nodes:
            all_nodes = torch.tensor(list(graph.iterNodes()), dtype=torch.long)
            same_class = all_nodes[(y[all_nodes] == y[u]) & (all_nodes != u)]

            if len(same_class) > 0:
                degrees = torch.tensor([graph.degree(nd.item()) for nd in same_class]).float()
                v = same_class[_sample_index(degrees)].item()
            else:
                fallback = all_nodes[all_nodes != u]
                if len(fallback) == 0:
                    continue
                v = fallback[torch.randint(len(fallback), (1,))].item()

            if not graph.hasEdge(u, v):
                graph.addEdge(u, v)

    def _estimate_interclass_matrix(self, graph: nk.Graph, y: torch.Tensor) -> torch.Tensor:
        valid_labels = y[y >= 0]
        if valid_labels.numel() == 0:
            return torch.zeros((0, 0), dtype=torch.float)

        matrix_size = int(valid_labels.max().item()) + 1
        N = torch.zeros((matrix_size, matrix_size), dtype=torch.float)

        for u, v in graph.iterEdges():
            cu = int(y[u])
            cv = int(y[v])
            if cu >= 0 and cv >= 0 and cu != cv:
                N[cu, cv] += 1
                N[cv, cu] += 1

        return N

    def _subgraphs_by_label(self, attributed_graph: AttributedGraph) -> dict[int, nk.Graph]:
        subgraphs = {}
        for label in torch.unique(attributed_graph.y).tolist():
            label = int(label)
            if label < 0:
                continue
            nodes = torch.where(attributed_graph.y == label)[0].tolist()
            subgraphs[label] = nk.graphtools.subgraphAndNeighborsFromNodes(attributed_graph.graph, nodes=nodes)
        return subgraphs

    def _add_heterogeneous_edges_until_density(
        self,
        graph: nk.Graph,
        y: torch.Tensor,
        target_density: float,
        N: torch.Tensor,
        max_attempts_factor: int = 200,
    ):
        actual_density = nk.graphtools.density(graph)
        if actual_density >= target_density:
            return

        valid_nodes = [u for u in graph.iterNodes() if int(y[u]) >= 0]
        labels = sorted({int(y[u]) for u in valid_nodes})
        if len(labels) < 2:
            return

        class_nodes = {label: [u for u in valid_nodes if int(y[u]) == label] for label in labels}
        matrix_size = max(max(labels) + 1, N.shape[0])
        N_eff = torch.zeros((matrix_size, matrix_size), dtype=torch.float)
        N_eff[: N.shape[0], : N.shape[1]] = N.float()
        N_eff.fill_diagonal_(0)

        if N_eff.sum() <= 0:
            for src in labels:
                for tgt in labels:
                    if src != tgt:
                        N_eff[src, tgt] = 1

        attempts = 0
        max_attempts = max(1, graph.upperEdgeIdBound() + graph.numberOfNodes()) * max_attempts_factor

        while nk.graphtools.density(graph) < target_density:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    "Nao foi possivel atingir rho com a matriz N informada. "
                    "Verifique se ha pares de classes disponiveis para novas arestas heterogeneas."
                )

            source_weights = torch.zeros(matrix_size, dtype=torch.float)
            for label in labels:
                source_weights[label] = N_eff[label].sum()

            community_u = _sample_index(source_weights)
            target_weights = torch.zeros(matrix_size, dtype=torch.float)
            for label in labels:
                target_weights[label] = N_eff[community_u, label]

            if target_weights.sum() <= 0:
                continue

            community_v = _sample_index(target_weights)
            if community_u == community_v:
                continue

            u_candidates = class_nodes.get(community_u, [])
            v_candidates = class_nodes.get(community_v, [])
            if not u_candidates or not v_candidates:
                continue

            u = u_candidates[torch.randint(len(u_candidates), (1,)).item()]
            v = v_candidates[torch.randint(len(v_candidates), (1,)).item()]

            if u != v and not graph.hasEdge(u, v):
                graph.addEdge(u, v)

    def _generate_attributes(self, graph: nk.Graph, y: torch.Tensor, dimensions: int, sigma: float, alpha: float):
        labels = sorted(int(label) for label in torch.unique(y[y >= 0]).tolist())
        k = len(labels)
        if dimensions < k:
            raise ValueError(f"dimensions deve ser >= numero de classes. dimensions={dimensions}, classes={k}")

        A = np.random.randn(dimensions, k)
        Q, _ = np.linalg.qr(A)
        prototypes = Q[:, :k].T
        label_to_prototype = {label: idx for idx, label in enumerate(labels)}

        x0 = np.zeros(shape=(graph.numberOfNodes(), dimensions))
        for nd in range(graph.numberOfNodes()):
            prototype_idx = label_to_prototype[int(y[nd])]
            x0[nd] = prototypes[prototype_idx] + sigma * np.random.normal(size=(dimensions))

        neighbor_sum = np.zeros_like(x0)
        for u, v in graph.iterEdges():
            neighbor_sum[u] += x0[v]
            neighbor_sum[v] += x0[u]

        degrees = np.array([graph.degree(nd) for nd in graph.iterNodes()], dtype=float)
        degree_scale = np.zeros_like(degrees)
        mask = degrees > 0
        degree_scale[mask] = degrees[mask] ** (-0.5)

        smoothed = degree_scale[:, None] * neighbor_sum
        x = alpha * x0 + (1 - alpha) * smoothed
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
        y = base_graph.y
        interclass_matrix = self._estimate_interclass_matrix(base_graph.graph, y)

        for index, sub_g in base_graph.subgraphs.items():
            prt = graphPartition(subgraph=sub_g, base_community_detector=base_community_detector)
            deg = {node: sub_g.degree(node) for node in sub_g.iterNodes()}

            keys = list(deg.keys())

            # Somando 1 para considerar vértices que são isolados
            # weights = torch.tensor(list(deg.values()), dtype=torch.float) + 1
            weights = torch.tensor(list(deg.values()), dtype=torch.float)

            for _ in range(sub_g.numberOfEdges()):
                # Escolho um vértice de acordo com sua node degree distribution
                tmp_idx = _sample_index(weights)
                u = keys[tmp_idx]

                # Partição 
                src_partition = prt._get_partition(u)

                # Escolho em C algum vértice para ligar u
                tgt_partition = self._sample_target_partition(prt.C, src_partition)

                # Seleciono algum vértice v de acordo com a probabilidade de ligação dos vértices em tgt_partition
                tmp_deg = {node:sub_g.degree(node) for node in prt.graph_partitions[tgt_partition]}
                tmp_keys = list(tmp_deg.keys())
                tmp_weights = torch.tensor(list(tmp_deg.values()), dtype=torch.float)

                tmp_idx = _sample_index(tmp_weights)
                v = tmp_keys[tmp_idx]

                if u != v:
                    graph.addEdge(u,v)

        # Adding noise to the graph

        aim_density = nk.graphtools.density(base_graph.graph)

        self._add_heterogeneous_edges_until_density(
            graph=graph,
            y=y,
            target_density=aim_density,
            N=interclass_matrix,
        )


        self._connect_isolated_nodes(graph, y)

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
                    row = _global_partitions[c_i].C[src_partition]
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
                            _global_partitions[c_i].C[src_partition, tgt_partition] += 1
                            _global_partitions[c_i].C[tgt_partition, src_partition] += 1
                            continue_add = False

            # Adicionar Ruído
            self._add_heterogeneous_edges_until_density(
                graph=attGraph.graph,
                y=attGraph.y,
                target_density=nk.graphtools.density(base_graph.graph),
                N=interclass_matrix,
            )

        # REMOVE IF THERE WAS AN ERROR
        attGraph.create_subgraphs()
        return attGraph
    
    def _run_scatt(self, num_nodes: int, y: list[int], k: int, e: list[int], C: list[torch.tensor], d: list[str], rho: float, N: torch.tensor, M: list[torch.tensor], sigma = 1, dimensions = 10, alpha = 0.8, alpha_powerlaw = 2, sigma_distribution = None, mu_distribution = None, eps = 1e-6):
        '''

        :param num_nodes: number of nodes for the graph
        :type num_nodes: int
        :param y: list of number of nodes for each class, i.e, y[1] is a int that represents the number of nodes in class 1
        :type y: list[int]
        :param k: number of classes. len(y) == k
        :type k: int
        :param e: list of number of edges for each class. 
        :type e: list[int]
        :param C: list of torch.tensor elements. Each element in C represents the distribution probability for connection between the partitions. As every class has different number of partitions, C can have tensors with different shapes.
        :type C: list[torch.tensor]
        :param d: list of strings that represents the distributions of each class. The available distributions are: TODO
        :type d: list[str]
        :param rho: density value of the model. The density will guide the number of total edges, since the edges added to reach the target density are all heterogeneous (between different classes).
        :type density: float
        :param N: torch.tensor that represents the probability assignment of edges between classes.
        :type N: torch.tensor
        :param M: torch.tensor that represents the number of element in each partition
        :type: list[torch.tensor]
        '''

        y, e, C, N, M = self._validate_generate_inputs(
            num_nodes=num_nodes,
            y=y,
            k=k,
            e=e,
            C=C,
            d=d,
            rho=rho,
            N=N,
            M=M,
        )

        tmp_graph = nk.Graph(num_nodes, weighted = False, directed = False)
        tmp_y = torch.tensor([i for i, v in enumerate(y) for _ in range(v)], dtype=torch.long)

        # Atribuir cada um dos vértices de cada grupo a cada partição

        out = split_by_class_and_partitions(n = num_nodes, y = y, parts=[x.tolist() for x in M])
        for label, label_num_nodes in enumerate(y):
            max_edges = label_num_nodes * (label_num_nodes - 1) // 2
            if e[label] > max_edges:
                raise ValueError(
                    f"e[{label}]={e[label]} excede o maximo de arestas simples na classe "
                    f"com {label_num_nodes} nos ({max_edges})."
                )

            distributions = []
            partition_sizes = torch.tensor([len(partition) for partition in out[label]], dtype=torch.float)
            for ptt in range(len(M[label])):
                if len(out[label][ptt]) == 0:
                    distributions.append(torch.empty(0))
                    continue
                distributions.append(
                    distribution_transform(
                        d=d[label],
                        n=len(out[label][ptt]),
                        alpha=alpha_powerlaw,
                        mu=mu_distribution,
                        sigma=sigma_distribution,
                        eps=eps,
                    )
                )

            added_edges = 0
            attempts = 0
            max_attempts = max(1_000, e[label] * 200)
            src_partition_weights = M[label].float().clone()
            src_partition_weights[partition_sizes == 0] = 0

            if src_partition_weights.sum() <= 0 and e[label] > 0:
                raise ValueError(f"A classe {label} nao possui subcomunidades com nos.")

            while added_edges < e[label]:
                attempts += 1
                if attempts > max_attempts:
                    raise RuntimeError(
                        f"Nao foi possivel criar e[{label}]={e[label]} arestas na classe {label}. "
                        "Verifique C, M e o numero de nos por subcomunidade."
                    )

                src_partition = _sample_index(src_partition_weights)
                target_weights = C[label][src_partition].float().clone()
                target_weights[partition_sizes == 0] = 0

                if target_weights.sum() <= 0:
                    target_weights = src_partition_weights.clone()

                tgt_partition = _sample_index(target_weights)

                # seleciona u e v em src_partition
                u_index = _sample_index(distributions[src_partition])
                v_index = _sample_index(distributions[tgt_partition])
                u = out[label][src_partition][u_index]
                v = out[label][tgt_partition][v_index]

                # verificar se u é igual a v
                while u == v:
                    if len(out[label][src_partition]) == 1 and src_partition == tgt_partition:
                        break
                    u_index = _sample_index(distributions[src_partition])
                    v_index = _sample_index(distributions[tgt_partition])
                    u = out[label][src_partition][u_index]
                    v = out[label][tgt_partition][v_index]

                if u == v:
                    continue

                if not tmp_graph.hasEdge(u,v):
                    tmp_graph.addEdge(u,v)
                    added_edges += 1

        # removing isolated nodes
        isolated_nodes = [u for u in tmp_graph.iterNodes() if tmp_graph.degree(u) == 0]

        isolated_nodes = torch.tensor(isolated_nodes, dtype=torch.long)

        mask = torch.ones(tmp_y.size(0), dtype=torch.bool)
        mask[isolated_nodes] = False

        # tmp_y = tmp_y[mask]
        tmp_y[~mask] = -1

        for node in isolated_nodes:
            tmp_graph.removeNode(node)

        # print(f'tamanho do y {tmp_y.shape}, numero de vértices {tmp_graph.numberOfNodes()}')

        self._add_heterogeneous_edges_until_density(
            graph=tmp_graph,
            y=tmp_y,
            target_density=rho,
            N=N,
        )

        # TESTE REMOVENDO OS VÉRTICES E REINDEXANDO
        tmp_graph = compact_graph_ids(tmp_graph)

        mask = torch.ones_like(tmp_y, dtype = torch.bool)
        mask[tmp_y == -1] = False
        tmp_y = tmp_y[mask]

        x = self._generate_attributes(
            graph=tmp_graph,
            y=tmp_y,
            dimensions=dimensions,
            sigma=sigma,
            alpha=alpha,
        )

        return  AttributedGraph(graph = tmp_graph, y = tmp_y, x = x)

    def generate(self, num_nodes: int, y: list[int], k: int, e: list[int], C: list[torch.tensor], d: list[str], rho: float, N: torch.tensor, M: list[torch.tensor], sigma = 1, dimensions = 10, alpha = 0.8, alpha_powerlaw = 2, sigma_distribution = None, mu_distribution = None, eps = 1e-6):
        return self._run_scatt(num_nodes=num_nodes, y=y, k=k, e=e, C=C, d=d, rho=rho, N=N, M=M, sigma=sigma, dimensions = dimensions, alpha = alpha, alpha_powerlaw=alpha_powerlaw, sigma_distribution=sigma_distribution, mu_distribution=mu_distribution, eps = eps)
    

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
