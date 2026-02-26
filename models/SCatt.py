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

    def rank_based_matching(self, base_graph: AttributedGraph, mimic_graph: nk.Graph, noise_mean: float = 0, noise_std: float = 1):
        x = torch.zeros((base_graph.graph.numberOfNodes(), base_graph.x.size(1)))

        for label in torch.unique(base_graph.y):
            # Aplicar a máscara para pegar apenas os nós com o rótulo (label) atual
            mask = base_graph.y == label

            # Filtrando os nós pelo label
            nodes_with_label = torch.nonzero(mask).squeeze()  # Isso nos dá os índices dos nós com o rótulo desejado

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

        for index, sub_g in base_graph.subgraphs.items():
            prt = graphPartition(subgraph=sub_g, base_community_detector=base_community_detector)
            deg = {node: sub_g.degree(node) for node in sub_g.iterNodes()}

            keys = list(deg.keys())

            # Somando 1 para considerar vértices que são isolados
            # weights = torch.tensor(list(deg.values()), dtype=torch.float) + 1
            weights = torch.tensor(list(deg.values()), dtype=torch.float)

            for _ in range(sub_g.numberOfEdges()):
                # Escolho um vértice de acordo com sua node degree distribution
                tmp_idx = torch.multinomial(weights, 1).item()
                u = keys[tmp_idx]

                # Partição 
                src_partition = prt._get_partition(u)

                # Escolho em C algum vértice para ligar u
                tgt_partition = torch.multinomial(input = prt.C[src_partition], num_samples=1)[0]

                # Seleciono algum vértice v de acordo com a probabilidade de ligação dos vértices em tgt_partition
                tmp_deg = {node:sub_g.degree(node) for node in prt.graph_partitions[tgt_partition.item()]}
                tmp_keys = list(tmp_deg.keys())
                tmp_weights = torch.tensor(list(tmp_deg.values()), dtype=torch.float)

                tmp_idx = torch.multinomial(tmp_weights, 1).item()
                v = tmp_keys[tmp_idx]

                if u != v:
                    graph.addEdge(u,v)

        # Adding noise to the graph

        aim_density = nk.graphtools.density(base_graph.graph)

        actual_density = nk.graphtools.density(graph)
        tmp_weights = torch.ones(graph.numberOfNodes())

        # print(actual_density ,'/', aim_density)

        # TODO: verificar se essa adição aleatória de densidade faz sentido para a criação do algoritmo
        while actual_density < aim_density:
            # print(actual_density ,'/', aim_density, end = '\r')
            # Adiciona arestas aleatorias no grafo
            u = tmp_weights.multinomial(num_samples=1, replacement=False)
            v = tmp_weights.multinomial(num_samples=1, replacement=False)

            if y[u] == y[v]:
                continue
            else:
                if not graph.hasEdge(u,v):
                    graph.addEdge(u,v)
            
            actual_density = nk.graphtools.density(graph)


        isolated_nodes = [u for u in graph.iterNodes() if graph.degree(u) == 0]

        # Add isolated nodes in nodes of the same class, according to degree probability

        # for u in isolated_nodes:
        #     selected_candidates = torch.tensor(list(graph.iterNodes()))[y == y[u]]
        #     degrees = torch.tensor([graph.degree(nd) for nd in selected_candidates]).type(torch.float)
        #     v = torch.multinomial(degrees, num_samples = 1, replacement=False)[0].item()

        #     added_var = True

        #     max_iter = 0
        #     while added_var:
        #         # print(u,v)
                
        #         if u != v:
        #             graph.addEdge(u,v)
        #             added_var = False
        #             max_iter += 1
        #             if max_iter > 100:
        #                 break

        for u in isolated_nodes:

            all_nodes = torch.tensor(list(graph.iterNodes()))

            # candidatos da mesma classe exceto u
            same_class = all_nodes[(y == y[u]) & (all_nodes != u)]

            if len(same_class) > 0:
                degrees = torch.tensor([graph.degree(nd.item()) for nd in same_class]).float()

                if degrees.sum() > 0:
                    probs = degrees / degrees.sum()
                    v = same_class[torch.multinomial(probs, 1)].item()
                else:
                    # todos têm grau zero → escolha uniforme
                    v = same_class[torch.randint(len(same_class), (1,))].item()

            else:
                # fallback: qualquer nó diferente de u
                fallback = all_nodes[all_nodes != u]
                v = fallback[torch.randint(len(fallback), (1,))].item()

            graph.addEdge(u, v)

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
            _global_partitions = {index:graphPartition(subgraph, base_community_detector) for index,subgraph in attGraph.subgraphs.items()}
            
            for _ in range(nodes_to_add):
                # added node
                v_i = attGraph.graph.addNode()

                # class of the added node
                c_i = torch.multinomial(torch.unique(attGraph.y, return_counts = True)[1].float(), num_samples=1)[0].item()

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
                        tgt_partition = torch.multinomial(row, 1)[0].item()

                    continue_add = True
                    while continue_add:
                        # Seleciona o vértice a ser ligado a partir do degree
                        v_j = torch.multinomial(torch.tensor([attGraph.graph.degree(nd) for nd in _global_partitions[c_i].graph_partitions[tgt_partition]]).type(torch.float), num_samples=1, replacement=False)[0].item()
                        v_j = _global_partitions[c_i].graph_partitions[tgt_partition][v_j]
                        if v_i != v_j:
                            graph.addEdge(v_i,v_j)
                            _global_partitions[c_i].C[src_partition, tgt_partition] += 1
                            _global_partitions[c_i].C[tgt_partition, src_partition] += 1
                            continue_add = False

            # Adicionar Ruído
            actual_density = nk.graphtools.density(attGraph.graph)
            aim_density = nk.graphtools.density(base_graph.graph)
            tmp_weights = torch.ones(attGraph.graph.numberOfNodes())

            # print(actual_density ,'/', aim_density)
            while actual_density < aim_density:
                # print(round(actual_density,4) ,'/', round(aim_density,4))
                # Adiciona arestas aleatorias no grafo
                u = tmp_weights.multinomial(num_samples=1, replacement=False).item()
                v = tmp_weights.multinomial(num_samples=1, replacement=False).item()

                # print(u,v)

                if attGraph.y[u] == attGraph.y[v]:
                    continue
                else:
                    if not attGraph.graph.hasEdge(u,v):
                        attGraph.graph.addEdge(u,v)
                
                actual_density = nk.graphtools.density(attGraph.graph)

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

        # TODO: colocar um iterador máximo pra não dar erro quando não consegue sortear os vértices corretos.
        is_zero_diag = torch.all(torch.diagonal(N) == 0)
        if not is_zero_diag:
            raise ValueError('Matrix N diagonal should be zero')

        tmp_graph = nk.Graph(num_nodes, weighted = False, directed = False)
        tmp_y = torch.tensor([i for i, v in enumerate(y) for _ in range(v)])

        # Atribuir cada um dos vértices de cada grupo a cada partição

        # out = split_by_class_and_partitions(n = num_nodes, y = y, parts=M.detach().cpu().flatten().tolist())
        out = split_by_class_and_partitions(n = num_nodes, y = y, parts=[x.tolist() for x in M])
        for label, label_num_nodes in enumerate(y):
            # distribution = distribution_transform(d = d[label], n = label_num_nodes)
            distributions = [distribution_transform(d = d[label], n = len(out[label][ptt]), alpha=alpha_powerlaw, mu = mu_distribution, sigma=sigma_distribution) for ptt in range(len(M[label]))]

            added_edges = 0
            tmp_count = 0

            while added_edges < e[label]:
                # seleciona src_partition e tgt_partition
                src_partition = torch.multinomial(M[label].float(), 1).item()
                tgt_partition = torch.multinomial(M[label].float(), 1).item()


                # row = C[label][src_partition].float()

                # if row.sum() == 0:
                #     tgt_partition = src_partition
                # else:
                #     tgt_partition = torch.multinomial(row, 1).item()

                # seleciona u e v em src_partition
                u_index = torch.multinomial(input = distributions[src_partition], num_samples=1).item()
                v_index = torch.multinomial(input = distributions[tgt_partition], num_samples=1).item()
                u = out[label][src_partition][u_index]
                v = out[label][tgt_partition][v_index]

                # verificar se u é igual a v
                while u == v:
                    u_index = torch.multinomial(input = distributions[src_partition], num_samples=1).item()
                    v_index = torch.multinomial(input = distributions[tgt_partition], num_samples=1).item()
                    u = out[label][src_partition][u_index]
                    v = out[label][tgt_partition][v_index]

                if not tmp_graph.hasEdge(u,v):
                    tmp_graph.addEdge(u,v)
                    added_edges += 1
                else:
                    tmp_count += 1

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

        # noise stage

        actual_node_density = nk.graphtools.density(tmp_graph)

        # TODO: Verificar a adição de arestas heterogêneas
        while actual_node_density < rho:
            
            # adiciona uma aresta aleatoria entre as classes que estão disponíveis em N
            # p = N.flatten()
            # p = p / p.sum()
            p = N.sum(dim = 1)
            # print(p)

            # amostra um índice linear
            community_u = torch.multinomial(p, num_samples=1).item()
            community_v = torch.multinomial(N[community_u], num_samples=1).item()
            # print(community_u, community_v)

            idx1 = torch.nonzero(tmp_y == community_u, as_tuple=False).squeeze()
            idx2 = torch.nonzero(tmp_y == community_v, as_tuple=False).squeeze()
            # print(idx1, idx2)

            u = idx1[torch.randint(len(idx1), (1,))].item()
            v = idx2[torch.randint(len(idx2), (1,))].item()
            # print(u,v, tmp_graph.hasEdge(u,v))

            # print(community_u, u,community_v,v)
            
            tmp_graph.addEdge(u,v)

            actual_node_density = nk.graphtools.density(tmp_graph)

        # TESTE REMOVENDO OS VÉRTICES E REINDEXANDO
        tmp_graph = compact_graph_ids(tmp_graph)

        mask = torch.ones_like(tmp_y, dtype = torch.bool)
        mask[tmp_y == -1] = False
        tmp_y = tmp_y[mask]

        A = np.random.randn(dimensions,k)
        Q, _ = np.linalg.qr(A)
        Q = Q[:,:k].T

        # print('vetores ortogonais', Q)

        x = np.zeros(shape = (tmp_graph.numberOfNodes(), dimensions))

        for nd in range(tmp_graph.numberOfNodes()):
            x[nd] = Q[tmp_y[nd]] + sigma * np.random.normal(size = (dimensions))

        dd = nk.centrality.DegreeCentrality(tmp_graph, normalized=False)
        dd.run()
        degrees = np.array(dd.scores(), dtype=float)

        D = np.zeros_like(degrees)

        mask = degrees > 0
        D[mask] = degrees[mask] ** (-0.5)

        D = np.diag(D)

        A = nk.algebraic.adjacencyMatrix(tmp_graph, matrixType="dense")

        x = (alpha) * x + (1-alpha) * D @ (A @ x)

        return  AttributedGraph(graph = tmp_graph, y = tmp_y, x = torch.tensor(x))

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

        final_num_nodes = int(mimic_num_nodes * train_graph.graph.numberOfNodes())
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