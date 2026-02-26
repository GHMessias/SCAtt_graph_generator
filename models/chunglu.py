import sys
sys.path.append('')

import random
import numpy as np
import torch
import networkit as nk

from core.attributed_graph import AttributedGraph
from core.base_graph_generator import BaseGenerator


from typing import Optional, Dict, List, Tuple


class ChungLuGenerator(BaseGenerator):
    """
    Chung-Lu por classe:
    1) Para cada classe c, cria um grafo intra-classe H_c via ChungLu usando a sequência de graus
       calculada no subgrafo induzido pelos nós da classe.
    2) Junta todos os H_c em um grafo global H.
    3) Adiciona arestas entre classes para manter exatamente a quantidade de arestas heterogêneas,
       e também tenta manter a distribuição por par de classes (mixing matrix).
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(name="ChungLuPerClass", supports_mimic=True, supports_augment=False)
        self.seed = seed
        if seed is not None:
            self._set_seed(seed)

    def generate(self) -> AttributedGraph:
        raise NotImplementedError(
            f"{self.__class__.__name__} não implementa generate(); use mimic(base_graph=...)."
        )

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        nk.setSeed(seed, useThreadId=False)

    def mimic(
        self,
        base_graph: AttributedGraph,
        *,
        num_nodes = None,
        seed: Optional[int] = None,
        copy_x: bool = False,
        shuffle_x_within_class: bool = False,
    ) -> AttributedGraph:
        """
        Parâmetros
        ----------
        base_graph:
            AttributedGraph original (com y obrigatório)

        seed:
            Seed opcional por chamada

        copy_x:
            Se True, cria x para o novo grafo usando as features do grafo original por classe

        shuffle_x_within_class:
            Se True, embaralha as linhas de x dentro de cada classe ao atribuir aos novos nós
            (evita "copiar na mesma ordem").
        """
        if seed is not None:
            self._set_seed(seed)

        if base_graph.y is None:
            raise ValueError("ChungLuPerClassMimicGenerator requer base_graph.y (rótulo por nó).")

        G = base_graph.graph
        x_base = base_graph.x
        y_base = base_graph.y

        # Classes (ignorando -1 se existir)
        classes = [c for c in torch.unique(y_base).tolist() if c != -1]
        classes = sorted(classes)

        # Nós por classe (no grafo base)
        nodes_by_class: Dict[int, List[int]] = {
            c: torch.where(y_base == c)[0].tolist()
            for c in classes
        }

        # 1) Estatísticas do grafo base: arestas intra e inter (heterogêneas)
        intra_target_edges: Dict[int, int] = {c: 0 for c in classes}
        inter_target_edges: Dict[Tuple[int, int], int] = {}  # (ci, cj) com ci < cj -> count

        for u, v in G.iterEdges():
            cu = int(y_base[u].item())
            cv = int(y_base[v].item())
            if cu == -1 or cv == -1:
                continue

            if cu == cv:
                intra_target_edges[cu] += 1
            else:
                a, b = (cu, cv) if cu < cv else (cv, cu)
                inter_target_edges[(a, b)] = inter_target_edges.get((a, b), 0) + 1

        total_hetero_target = sum(inter_target_edges.values())

        # 2) Criar mapeamento global de nós: cada classe vira um bloco contíguo
        global_nodes_by_class: Dict[int, List[int]] = {}
        offsets: Dict[int, int] = {}

        current = 0
        for c in classes:
            offsets[c] = current
            n_c = len(nodes_by_class[c])
            global_nodes_by_class[c] = list(range(current, current + n_c))
            current += n_c

        n_total = current
        H = nk.graph.Graph(n_total, weighted=False, directed=False)

        # 3) Gerar cada subgrafo intra-classe via Chung-Lu e inserir no grafo global
        for c in classes:
            base_nodes = nodes_by_class[c]
            n_c = len(base_nodes)
            if n_c <= 1:
                continue

            base_set = set(base_nodes)

            # graus intra-classe no grafo base (somente arestas com ambas pontas na classe)
            deg_seq = [0.0] * n_c
            base_index = {node: i for i, node in enumerate(base_nodes)}

            for u, v in G.iterEdges():
                if u in base_set and v in base_set:
                    deg_seq[base_index[u]] += 1.0
                    deg_seq[base_index[v]] += 1.0

            # Chung-Lu dentro da classe
            gen = nk.generators.ChungLuGenerator(deg_seq)
            Hc = gen.generate()

            # Ajustar número de arestas intra para bater com o do grafo base nessa classe
            m_target = intra_target_edges[c]
            Hc = self._force_num_edges(Hc, m_target)

            # Inserir arestas de Hc em H (fazendo o shift pelo offset)
            off = offsets[c]
            for u, v in Hc.iterEdges():
                H.addEdge(off + u, off + v)

        # 4) Adicionar arestas inter-classe para manter a heterogeneidade (e a mistura por par)
        for (a, b), cnt in inter_target_edges.items():
            if cnt <= 0:
                continue

            A_nodes = global_nodes_by_class[a]
            B_nodes = global_nodes_by_class[b]

            self._add_bipartite_edges(H, A_nodes, B_nodes, cnt)

        # Checagem rápida (opcional, mas útil)
        hetero_now = self._count_hetero_edges(H, self._build_y_from_blocks(classes, global_nodes_by_class))
        if hetero_now != total_hetero_target:
            # fallback simples: tenta corrigir para bater exatamente (raramente necessário)
            H = self._fix_hetero_count_exact(
                H,
                y=self._build_y_from_blocks(classes, global_nodes_by_class),
                target=total_hetero_target,
                nodes_by_class=global_nodes_by_class,
            )

        # 5) Construir y novo (por blocos)
        y_new = self._build_y_from_blocks(classes, global_nodes_by_class)

        # 6) Construir x novo (por classe)
        if copy_x:
            if not isinstance(x_base, torch.Tensor):
                raise TypeError("copy_x=True requer base_graph.x como torch.Tensor")

            x_new = torch.empty((n_total, x_base.shape[1]), dtype=x_base.dtype)

            for c in classes:
                base_nodes = nodes_by_class[c]
                new_nodes = global_nodes_by_class[c]

                x_c = x_base[base_nodes]

                if shuffle_x_within_class:
                    perm = torch.randperm(x_c.shape[0])
                    x_c = x_c[perm]

                x_new[new_nodes] = x_c
        else:
            # se você quiser gerar features novas depois, deixe um placeholder
            x_new = torch.zeros((n_total, x_base.shape[1]), dtype=x_base.dtype)

        return AttributedGraph(graph=H, x=x_new, y=y_new)

    # ------------------------
    # Helpers
    # ------------------------

    def _force_num_edges(self, H: nk.Graph, m_target: int) -> nk.Graph:
        """
        Ajusta o número de arestas de H para ser exatamente m_target.
        """
        n = H.numberOfNodes()
        m = H.numberOfEdges()
        if m == m_target:
            return H

        edges = list(H.iterEdges())

        if m > m_target:
            to_remove = m - m_target
            random.shuffle(edges)
            removed = 0
            for u, v in edges:
                if removed >= to_remove:
                    break
                if H.hasEdge(u, v):
                    H.removeEdge(u, v)
                    removed += 1
            return H

        # m < m_target
        to_add = m_target - m
        added = 0
        tries = 0
        max_tries = to_add * 100 + 10_000

        while added < to_add and tries < max_tries:
            tries += 1
            u = random.randrange(n)
            v = random.randrange(n)
            if u == v:
                continue
            if H.hasEdge(u, v):
                continue
            H.addEdge(u, v)
            added += 1

        return H

    def _add_bipartite_edges(self, H: nk.Graph, A: List[int], B: List[int], cnt: int):
        """
        Adiciona cnt arestas entre conjuntos A e B (A != B).
        Evita duplicatas e loops.
        """
        if not A or not B or cnt <= 0:
            return

        added = 0
        tries = 0
        max_tries = cnt * 200 + 50_000

        while added < cnt and tries < max_tries:
            tries += 1
            u = random.choice(A)
            v = random.choice(B)
            if u == v:
                continue
            if H.hasEdge(u, v):
                continue
            H.addEdge(u, v)
            added += 1

        if added < cnt:
            # Em grafos pequenos e densos, pode ficar difícil achar pares novos.
            # Aqui você pode decidir se aceita ficar abaixo ou lançar erro.
            pass

    def _build_y_from_blocks(self, classes: List[int], global_nodes_by_class: Dict[int, List[int]]) -> torch.Tensor:
        n_total = sum(len(global_nodes_by_class[c]) for c in classes)
        y = torch.empty((n_total,), dtype=torch.long)

        for c in classes:
            y[global_nodes_by_class[c]] = int(c)

        return y

    def _count_hetero_edges(self, H: nk.Graph, y: torch.Tensor) -> int:
        count = 0
        for u, v in H.iterEdges():
            if y[u] != y[v]:
                count += 1
        return count

    def _fix_hetero_count_exact(
        self,
        H: nk.Graph,
        *,
        y: torch.Tensor,
        target: int,
        nodes_by_class: Dict[int, List[int]],
    ) -> nk.Graph:
        """
        Ajuste de segurança:
        - Se hetero > target: remove algumas arestas heterogêneas aleatórias.
        - Se hetero < target: adiciona arestas heterogêneas aleatórias.
        Mantém o objetivo de bater exatamente o número de hetero.
        """
        hetero = self._count_hetero_edges(H, y)

        if hetero == target:
            return H

        # Listas úteis
        classes = sorted(nodes_by_class.keys())

        if hetero > target:
            # remover heterogêneas
            to_remove = hetero - target
            hetero_edges = [(u, v) for (u, v) in H.iterEdges() if y[u] != y[v]]
            random.shuffle(hetero_edges)
            removed = 0
            for u, v in hetero_edges:
                if removed >= to_remove:
                    break
                if H.hasEdge(u, v) and y[u] != y[v]:
                    H.removeEdge(u, v)
                    removed += 1
            return H

        # hetero < target: adicionar heterogêneas
        to_add = target - hetero
        added = 0
        tries = 0
        max_tries = to_add * 200 + 50_000

        while added < to_add and tries < max_tries:
            tries += 1
            ca, cb = random.sample(classes, 2)
            u = random.choice(nodes_by_class[ca])
            v = random.choice(nodes_by_class[cb])
            if u == v:
                continue
            if H.hasEdge(u, v):
                continue
            H.addEdge(u, v)
            added += 1

        return H
