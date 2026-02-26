from __future__ import annotations
from typing import Optional



import networkit as nk
import torch

from torch_geometric.data import Data
from torch_geometric.utils import from_networkit

class AttributedGraph:
    """
    Representa um grafo com atributos padrão
    dentro da biblioteca synthetic_graphgen

    Atributos principais
    ----------------------------
    graph: nk.Graph
        Grafo da biblioteca networkit, contendo nós e arestas

    x: torch.Tensor
        Features de cada vértice, x.shape[0] deve ter o número de vértices.
    
    y: torch.Tensor | None
        Rótulos/targets, podendo ser:
        - None
        - tensor com shape[0] == número de nós (rótulo por nó)
    """

    def __init__(
        self,
        graph: nk.Graph,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None
    ) -> None:
        self.graph = graph
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self._validate()

        if self.y is not None:
            self.create_subgraphs()

    # Métodos auxiliares
    
    def _validate(self):
        """
        Faz checagens de consistência para o objeto criado
        """

        n_nodes = self.num_nodes()

        # if not isinstance(self.x, torch.Tensor):
        #     raise TypeError("AttributedGraph.x deve ser um torch.Tensor")

        # if self.x.shape[0] != n_nodes:
        #     raise ValueError(
        #         f"AttributedGraph.x.shape[0] : {self.x.shape[0]}"
        #         f"Número de nós no grafo : {n_nodes}"
        #     )

        # if self.y is not None:
        #     if not isinstance(self.y, torch.Tensor):
        #         raise TypeError("AttributedGraph.y deve ser um torch.Tensor ou None")

        #     if self.y.shape[0] != n_nodes:
        #         raise ValueError(f"AttributedGraph.y.shape[0] {self.y.shape[0]} é diferente do número de nós {n_nodes}")
            
        if self.graph.isDirected():
            raise NotImplementedError(f"Not implemented for directed AttributedGraph.graph")


    # Propriedades úteis da classe
    def num_nodes(self) -> int:
        """
        Retorna o número de nós de um grafo
        """
        return self.graph.numberOfNodes()

    def num_edges(self) -> int:
        """
        Retorna o número de arestas de um grafo
        """
        return self.graph.numberOfEdges()

    def has_labels(self) -> bool:
        """
        Indica se o grafo possui rótulos (y != None)
        """
        return self.y is not None
    
    def num_classes(self) -> int:
        """
        Retorna a quantidade de classes distintas do grafo.
        """
        return len(torch.unique(self.y))

    def __repr__(self) -> str:
        return (
            f"AttributedGraph("
            f"n_nodes={self.num_nodes()}, "
            f"n_edges={self.num_edges()}, "
            f"x_shape={tuple(self.x.shape)}, "
            f"y_shape={tuple(self.y.shape) if self.y is not None else None})"
        )
    
    def create_subgraphs(self):
        """
        Cria uma estrutura de subgrafos que compartilham um mesmo rótulo em self.y. Essas estruturas 
        de subgrafos serão utilizadas para encontrar partições (ou seja, subsubgrafos). Essas estruturas serão criadas
        a partir de nk.subgraphAndNeighborsFromNodes().        
        """
        self.subgraphs = dict()

        # unique_labels = 

        for index,value in enumerate(torch.unique(self.y).tolist()):
            if value == -1:
                continue
            set_of_nodes = torch.where(self.y == value)[0].tolist()
            self.subgraphs[index] = nk.graphtools.subgraphAndNeighborsFromNodes(self.graph, nodes = set_of_nodes)

        return
    
    def _compute_heterogeneity(self) -> int:
        """
        Calcula a quantidade de arestas que ligam grupos (ou classes) distintos em um mesmo grafo. Retorna a proporção de arestas heterogêneas
        
        :param self: Description
        """

        num_hetero_edges = 0
        for u,v in self.graph.iterEdges():
            if self.y[u] != self.y[v]:
                num_hetero_edges += 1
        
        return num_hetero_edges / self.graph.numberOfNodes()

    def to_data_pytorch(self) -> torch.data.Data:
        
        return Data(x = self.x.float(), y = self.y, edge_index = from_networkit(self.graph)[0])