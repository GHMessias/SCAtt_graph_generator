import argparse

import torch
import torch.nn.functional as F

from torch_geometric.nn import GAE, GCNConv
from torch_geometric.utils import train_test_split_edges

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import numpy as np

import os
import sys
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch  # <-- importante

sys.path.append("../")

from experiments.func import *   # arguments(), train_gae_kmeans_nmi, etc.
from models.SynCo import SynCoGenerator

def arguments():
    parser = argparse.ArgumentParser(prog="param_eval.py")

    # ======================
    # MODELO BASE (fixo)
    # ======================
    parser.add_argument("--n", type = int, default = 120, help = "number of nodes for the graph benchmark")
    parser.add_argument("--num_edges", type = int, nargs="+", default = [150, 200, 100], help = "number of edges for each class")
    parser.add_argument("--k", type = int, default = 3, help = "number of classes for the graph benchmark")
    parser.add_argument("--y", type = int, nargs= "+", default = [40,40,40], help = "list that represents the number of elements in each class.")
    parser.add_argument("--A_in", type = float, nargs = "+", default = [[1],
                                                                     [[0.6, 0.3, 0.1],
                                                                      [0.1, 0.7, 0.2],
                                                                      [0.0, 0.3, 0.7]],
                                                                     [[0.9,0.1],
                                                                      [0.1,0.9]]])
    parser.add_argument("--dst", type = str, nargs = "+", default = ["power_law", "normal", "uniform"])
    parser.add_argument("--A_out", type = float, nargs = "+", default = [[0, 0.2, 0.8], [0.3, 0, 0.7], [0.5, 0.5, 0]])
    parser.add_argument("--rho", type = int, default = 600, help = "total number of edges for graph generation")
    parser.add_argument("--S", type = float, nargs = "+", default = [[1], [0.4,0.3,0.3], [0.9,0.1]])
    parser.add_argument("--d", type = int, default = 60)

    # ======================
    # SELETOR DE EXPERIMENTO
    # ======================
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["topology_rho", "sfanalysis", "DGCluster"],
        # default="single",
        help="Tipo de experimento"
    )

    parser.add_argument(
        "--num_executions",
        type = int,
        default = 5,
        help = "numero de execuções de cada experimento"
    )

    # ======================
    # PARÂMETROS DO rho_exp
    # ======================
    parser.add_argument(
        "--rho_list",
        type=int,
        nargs="+",
        default=[600, 900, 1200, 1500, 1800],
        help="Lista de valores de rho (usado apenas no rho_exp)"
    )

    parser.add_argument(
        "--grid_step",
        type = float,
        default = 0.2,
        help = "step of execution for alpha_feat and alpha_topo in range [0,1.0]"
    )
    
    parser.add_argument(
        "--seed",
        type = float,
        default = 2026,
        help = "seed for reproduction. It is recommended to use a seed for the correct evaluation of alpha_feat and alpha_topo"
    )
    # ======================
    # PARÂMETROS DO sfanalysis
    # ======================

    parser.add_argument(
        "--base_graph_path",
        type = str,
        help = "path to graph to be cloned in data.pt format (pytorch).",
        default = "datasets/Cora/processed/data.pt"
    )

    parser.add_argument(
        "--num_aug_nodes",
        default = None,
        type = float,
        help = "number of nodes to clone base_graph (augmentation phase). This value is a float that will be use to multiply the number of original nodes in the graph"
    )

    parser.add_argument(
        "--clone_models",
        type = str,
        default = ["GenCAT", "SynCo", "chung-lu"],
        nargs = "+"
    )

    # ======================
    # PARÂMETROS DO DGCluster
    # ======================

    parser.add_argument(
        "--dgcluster_test",
        default = "test1",
        help = "Test1: DGCluster varying size of communities.",
        choices = ["test1", "test2"]
    )

    parser.add_argument(
        "--max_num_communities",
        type = int,
        default = 20)
    
    parser.add_argument(
        "--default_node_values",
        default = 15_000,
        type = int
    )

    

    return parser.parse_args()

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


@torch.no_grad()
def _to_numpy_1d(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().view(-1).numpy()


def train_gae_kmeans_nmi(
    data,
    hidden_channels: int = 64,
    embedding_dim: int = 32,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    epochs: int = 200,
    dropout: float = 0.0,
    split_edges: bool = False,
    seed = None,
    device: str | None = None,
):
    """
    Dado um objeto 'data' (PyG Data) com:
      - data.edge_index (shape [2, E])
      - data.x (shape [N, F])
      - data.y (shape [N]) (labels verdadeiros)
    Treina um modelo GAE e depois:
      - extrai embeddings z
      - roda KMeans (k = número de classes em y)
      - calcula NMI entre clusters e y

    Retorna:
      dict com loss_final, nmi, y_pred, z (embeddings)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Seeds (reprodutibilidade razoável)
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Sanity checks
    assert hasattr(data, "edge_index") and data.edge_index is not None
    assert hasattr(data, "x") and data.x is not None
    assert hasattr(data, "y") and data.y is not None

    # Copia/coloca no device
    data = data.clone()
    data = data.to(device)

    # Se quiser split (recomendado para avaliar, mas aqui pediram só treinar e clusterizar)
    # Ainda assim é bom para treinar com edges de treino e reconstruir.
    if split_edges:
        # Observação: train_test_split_edges cria data.train_pos_edge_index etc.
        # e remove do edge_index original (deixa só treino).
        data = train_test_split_edges(data, val_ratio=0.05, test_ratio=0.10)

        # Para GAE: usamos train_pos_edge_index durante treino
        train_edge_index = data.train_pos_edge_index
    else:
        train_edge_index = data.edge_index

    # Modelo
    encoder = GCNEncoder(
        in_channels=data.x.size(-1),
        hidden_channels=hidden_channels,
        out_channels=embedding_dim,
        dropout=dropout,
    ).to(device)

    model = GAE(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_one_epoch():
        model.train()
        optimizer.zero_grad()

        z = model.encode(data.x, train_edge_index)

        # Reconstrução de arestas positivas (treino)
        pos_loss = model.recon_loss(z, train_edge_index)

        # Regularização KL (se fosse VGAE). Para GAE, é zero.
        loss = pos_loss

        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu())

    # Treino
    last_loss = None
    for epoch in range(1, epochs + 1):
        last_loss = train_one_epoch()

    # Embeddings finais
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, train_edge_index)

    # KMeans com k = número de classes únicas em y (ignorando possíveis -1)
    y_true = data.y
    mask = y_true >= 0
    y_true_masked = y_true[mask]
    z_masked = z[mask]

    k = int(torch.unique(y_true_masked).numel())
    if k < 2:
        raise ValueError(f"Número de classes únicas em data.y (>=0) é {k}. KMeans precisa de k>=2.")

    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    y_pred = km.fit_predict(_to_numpy_1d(z_masked[:, 0]).reshape(-1, 1)) if z_masked.size(1) == 1 else km.fit_predict(
        z_masked.detach().cpu().numpy()
    )

    # NMI
    nmi = normalized_mutual_info_score(_to_numpy_1d(y_true_masked), y_pred)

    return {
        "loss_final": last_loss,
        "nmi": float(nmi),
        "y_pred": y_pred,          # numpy array (apenas nós com y>=0)
        "z": z.detach().cpu(),     # torch tensor (todos os nós)
        "mask_used": mask.detach().cpu(),
        "model": model
    }


# ----------------------------
# Helpers de serialização
# ----------------------------
def _to_jsonable(obj):
    """Converte objetos comuns (tensor, numpy, etc.) para formatos serializáveis em JSON."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return obj


def save_experiment_summary(args, exp_dir: Path):
    """
    Salva os inputs base do grafo (sem rho).
    """
    summary_dir = exp_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # pega tudo que existe em args
    args_dict = vars(args).copy()

    # remove o que NÃO é "base do grafo"
    # (mantive rho_list separado porque é parte do experimento, mas você pode remover também)
    args_dict.pop("rho", None)         # remove rho single, se existir
    # args_dict.pop("rho_list", None)  # se quiser também remover a lista do summary base

    # deixa JSON-friendly
    args_dict = _to_jsonable(args_dict)

    with open(summary_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=4, ensure_ascii=False)

    # opcional: salva o rho_list separado como registro do sweep
    if "rho_list" in vars(args):
        with open(summary_dir / "rho_list.json", "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(args.rho_list), f, indent=4, ensure_ascii=False)


def save_graph_data(dt, exp_dir: Path, exc: int = None, rho_value: float = None, model_name: str = None):
    """
    Salva o objeto Data do PyTorch Geometric.
    """
    graphs_dir = exp_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    if exc is not None and rho_value is not None:
        fname = f"exc{exc:03d}_rho{rho_value:.4f}_data.pt"

    if model_name is not None:
        fname = f"{model_name}_data.pt"
    torch.save(dt, graphs_dir / fname)


def save_model(model, exp_dir: Path, exc: int, rho_value: float):
    """
    Salva o modelo treinado (state_dict).
    """
    models_dir = exp_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    fname = f"exc{exc:03d}_rho{rho_value:.4f}_model_state.pt"
    torch.save(model.state_dict(), models_dir / fname)


def append_result(df, out: dict, dt, exc: int, rho_value: int, alpha_feat: float, alpha_topo: float):
    """
    Adiciona uma linha de resultados no DataFrame.
    """
    # num_edges no PyG geralmente é edge_index.shape[1]
    num_nodes = int(dt.num_nodes)
    num_edges = int(dt.edge_index.size(1)) / 2

    row = {
        "execution": exc,
        "rho": int(rho_value),
        "nmi": float(out["nmi"]),
        "num_nodes_generated": num_nodes,
        "num_edges_generated": num_edges,
        "alpha_feat": alpha_feat,
        "alpha_topo": alpha_topo,
    }

    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def save_metrics(df, exp_dir: Path):
    """
    Salva o CSV com resultados.
    """
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(metrics_dir / "results.csv", index=False)
