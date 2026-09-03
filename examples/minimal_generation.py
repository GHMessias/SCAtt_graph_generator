from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synco import SynCoGenerator


def main():
    y = [40, 40, 40]
    e = [150, 200, 100]

    graph = SynCoGenerator(seed=2026).generate(
        n=sum(y),
        y=y,
        k=len(y),
        e=e,
        A_in=[
            torch.tensor([[1.0]]),
            torch.tensor([[0.6, 0.3, 0.1], [0.1, 0.7, 0.2], [0.0, 0.3, 0.7]]),
            torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
        ],
        dst=["power_law", "normal", "uniform"],
        rho=600,
        A_out=torch.tensor([[0.0, 0.2, 0.8], [0.3, 0.0, 0.7], [0.5, 0.5, 0.0]]),
        S=[torch.tensor([1.0]), torch.tensor([0.4, 0.3, 0.3]), torch.tensor([0.9, 0.1])],
        d=60,
        alpha_feat=0.3,
        alpha_topo=0.5,
    )

    print(f"nodes={graph.num_nodes()} edges={graph.num_edges()} features={tuple(graph.x.shape)}")


if __name__ == "__main__":
    main()

