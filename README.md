# SynCo Graph Generator

SynCo is a synthetic community-aware attributed graph generator for graph learning benchmarks, community detection, clustering, and Graph Neural Network experiments.

The project exposes a small public Python API through `synco/`. Paper experiments, notebooks, plots, and generated outputs are kept outside the API package so the repository is easier to reuse.

## Repository Layout

```text
synco/          # Public API: recommended imports for new users
core/           # Attributed graph container and base generator interfaces
models/         # SynCo implementation and baseline generators
examples/       # Reproducible notebooks and usage examples
experiments/    # Legacy paper experiment runners, helpers, and outputs
analysis/       # Scale-free and power-law analysis utilities
plot/           # Publication plotting helpers
docs/           # Short API and parameter notes
```

`models/SCatt.py` remains as a compatibility shim for old code, but new code should use `synco`.

## Installation

Create a virtual environment and install the scientific stack:

```bash
pip install -r requirements.txt
```

For editable local development:

```bash
pip install -e .
```

## Minimal Example

```python
import torch

from synco import SynCoGenerator

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

print(graph.num_nodes(), graph.num_edges(), graph.x.shape)
data = graph.to_data_pytorch()
```

You can also run:

```bash
python examples/minimal_generation.py
```

## Main Parameters

- `n`, `k`, `y`: total vertices, number of classes, and vertices per class.
- `e`: homogeneous edges sampled inside each class.
- `rho`: target total number of edges, with `rho >= sum(e)`.
- `S`: relative subcommunity sizes inside each class.
- `A_in`: subcommunity mixing matrices for homogeneous edges.
- `A_out`: class mixing matrix for heterogeneous edges.
- `dst`: degree preference per class: `"power_law"`, `"normal"`, or `"uniform"`.
- `d`: feature dimension.
- `alpha_feat`: Gaussian perturbation strength in the feature space.
- `alpha_topo`: trade-off between class-conditioned features and topology smoothing.
- `connect_isolated_nodes`: optional post-processing step that links isolated vertices.
- `build_subgraphs`: whether to build per-class subgraphs in the returned object.

See [docs/parameters.md](docs/parameters.md) for a compact parameter reference.

## Examples and Paper Workflows

Reusable workflows live in `examples/`:

```bash
jupyter lab examples/
```

Current examples include:

- `time_complexity_analysis.ipynb`: runtime scaling and graph-quality metrics.
- `feature_space_alpha_analysis.ipynb`: t-SNE visualizations across `alpha_feat` and `alpha_topo`.
- `subcommunity_validation.ipynb`: subcommunity preservation analysis.
- `paper_plotter.ipynb`: publication plotting workflows.

Generated CSVs, figures, and TeX files should stay under `experiments/results/`, `experiments/complexity/`, or a local output directory rather than inside `synco/`.

## Development Checks

There is no formal test suite yet. Use these commands as basic checks:

```bash
python -m compileall synco core models analysis experiments plot examples
python examples/minimal_generation.py
```

If you add tests, place them in `tests/` and name files `test_*.py`.

## Notes

SynCo generates undirected, unweighted attributed graphs. The diagonal of `A_out` should be zero because homogeneous edges are controlled by `e`, `S`, `A_in`, and `dst`.
