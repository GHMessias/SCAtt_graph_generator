# Examples

This directory contains runnable examples and notebooks that use the public `synco` API.

- `minimal_generation.py`: small script that generates one attributed graph.
- `time_complexity_analysis.ipynb`: runtime scaling experiment.
- `feature_space_alpha_analysis.ipynb`: feature-space visualization across `alpha_feat` and `alpha_topo`.
- `subcommunity_validation.ipynb`: subcommunity preservation analysis.
- `paper_plotter.ipynb`: publication plotting workflows.
- `DGCluster_evaluation.ipynb` and `evaluation.ipynb`: legacy evaluation notebooks.

Run examples from the repository root:

```bash
python examples/minimal_generation.py
jupyter lab examples/
```

Generated outputs should be written under `examples/outputs/` or `experiments/results/`, not inside the API package.
