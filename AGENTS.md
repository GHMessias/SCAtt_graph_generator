# Repository Guidelines

## Project Structure & Module Organization

This is a Python research codebase for SynCo, a synthetic community-aware attributed graph generator. The public API lives in `synco/`; new users should import from `synco` rather than directly from implementation modules. Core graph abstractions remain in `core/`, generator implementations and baselines live in `models/`, reusable notebooks and usage examples are in `examples/`, experiment runners and generated paper outputs are in `experiments/`, analysis utilities are in `analysis/`, and publication plotting helpers are in `plot/`. Generated artifacts use `.pt`, `.csv`, `.png`, `.svg`, `.tex`, and `.pdf`, mostly under `experiments/results/`, `experiments/complexity/`, or `experiments/feature_space/`.

## Build, Test, and Development Commands

There is no packaged build system or locked dependency file. Create a virtual environment and install the imported scientific stack manually, including `torch`, `torch_geometric`, `networkit`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `networkx`, `netgraph`, and `powerlaw`.

Useful commands:

```bash
python -m compileall synco core models analysis experiments plot examples
python examples/minimal_generation.py
python experiments/experiments.py --experiment topology_rho
python experiments/experiments.py --experiment sfanalysis
python experiments/experiments.py --experiment DGCluster
jupyter lab
```

Use `compileall` as a syntax smoke check. Run commands from the repository root so paths such as `experiments/results/...` resolve correctly.

## Coding Style & Naming Conventions

Use Python with 4-space indentation. Follow the existing style: generator classes use `PascalCase` names ending in `Generator`, public methods use `snake_case`, and experiment options use argparse flags such as `--experiment` and `--rho_list`. Keep public imports in `synco/`, model implementation code in `models/`, shared graph logic in `core/`, and avoid adding notebook-only helpers to library modules unless reusable.

## Testing Guidelines

No formal test suite is configured. For code changes, run `python -m compileall ...` and at least one minimal generation or mimic workflow relevant to the edit. If adding tests, place them under `tests/`, name files `test_*.py`, and prefer deterministic cases with fixed seeds, for example `SynCoGenerator(seed=2026)`.

## Commit & Pull Request Guidelines

Recent commits use short, descriptive messages such as `minor changes` and `plots, figures and node clustering models`. Keep commits focused and mention the affected area when possible, for example `models: adjust SynCo edge sampling`. Pull requests should include a summary, commands or notebooks run, generated artifact notes, and screenshots for plot changes. Avoid committing large generated outputs unless required for reproduction or publication.

## Security & Configuration Tips

Do not commit local datasets, caches, or bulky regenerated experiment outputs unless intentional. The `.gitignore` already excludes `datasets/`, `DGCluster/`, `test/`, selected `experiments/results/...` paths, `__pycache__/`, and `*.pyc`.
