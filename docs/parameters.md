# SynCo Parameters

Core generation parameters:

- `n`: total number of vertices. Must match `sum(y)`.
- `k`: number of classes/communities. Must match `len(y)`.
- `y`: number of vertices per class, for example `[40, 40, 40]`.
- `e`: number of homogeneous edges sampled inside each class.
- `rho`: target total number of edges. Must be at least `sum(e)`.
- `S`: relative sizes of subcommunities inside each class.
- `A_in`: subcommunity interaction matrices for homogeneous edges.
- `A_out`: class interaction matrix for heterogeneous edges.
- `dst`: degree preference per class: `"power_law"`, `"normal"`, or `"uniform"`.
- `d`: feature dimension.
- `alpha_feat`: Gaussian feature perturbation strength.
- `alpha_topo`: trade-off between class prototypes and topological smoothing.
- `connect_isolated_nodes`: if `True`, adds a post-processing pass to connect isolated vertices.
- `build_subgraphs`: if `False`, skips subgraph construction for faster bulk generation.

For large-scale runtime experiments, prefer:

```python
build_subgraphs=False
connect_isolated_nodes=False
```

For final graph quality checks, measure the number of isolated vertices and connected components before deciding whether to enable `connect_isolated_nodes`.

