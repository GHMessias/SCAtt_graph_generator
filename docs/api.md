# API Overview

The recommended public import path is:

```python
from synco import SynCoGenerator, AttributedGraph
```

`SynCoGenerator.generate(...)` creates a synthetic attributed graph from user-defined structural and feature parameters. It returns an `AttributedGraph` with:

- `graph.graph`: the underlying NetworKit graph;
- `graph.x`: node feature matrix;
- `graph.y`: node labels/classes;
- `graph.to_data_pytorch()`: conversion to a PyTorch Geometric `Data` object.

`SynCoGenerator.mimic(...)` estimates structure from an existing attributed graph and generates a synthetic graph with similar class, subcommunity, edge, and feature patterns.

Legacy imports are still supported:

```python
from models.SynCo import SynCoGenerator
from models.SCatt import SCAttGenerator
```

