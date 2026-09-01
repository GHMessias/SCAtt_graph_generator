"""Backward-compatible import path for the former SCAtt module name."""

from models.SynCo import SynCoGenerator

SCAttGenerator = SynCoGenerator

__all__ = ["SynCoGenerator", "SCAttGenerator"]
