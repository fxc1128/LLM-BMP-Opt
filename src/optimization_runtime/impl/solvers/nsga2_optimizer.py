from __future__ import annotations

from ...core.interfaces import AbstractSolver
from .unified_solver import UnifiedSolver


class NSGA2Optimizer(AbstractSolver):
    """Registry-facing NSGA-II wrapper around the compatible UnifiedSolver."""

    def __init__(self, operators, cfg: dict):
        self._solver = UnifiedSolver(operators, cfg)

    def solve(self, problem, on_generation_callback=None):
        return self._solver.solve(problem, on_generation_callback=on_generation_callback)
