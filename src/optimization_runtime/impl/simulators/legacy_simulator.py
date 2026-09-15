from __future__ import annotations

from ...core.interfaces import AbstractSimulator, DecisionSpace, SimulationResult


class LegacySimulator(AbstractSimulator):
    """Compatibility bridge while legacy evaluators own model execution.

    New simulators must materialize the scenario and populate outputs here. This
    bridge preserves existing SEIMS/SWAT+/SWMM evaluators during migration.
    """

    def run(self, individual, decision_space: DecisionSpace, context: dict) -> SimulationResult:
        return SimulationResult(metadata={"mode": "legacy_evaluator_executes_simulator", "unit_ids": decision_space.unit_ids})
