from __future__ import annotations

from typing import Any, Mapping

from ...core.interfaces import AbstractEncoder, DecisionSpace


class PayloadEncoder(AbstractEncoder):
    """Build an ordered decision space from explicit payload bindings.

    This is the public default for SWAT+/SWMM/custom profiles. Real encoders may
    read GIS/model files and return richer metadata using the same contract.
    """

    def build(self, agent_payload: Mapping[str, Any]) -> DecisionSpace:
        problem = agent_payload.get("optimization_problem", {})
        decision = problem.get("decision_variable", {}) if isinstance(problem, Mapping) else {}
        bmp_value = decision.get("BMP_value", {}) if isinstance(decision, Mapping) else {}
        values = bmp_value.get("BMP_type", [0, 1]) if isinstance(bmp_value, Mapping) else [0, 1]
        domain = [int(value) for value in values]
        binding = self._binding(agent_payload)
        unit_ids = self._unit_ids(decision, binding)
        bmp_catalog = self._bmp_catalog(binding, domain)
        return DecisionSpace(
            unit_ids=unit_ids,
            gene_domains=[list(domain) for _ in unit_ids],
            gene_to_unit=dict(enumerate(unit_ids)),
            unit_to_gene={unit_id: index for index, unit_id in enumerate(unit_ids)},
            bmp_catalog=bmp_catalog,
            unit_metadata={unit_id: {} for unit_id in unit_ids},
            spatial_relations=dict(binding.get("spatial_relations", {})) if isinstance(binding, Mapping) else {},
        )

    @staticmethod
    def _binding(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        evaluators = payload.get("evaluator", {})
        if not isinstance(evaluators, Mapping):
            return {}
        for cfg in evaluators.values():
            if isinstance(cfg, Mapping) and isinstance(cfg.get("decision_binding"), Mapping):
                return cfg["decision_binding"]
        return {}

    @staticmethod
    def _unit_ids(decision: Mapping[str, Any], binding: Mapping[str, Any]) -> list[str]:
        for key in ("unit_ids", "subcatchments", "lu_mgt_names", "target_lu_mgt_names"):
            values = binding.get(key, decision.get(key, []))
            if isinstance(values, list) and values:
                return [str(value) for value in values]
        count = int(decision.get("candidate_unit_count", decision.get("genes_num", 0)) or 0)
        if count <= 0:
            raise ValueError("Encoder requires decision_binding unit IDs or candidate_unit_count.")
        return [f"unit_{index + 1}" for index in range(count)]

    @staticmethod
    def _bmp_catalog(binding: Mapping[str, Any], domain: list[int]) -> dict[int, str]:
        actions = binding.get("actions", []) if isinstance(binding, Mapping) else []
        if isinstance(actions, list) and actions:
            return {index: str(value) for index, value in enumerate(actions)}
        return {value: f"BMP_{value}" for value in domain}
