from __future__ import annotations

from typing import Tuple

from ..impl.constraints.structured_constraint import normalize_bmp_types
from ..impl.solvers.unified_solver import SolverOperators


class LightweightBuilder:
    """Build a lightweight optimization context from an AGENT_PAYLOAD."""

    @staticmethod
    def build(agent_payload: dict) -> Tuple[SolverOperators, dict]:
        prob = agent_payload.get("optimization_problem", {})
        decision = prob.get("decision_variable", {})
        bmp_val = prob.get("decision_variable", {}).get("BMP_value", {})
        bmp_types = normalize_bmp_types(bmp_val.get("BMP_type", [1, 2]))
        bmp_times = normalize_bmp_types(bmp_val.get("BMP_time", []), default=[])

        eval_cfg = agent_payload.get("evaluator", {})
        genes_num = LightweightBuilder._infer_genes_num(agent_payload, eval_cfg, bmp_types)
        bmp_cost_table = LightweightBuilder._default_cost_table(bmp_types)
        unit_area = float(decision.get("unit_area", decision.get("default_unit_area", 10.0 if bmp_times else 1.0)))

        context = {
            "genes_num": genes_num,
            "bmp_types": bmp_types,
            "bmp_time_values": bmp_times,
            "units_infos": {},
            "gene_to_unit": {},
            "unit_to_gene": {},
            "bmp_cost_table": bmp_cost_table,
            "unit_areas": {idx: unit_area for idx in range(genes_num)},
            "zero_probability": float(decision.get("zero_probability", 0.4 if bmp_times else 0.0)),
            "mut_indpb": 0.1,
            "_is_seims": False,
        }

        operators = SolverOperators(source="lightweight")
        return operators, context

    @staticmethod
    def _infer_genes_num(agent_payload: dict, eval_cfg: dict, bmp_types: list) -> int:
        decision = agent_payload.get("optimization_problem", {}).get("decision_variable", {})
        for key in ("candidate_unit_count", "genes_num", "unit_count"):
            if key in decision and decision[key]:
                return int(decision[key])

        surrogate_cfg = eval_cfg.get("surrogate", {})
        model_dir = surrogate_cfg.get("model_dir", "")
        if model_dir:
            import json
            import os

            meta_path = os.path.join(model_dir, "model_metadata.json")
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    return meta.get("input_dim", 10)
                except Exception:
                    pass

        for key in ("swmm_lid", "swmm", "epa_swmm", "swmm_lid_demo"):
            swmm_cfg = eval_cfg.get(key, {})
            binding = swmm_cfg.get("decision_binding", {}) if isinstance(swmm_cfg, dict) else {}
            subcatchments = binding.get("subcatchments", [])
            if isinstance(subcatchments, list) and subcatchments:
                return len(subcatchments)

        for key in ("swat", "swatplus", "swat+"):
            swat_cfg = eval_cfg.get(key, {})
            binding = swat_cfg.get("decision_binding", {}) if isinstance(swat_cfg, dict) else {}
            targets = (
                binding.get("target_lu_mgt_names")
                or binding.get("lu_mgt_names")
                or swat_cfg.get("target_lu_mgt_names")
                or swat_cfg.get("lu_mgt_names")
                or []
            )
            if isinstance(targets, list) and targets:
                return len(targets)

        return 10

    @staticmethod
    def _default_cost_table(bmp_types: list) -> dict:
        ref_costs = {1: 800, 2: 1500, 3: 1200, 4: 3000, 5: 500, 6: 2000, 7: 1000}
        table = {t: ref_costs.get(t, 1000) for t in list(bmp_types)}
        table[0] = 0
        return table
