"""
OptimizationFactory: AGENT_PAYLOAD → (OptimizationProblem, UnifiedSolver)

自动检测路径：
  SEIMS路径（有MongoDB + SEIMS-dev）→ 空间感知算子 + 真实成本
  轻量路径（无依赖）               → 通用算子 + 简化成本

两条路径共用同一个 OptimizationProblem + UnifiedSolver，
差异只在 context dict 内容和 SolverOperators 实现。
"""
from __future__ import annotations

from .core.problem import OptimizationProblem
from .registry import EncoderRegistry, EvaluatorRegistry, OptimizerRegistry, SimulatorRegistry
from .core.interfaces import DecisionSpace
from .impl.constraints.budget_constraint import BudgetConstraint
from .impl.constraints.func_constraint import CustomFunctionConstraint
from .impl.constraints.structured_constraint import StructuredConstraint, normalize_bmp_types
from .impl.solvers.unified_solver import UnifiedSolver
from .impl.solvers.seims_plugin import SEIMSPlugin
from .utils.lightweight_builder import LightweightBuilder


class OptimizationFactory:
    """
    从 AGENT_PAYLOAD 构造 (OptimizationProblem, UnifiedSolver)。

    自动选择路径：
      - SEIMS可用 → SEIMSPlugin.build_operators() + SEIMSPlugin.build_context()
      - SEIMS不可用 → LightweightBuilder.build()
    """

    @staticmethod
    def create(agent_payload: dict):
        agent_payload = OptimizationFactory._normalize_payload(agent_payload)
        """
        Parameters
        ----------
        agent_payload: E2输出的数学模型JSON

        Returns
        -------
        (OptimizationProblem, UnifiedSolver)
        """
        # ── Step 1: 选择路径，构造 operators + context ────────────────────────
        force_lightweight = bool(agent_payload.get("force_lightweight_runtime"))
        if (not force_lightweight) and SEIMSPlugin.is_available(agent_payload):
            print("[OptimizationFactory] SEIMS路径：使用空间感知算子")
            from .bridge.config_bridge import ConfigBridge
            raw_cfg, is_temporal = ConfigBridge.create_legacy_config_object(agent_payload)
            operators = SEIMSPlugin.build_operators(raw_cfg, is_temporal)
            context   = SEIMSPlugin.build_context(agent_payload, raw_cfg)
        else:
            print("[OptimizationFactory] 轻量路径：使用通用算子")
            operators, context = LightweightBuilder.build(agent_payload)
            is_temporal = False

        # Step 1.5: build the explicit decision space before any individual is evaluated.
        encoder_cfg = agent_payload.get("encoder", {})
        encoder_name = encoder_cfg.get("type", "payload") if isinstance(encoder_cfg, dict) else "payload"
        try:
            decision_space = EncoderRegistry.create(encoder_name).build(agent_payload)
        except ValueError:
            if not context.get("_is_seims"):
                raise
            gene_to_unit = {int(key): str(value) for key, value in context.get("gene_to_unit", {}).items()}
            decision_space = DecisionSpace(
                unit_ids=[gene_to_unit[index] for index in sorted(gene_to_unit)],
                gene_domains=[list(context.get("bmp_types", [])) for _ in gene_to_unit],
                gene_to_unit=gene_to_unit,
                unit_to_gene={str(value): int(key) for key, value in context.get("gene_to_unit", {}).items()},
            )
        context["decision_space"] = decision_space
        context.setdefault("gene_to_unit", decision_space.gene_to_unit)
        context.setdefault("unit_to_gene", decision_space.unit_to_gene)
        context.setdefault("genes_num", len(decision_space.unit_ids))
        simulator_cfg = agent_payload.get("simulator", {})
        simulator_name = simulator_cfg.get("type", "legacy") if isinstance(simulator_cfg, dict) else "legacy"
        simulator = SimulatorRegistry.create(simulator_name)

        # ── Step 2: 构造评估器列表 ────────────────────────────────────────────
        context["_agent_payload"] = agent_payload
        context["_evaluator_cfgs"] = agent_payload.get("evaluator", {})

        prob_def     = agent_payload.get("optimization_problem", {})
        evaluator_cfgs = agent_payload.get("evaluator", {})
        evaluators   = []
        weights      = []

        is_seims_path = context.get("_is_seims", False)

        objectives = prob_def.get("objectives", [])
        for obj in objectives:
            eval_type = obj.get("evaluator", "")
            obj_type  = obj.get("type", "min")

            eval_key = eval_type.lower().strip()
            eval_specific = OptimizationFactory._objective_eval_config(
                evaluator_cfgs,
                eval_key,
                obj,
            )
            if not isinstance(eval_specific, dict):
                eval_specific = {}
            if eval_type.lower().strip() in ("swmm", "swmm_lid", "epa_swmm", "swmm_lid_demo"):
                eval_specific = dict(eval_specific)
                eval_specific["metric"] = obj.get("name", eval_specific.get("metric", "runoff_volume"))
            model_dir = eval_specific.get("model_dir", None)
            code      = obj.get("code", None)

            # 评估器构造：传入 context（dict或raw_cfg均可）
            raw_cfg_for_ev = context.get("_raw_cfg", None)

            # 轻量路径下自动降级评估器类型
            raw_cfg_for_ev = context.get("_raw_cfg", None)
            effective_eval_type = eval_type
            if eval_key == "custom_needed" or eval_key.startswith("placeholder"):
                effective_eval_type = "placeholder_metric"
                eval_specific = dict(eval_specific)
                eval_specific.setdefault("objective_name", obj.get("name", eval_key))
                eval_specific.setdefault("placeholder", True)
            if not is_seims_path and raw_cfg_for_ev is None:
                lkey = eval_type.lower().strip()
                if lkey == "seims":
                    # SEIMS不可用，降级为surrogate（若有model_dir）或占位
                    if model_dir:
                        effective_eval_type = "surrogate"
                        print(f"[OptimizationFactory] 轻量路径: {eval_type} -> surrogate")
                    else:
                        print(f"[OptimizationFactory] 轻量路径: {eval_type} 无法使用，"
                              f"使用占位评估器(返回随机值)")
                        effective_eval_type = "placeholder_metric"
                        eval_specific = dict(eval_specific)
                        eval_specific.setdefault("objective_name", obj.get("name", lkey))
                        eval_specific.setdefault("placeholder", True)
                        eval_specific.setdefault(
                            "reason",
                            "SEIMS evaluator is unavailable and no surrogate model_dir was provided.",
                        )
                elif lkey in ("spatiotemporal_seims", "spatial_seims") or (
                    lkey.startswith("seims_")
                    and any(token in lkey for token in ("early", "late", "window", "sediment"))
                ):
                    # Masked SEIMS metrics require raw SEIMS cfg and raster outputs. In
                    # lightweight mode keep the generated code runnable and surface the
                    # dependency through stdout instead of crashing E4.
                    if model_dir:
                        effective_eval_type = "surrogate"
                        print(f"[OptimizationFactory] 轻量路径: {eval_type} -> surrogate")
                    else:
                        print(f"[OptimizationFactory] 轻量路径: {eval_type} 需要SEIMS输出，"
                              f"使用占位评估器(返回0.0)")
                        effective_eval_type = "placeholder_metric"
                        eval_specific = dict(eval_specific)
                        eval_specific.setdefault("objective_name", obj.get("name", lkey))
                        eval_specific.setdefault("placeholder", True)
                        eval_specific.setdefault(
                            "reason",
                            "Spatiotemporal SEIMS evaluator is unavailable and no surrogate model_dir was provided.",
                        )
                elif lkey in ("bmp_cost", "bmp_cost_model", "bmp_net_cost_model", "bmp_net_cost"):
                    effective_eval_type = "lightweight_cost"
                    print(f"[OptimizationFactory] 轻量路径: {eval_type} -> lightweight_cost")

            if force_lightweight and eval_key in (
                "swat",
                "swatplus",
                "swat+",
                "swmm",
                "swmm_lid",
                "epa_swmm",
                "swmm_lid_demo",
            ):
                effective_eval_type = "placeholder_metric"
                eval_specific = dict(eval_specific)
                eval_specific.setdefault("objective_name", obj.get("name", eval_key))
                eval_specific.setdefault("placeholder", True)
                eval_specific.setdefault(
                    "reason",
                    "External simulator objective is represented by a runnable placeholder in lightweight E3 validation.",
                )

            ev = EvaluatorRegistry.create(
                evaluator_type=effective_eval_type,
                raw_cfg=raw_cfg_for_ev,
                is_temporal=is_temporal,
                model_dir=model_dir,
                code=code,
                context=context,
                eval_config=eval_specific,
            )
            evaluators.append(ev)
            weights.append(1.0 if obj_type == "max" else -1.0)

        if not evaluators:
            raise ValueError(
                "[OptimizationFactory] objectives 为空，无法构造评估器列表"
            )

        # ── Step 3: 构造约束列表 ──────────────────────────────────────────────
        cost_obj_index = OptimizationFactory._infer_cost_objective_index(objectives)
        constraints = []
        for cons in prob_def.get("constraints", []):
            c_type = cons.get("type", "").lower()
            if c_type in ("budget_limitation", "resource_limitation"):
                constraints.append(StructuredConstraint(cons, default_target_index=cost_obj_index))
                if c_type == "budget_limitation" and not any(
                    key in cons for key in ("threshold", "value", "limit", "rhs", "target")
                ):
                    constraints.append(BudgetConstraint(cons.get("budget", 1e8)))
            elif c_type in (
                "demand_satisfaction",
                "spatial_relation",
                "physical_feasibility",
                "bmp_type_limitation",
                "type_limitation",
                "value_range",
                "objective_lower_bound",
                "objective_upper_bound",
            ):
                constraints.append(StructuredConstraint(cons))
            elif c_type == "custom_python" and "code" in cons:
                constraints.append(CustomFunctionConstraint(cons["code"]))

        # ── Step 4: 组装 OptimizationProblem ─────────────────────────────────
        problem = OptimizationProblem(
            name="BMP_Opt_Problem",
            context=context,
            evaluators=evaluators,
            constraints=constraints,
            weights=tuple(weights),
            simulator=simulator,
            decision_space=decision_space,
        )

        # ── Step 5: 构造 UnifiedSolver ────────────────────────────────────────
        solver_cfg_raw = agent_payload.get("solver", {})
        solver_cfg = {
            "n_pop":    solver_cfg_raw.get("PopulationSize", 12),
            "n_gen":    solver_cfg_raw.get("GenerationsNum", 4),
            "cx_prob":  solver_cfg_raw.get("CrossoverRate",  0.8),
            "mut_prob": solver_cfg_raw.get("MutateRate",     0.1),
        }
        optimizer_name = str(solver_cfg_raw.get("Type", "nsga2")).lower()
        solver = OptimizerRegistry.create(optimizer_name, operators=operators, cfg=solver_cfg)

        return problem, solver

    @staticmethod
    def _infer_cost_objective_index(objectives: list) -> int:
        for index, obj in enumerate(objectives):
            if not isinstance(obj, dict):
                continue
            text = " ".join(
                str(obj.get(key, ""))
                for key in ("name", "indicator", "evaluator", "type")
            ).lower()
            if any(token in text for token in ("cost", "economy", "expense", "investment", "budget")):
                return index
        return 0

    @staticmethod
    def _normalize_payload(agent_payload: dict) -> dict:
        prob = agent_payload.get("optimization_problem", {})
        decision_var = prob.get("decision_variable", {})
        bmp_value = decision_var.get("BMP_value", {})
        if isinstance(bmp_value, dict) and "BMP_type" in bmp_value:
            bmp_value["BMP_type"] = normalize_bmp_types(bmp_value.get("BMP_type"))
        return agent_payload

    @staticmethod
    def _objective_eval_config(evaluator_cfgs: dict, eval_key: str, obj: dict) -> dict:
        base = evaluator_cfgs.get(eval_key, {}) if isinstance(evaluator_cfgs, dict) else {}
        if not isinstance(base, dict):
            return {}

        merged = dict(base)
        objective_name = str(obj.get("name", "") or "")
        candidates = []
        for key in ("future_objective_configs", "objective_configs"):
            value = base.get(key)
            if isinstance(value, list):
                candidates.extend(item for item in value if isinstance(item, dict))

        for candidate in candidates:
            candidate_name = str(
                candidate.get("objective_name")
                or candidate.get("name")
                or candidate.get("target_objective")
                or ""
            )
            if candidate_name and candidate_name == objective_name:
                merged.update(candidate)
                merged.setdefault("objective_name", objective_name)
                break

        if objective_name:
            merged.setdefault("objective_name", objective_name)
        return merged
