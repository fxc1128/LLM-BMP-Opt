"""
SEIMSPlugin: 将SEIMS空间感知算子封装为可选插件。

当SEIMS环境可用时（有MongoDB + pygeoc + SEIMS-dev），
OptimizationFactory自动使用此插件替换轻量默认算子，获得：
  - 坡位适宜性感知的种群初始化
  - 空间结构感知的交叉算子（crossover_slppos）
  - 规则约束的变异算子（mutate_rule_s）

当SEIMS不可用时，静默降级为轻量默认算子，接口完全一致。
"""
from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING, List
from pathlib import Path
import os
import random
import tempfile
from ..constraints.structured_constraint import normalize_bmp_types

if TYPE_CHECKING:
    from .unified_solver import SolverOperators

# ─── SEIMS可用性检测 ──────────────────────────────────────────────────────────

def _try_import_seims():
    """尝试导入SEIMS Legacy模块，返回 (available, modules_dict)"""
    try:
        import os
        import sys
        seims_path = os.environ.get("SEIMS_PATH", "")
        if not seims_path:
            return False, {}
        if seims_path not in sys.path:
            sys.path.insert(0, seims_path)
        from scenario_analysis.spatialunits.scenario import (
            SUScenario, initialize_scenario, initialize_scenario_s_t
        )
        from scenario_analysis.spatialunits.userdef import (
            crossover_slppos, crossover_updown, crossover_rdm,
            mutate_rule_s, mutate_rule_s_t
        )
        return True, {
            "SUScenario": SUScenario,
            "initialize_scenario": initialize_scenario,
            "initialize_scenario_s_t": initialize_scenario_s_t,
            "crossover_slppos": crossover_slppos,
            "crossover_updown": crossover_updown,
            "crossover_rdm": crossover_rdm,
            "mutate_rule_s": mutate_rule_s,
            "mutate_rule_s_t": mutate_rule_s_t,
        }
    except ImportError:
        return False, {}


# ─── SEIMSPlugin ─────────────────────────────────────────────────────────────

def _is_writable_directory(path_str: str) -> tuple[bool, str]:
    path = Path(path_str)
    if path.is_file():
        return False, f"path is a file, not a directory: {path}"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return False, f"cannot create directory {path}: {exc}"
    try:
        with tempfile.NamedTemporaryFile(prefix="e4_probe_", suffix=".tmp", dir=str(path), delete=True) as tmp:
            tmp.write(b"ok")
            tmp.flush()
        return True, ""
    except Exception as exc:
        return False, f"directory is not writable: {path} ({exc})"


def validate_seims_runtime(agent_payload: dict) -> List[str]:
    """Return blocking runtime issues for executing a SEIMS-backed payload."""
    issues: List[str] = []
    seims_ok, _ = _try_import_seims()
    if not seims_ok:
        issues.append("SEIMS legacy modules are not importable. Check SEIMS_PATH and Python dependencies.")
        return issues

    eval_cfg = agent_payload.get("evaluator", {}) if isinstance(agent_payload, dict) else {}
    objectives = agent_payload.get("optimization_problem", {}).get("objectives", []) if isinstance(agent_payload, dict) else []
    used_evaluators = {
        str(obj.get("evaluator", "")).lower()
        for obj in objectives
        if isinstance(obj, dict)
    }
    if "seims" in used_evaluators:
        seims_cfg = eval_cfg.get("seims") or {}
    elif "spatiotemporal_seims" in used_evaluators:
        seims_cfg = {**(eval_cfg.get("seims") or {}), **(eval_cfg.get("spatiotemporal_seims") or {})}
    elif "spatial_seims" in used_evaluators:
        seims_cfg = {**(eval_cfg.get("seims") or {}), **(eval_cfg.get("spatial_seims") or {})}
    else:
        return issues

    model_dir = seims_cfg.get("MODEL_DIR") or seims_cfg.get("model_dir") or ""
    if not model_dir:
        issues.append("SEIMS evaluator is selected but MODEL_DIR is missing.")
        return issues
    if not os.path.isdir(model_dir):
        issues.append(f"MODEL_DIR does not exist: {model_dir}")

    bin_dir = seims_cfg.get("BIN_DIR") or seims_cfg.get("bin_dir") or ""
    if bin_dir and not os.path.isdir(bin_dir):
        issues.append(f"BIN_DIR does not exist: {bin_dir}")

    graph_path = seims_cfg.get("spatial_graph_path") or ""
    if graph_path and not os.path.isfile(graph_path):
        issues.append(f"spatial_graph_path does not exist: {graph_path}")

    try:
        from ...bridge.config_bridge import ConfigBridge
        raw_cfg, _ = ConfigBridge.create_legacy_config_object(agent_payload)
        out_dir = getattr(getattr(raw_cfg, "opt", None), "out_dir", None)
        if out_dir:
            writable, detail = _is_writable_directory(str(out_dir))
            if not writable:
                issues.append(f"SEIMS output directory is not writable: {detail}")
    except Exception as exc:
        issues.append(f"failed to construct SEIMS runtime config: {exc}")

    return issues


class SEIMSPlugin:
    """
    SEIMS空间感知算子插件。

    用法：
        operators, context = SEIMSPlugin.build(agent_payload)
        # operators 是 SolverOperators，context 是 dict
        # 若SEIMS不可用，自动降级为轻量默认算子
    """

    @staticmethod
    def is_available(agent_payload: dict) -> bool:
        """
        判断是否可以使用SEIMS插件：
        1. SEIMS Legacy模块可导入
        2. evaluator配置中有 seims 或 bmp_net_cost_model 键（含MODEL_DIR）
        3. MODEL_DIR 路径存在
        """
        import os
        seims_ok, _ = _try_import_seims()
        if not seims_ok:
            return False

        eval_cfg = agent_payload.get("evaluator", {})
        objectives = agent_payload.get("optimization_problem", {}).get("objectives", [])
        used_evaluators = {
            str(obj.get("evaluator", "")).lower()
            for obj in objectives
            if isinstance(obj, dict)
        }
        base_seims_cfg = eval_cfg.get("seims") or {}
        if "seims" in used_evaluators:
            seims_cfg = base_seims_cfg
        elif "spatiotemporal_seims" in used_evaluators:
            seims_cfg = {**base_seims_cfg, **(eval_cfg.get("spatiotemporal_seims") or {})}
        elif "spatial_seims" in used_evaluators:
            seims_cfg = {**base_seims_cfg, **(eval_cfg.get("spatial_seims") or {})}
        else:
            return False
        model_dir = seims_cfg.get("MODEL_DIR") or seims_cfg.get("model_dir") or ""
        return bool(model_dir) and os.path.isdir(model_dir)

    @staticmethod
    def build_context(agent_payload: dict, raw_cfg) -> dict:
        """
        从 SASlpPosConfig 构造统一 dict context。

        dict context 与轻量版格式完全一致，评估器签名不变。
        """
        prob = agent_payload.get("optimization_problem", {})
        bmp_types = normalize_bmp_types(
            prob.get("decision_variable", {}).get("BMP_value", {}).get("BMP_type", [1, 2])
        )

        ctx = {
            # 空间单元信息（从raw_cfg读取）
            "units_infos":  getattr(raw_cfg, "units_infos", {}),
            "gene_to_unit": getattr(raw_cfg, "gene_to_unit", {}),
            "unit_to_gene": getattr(raw_cfg, "unit_to_gene", {}),
            "genes_num":    getattr(raw_cfg, "genes_num", len(getattr(raw_cfg, "gene_to_unit", {}))),
            "bmp_types":    bmp_types,
            "bmp_time_values": normalize_bmp_types(
                prob.get("decision_variable", {}).get("BMP_value", {}).get("BMP_time", []),
                default=[],
            ),
            # 变异参数
            "mut_indpb":    getattr(getattr(raw_cfg, "opt", None), "rmut", 0.1),
            # SEIMS专用：保留raw_cfg供评估器使用
            "_raw_cfg":     raw_cfg,
            "_is_seims":    True,
        }
        SEIMSPlugin._attach_action_domain(ctx, agent_payload, raw_cfg)
        return ctx

    @staticmethod
    def _attach_action_domain(ctx: dict, agent_payload: dict, raw_cfg) -> None:
        try:
            from src.knowledge.action_domain import ActionDomainResolver
            from src.knowledge.spatial_graph_builder import SEIMSSpatialGraphBuilder
            from src.knowledge.watershed_spatial_graph import WatershedSpatialGraph
        except ImportError as exc:
            print(f"[SEIMSPlugin] action domain disabled, import failed: {exc}")
            return

        eval_cfgs = agent_payload.get("evaluator", {})
        seims_cfg = (
            eval_cfgs.get("spatiotemporal_seims")
            or eval_cfgs.get("spatial_seims")
            or eval_cfgs.get("seims")
            or {}
        )
        spatial_scope_text = (
            seims_cfg.get("spatial_scope")
            or agent_payload.get("basic_info", {}).get("spatial_scope")
            or "full_watershed"
        )
        model_dir = Path(getattr(raw_cfg.model, "model_dir", ""))
        graph_path = Path(
            seims_cfg.get("spatial_graph_path")
            or Path.cwd() / "src" / "knowledge" / "data" / f"{model_dir.name}_spatial_graph.json"
        )
        if seims_cfg.get("spatial_graph_auto_build", False) or not graph_path.is_file():
            SEIMSSpatialGraphBuilder(model_dir).write(graph_path)

        graph = WatershedSpatialGraph.load(graph_path)
        evaluation_scope = graph.resolve(spatial_scope_text)
        action_domain = ActionDomainResolver(graph).resolve(
            raw_cfg,
            evaluation_scope,
            seims_cfg.get("action_domain", {}),
        )
        ctx["evaluation_scope"] = evaluation_scope
        ctx["action_domain"] = action_domain
        ctx["action_domain_mode"] = action_domain.mode
        ctx["action_domain_unit_type"] = action_domain.unit_type
        ctx["action_domain_allowed_unit_ids"] = action_domain.allowed_unit_ids
        ctx["action_domain_allowed_gene_indices"] = action_domain.allowed_gene_indices
        ctx["action_domain_locked_gene_indices"] = action_domain.locked_gene_indices
        ctx["action_domain_reason"] = action_domain.reason
        print(
            "[SEIMSPlugin] action domain resolved: "
            f"mode={action_domain.mode}, allowed_genes={len(action_domain.allowed_gene_indices)}, "
            f"locked_genes={len(action_domain.locked_gene_indices)}"
        )

    @staticmethod
    def build_operators(raw_cfg, is_temporal: bool) -> "SolverOperators":
        """
        构造SEIMS空间感知算子，包装为统一签名。

        Parameters
        ----------
        raw_cfg:     SASlpPosConfig 对象
        is_temporal: 是否时空优化模式
        """
        from .unified_solver import SolverOperators

        seims_ok, mods = _try_import_seims()
        if not seims_ok:
            print("[SEIMSPlugin] SEIMS不可用，降级为轻量算子")
            return SolverOperators(source="lightweight")

        # 预加载静态数据（避免每次变异都查数据库）
        try:
            temp_sce = mods["SUScenario"](raw_cfg)
            suit_bmps  = temp_sce.suit_bmps
            bmps_grade = temp_sce.bmps_grade
        except Exception as e:
            print(f"[SEIMSPlugin] 预加载suit_bmps失败: {e}，降级为轻量算子")
            return SolverOperators(source="lightweight")

        unit_type = getattr(raw_cfg, "bmps_cfg_unit", "SLPPOS")

        # ── 初始化算子 ────────────────────────────────────────────────────────
        fast_native_operators = bool(getattr(raw_cfg, "fast_native_operators", False))
        if fast_native_operators:
            allowed_periods = normalize_bmp_types(
                getattr(raw_cfg, "allowed_bmp_time_values", []),
                default=[],
            )

            def _sample_gene(bmp_types: list[int], nonzero_prob: float) -> int:
                if random.random() >= nonzero_prob:
                    return 0
                bmp = random.choice(bmp_types)
                if is_temporal:
                    periods = allowed_periods or list(range(1, int(getattr(raw_cfg, "change_times", 1)) + 1))
                    return int(bmp) * 1000 + int(random.choice(periods))
                return int(bmp)

            def fast_initializer(context: dict):
                from src.knowledge.action_domain import apply_action_domain
                bmp_types = list(context.get("bmp_types", [1, 2, 3, 4]))
                nonzero_prob = float(getattr(raw_cfg, "fast_native_nonzero_prob", 0.05))
                genes = [
                    _sample_gene(bmp_types, nonzero_prob)
                    for _ in range(context.get("genes_num", 0))
                ]
                return apply_action_domain(genes, context)

            def fast_crossover(ind1, ind2, context: dict):
                from deap import tools
                from src.knowledge.action_domain import apply_action_domain
                if len(ind1) > 1 and len(ind2) > 1:
                    tools.cxTwoPoint(ind1, ind2)
                apply_action_domain(ind1, context)
                apply_action_domain(ind2, context)
                return ind1, ind2

            def fast_mutation(ind, context: dict):
                from src.knowledge.action_domain import apply_action_domain
                bmp_types = list(context.get("bmp_types", [1, 2, 3, 4]))
                indpb = float(context.get("mut_indpb", 0.1))
                nonzero_prob = float(getattr(raw_cfg, "fast_native_nonzero_prob", 0.05))
                for idx in range(len(ind)):
                    if random.random() < indpb:
                        ind[idx] = _sample_gene(bmp_types, nonzero_prob)
                apply_action_domain(ind, context)
                return ind

            print("[SEIMSPlugin] fast native smoke operators enabled")
            return SolverOperators(
                initializer=fast_initializer,
                crossover=fast_crossover,
                mutation=fast_mutation,
                source="seims-fast",
            )

        init_fn = (mods["initialize_scenario_s_t"] if is_temporal
                   else mods["initialize_scenario"])

        def seims_initializer(context: dict):
            from src.knowledge.action_domain import apply_action_domain
            return apply_action_domain(list(init_fn(raw_cfg)), context)

        # ── 交叉算子 ──────────────────────────────────────────────────────────
        if unit_type == "SLPPOS":
            _cx_fn = mods["crossover_slppos"]
            _cx_kwargs = {
                "hillslp_values_num": getattr(raw_cfg, "hillslp_genes_num", 0),
                "fixed_positions":    getattr(raw_cfg, "key_bmps", {}),
            }
            _cx_mode = "slppos"
        elif unit_type == "CONNFIELD":
            _cx_fn = mods["crossover_updown"]
            _cx_kwargs = {
                "updownunits": raw_cfg.updown_units,
                "gene2unit":   raw_cfg.gene_to_unit,
                "unit2gene":   raw_cfg.unit_to_gene,
            }
            _cx_mode = "updown"
        else:
            _cx_fn = mods["crossover_rdm"]
            _cx_kwargs = {}
            _cx_mode = "random"

        def seims_crossover(ind1, ind2, context: dict):
            from src.knowledge.action_domain import apply_action_domain
            if _cx_mode == "updown":
                _cx_fn(
                    _cx_kwargs["updownunits"],
                    _cx_kwargs["gene2unit"],
                    _cx_kwargs["unit2gene"],
                    ind1,
                    ind2,
                )
            else:
                _cx_fn(ind1, ind2, **_cx_kwargs)
            apply_action_domain(ind1, context)
            apply_action_domain(ind2, context)
            return ind1, ind2

        # ── 变异算子 ──────────────────────────────────────────────────────────
        _mut_kwargs = {
            "unitsinfo":       getattr(raw_cfg, "units_infos", {}),
            "gene2unit":       getattr(raw_cfg, "gene_to_unit", {}),
            "unit2gene":       getattr(raw_cfg, "unit_to_gene", {}),
            "suitbmps":        suit_bmps,
            "fixed_positions": getattr(raw_cfg, "key_bmps", {}),
            "perc":            getattr(getattr(raw_cfg, "opt", None), "pmut", 0.2),
            "indpb":           getattr(getattr(raw_cfg, "opt", None), "rmut", 0.1),
            "unit":            unit_type,
            "method":          getattr(raw_cfg, "bmps_cfg_method", "HILLSLP"),
            "bmpgrades":       bmps_grade,
            "tagnames":        getattr(raw_cfg, "slppos_tagnames", None),
        }

        if is_temporal:
            _mut_fn = mods["mutate_rule_s_t"]
            _mut_extra = {"low": 1, "up": getattr(raw_cfg, "change_times", 1)}
        else:
            _mut_fn = mods["mutate_rule_s"]
            _mut_extra = {}

        def seims_mutation(ind, context: dict):
            from src.knowledge.action_domain import apply_action_domain
            _mut_fn(individual=ind, **_mut_kwargs, **_mut_extra)
            apply_action_domain(ind, context)
            return ind

        return SolverOperators(
            initializer=seims_initializer,
            crossover=seims_crossover,
            mutation=seims_mutation,
            source="seims",
        )
