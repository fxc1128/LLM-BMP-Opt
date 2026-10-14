from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .registry import EncoderRegistry, EvaluatorRegistry, OptimizerRegistry, SimulatorRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLACEHOLDER_EVALUATORS = {"placeholder_metric", "placeholder", "custom_needed"}
SUPPORTED_CONSTRAINTS = {
    "budget_limitation",
    "resource_limitation",
    "demand_satisfaction",
    "spatial_relation",
    "physical_feasibility",
    "bmp_type_limitation",
    "type_limitation",
    "value_range",
    "objective_lower_bound",
    "objective_upper_bound",
    "custom_python",
}


def inspect_runtime_capability(
    agent_payload: Mapping[str, Any] | None,
    code_files: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    payload = dict(agent_payload or {})
    files = dict(code_files or {})
    components: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    missing_paths: list[dict[str, Any]] = []
    custom_files: set[str] = set()

    def component(kind: str, name: str, status: str, reason: str = "", **extra: Any) -> None:
        item = {"type": kind, "name": name, "status": status}
        if reason:
            item["reason"] = reason
        item.update(extra)
        components.append(item)
        if status not in {"ready", "demo_only"}:
            gaps.append(item)

    problem = payload.get("optimization_problem")
    if not isinstance(problem, Mapping):
        component("payload", "optimization_problem", "invalid", "optimization_problem is missing.")
        return _report("invalid_payload", components, gaps, missing_paths, custom_files)

    objectives = problem.get("objectives")
    if not isinstance(objectives, list) or not objectives:
        component("payload", "objectives", "invalid", "At least one objective is required.")

    support_status = str(payload.get("support_status", "unknown")).lower()
    if support_status == "unsupported":
        component("profile", str(payload.get("profile_id", "unknown")), "requires_custom_code", "The selected profile does not implement this demand.")
        custom_files.update({"encoder.py", "simulator.py", "evaluators.py", "constraints.py", "README.md"})

    encoder_cfg = payload.get("encoder", {})
    encoder_name = str(encoder_cfg.get("type", "payload")) if isinstance(encoder_cfg, Mapping) else "payload"
    try:
        decision_space = EncoderRegistry.create(encoder_name).build(payload)
        if not getattr(decision_space, "unit_ids", None):
            raise ValueError("decision space contains no units")
        component("encoder", encoder_name, "ready", unit_count=len(decision_space.unit_ids))
    except Exception as exc:
        component("encoder", encoder_name, "requires_custom_code", str(exc), customization_file="encoder.py")
        custom_files.add("encoder.py")

    simulator_cfg = payload.get("simulator", {})
    simulator_name = str(simulator_cfg.get("type", "legacy")) if isinstance(simulator_cfg, Mapping) else "legacy"
    try:
        SimulatorRegistry.create(simulator_name)
        component("simulator", simulator_name, "ready")
    except Exception as exc:
        component("simulator", simulator_name, "requires_custom_code", str(exc), customization_file="simulator.py")
        custom_files.add("simulator.py")

    solver_cfg = payload.get("solver", {})
    optimizer_name = str(solver_cfg.get("Type", solver_cfg.get("name", "nsga2"))).lower() if isinstance(solver_cfg, Mapping) else "nsga2"
    try:
        OptimizerRegistry.create(optimizer_name, operators=None, cfg={})
        component("optimizer", optimizer_name, "ready")
    except Exception as exc:
        component("optimizer", optimizer_name, "unsupported", str(exc))

    available_evaluators = EvaluatorRegistry.list_available()
    used_evaluators: set[str] = set()
    for index, objective in enumerate(objectives or []):
        if not isinstance(objective, Mapping):
            continue
        evaluator_name = str(objective.get("evaluator", "")).lower().strip()
        used_evaluators.add(evaluator_name)
        if evaluator_name in PLACEHOLDER_EVALUATORS:
            component("evaluator", evaluator_name, "requires_custom_code", "Placeholder objectives are not scientific model results.", objective_index=index, customization_file="evaluators.py")
            custom_files.update({"evaluators.py", "README.md"})
        elif evaluator_name not in available_evaluators:
            component("evaluator", evaluator_name or "missing", "requires_custom_code", "Evaluator is not registered.", objective_index=index, customization_file="evaluators.py")
            custom_files.add("evaluators.py")
        else:
            component("evaluator", evaluator_name, "ready", objective_index=index)

    for index, constraint in enumerate(problem.get("constraints", []) or []):
        if not isinstance(constraint, Mapping):
            component("constraint", f"constraint_{index}", "invalid", "Constraint must be an object.")
            continue
        constraint_type = str(constraint.get("type", "")).lower()
        if constraint_type not in SUPPORTED_CONSTRAINTS:
            component("constraint", constraint_type or f"constraint_{index}", "requires_custom_code", "Constraint type is not implemented.", customization_file="constraints.py")
            custom_files.add("constraints.py")
        else:
            component("constraint", constraint_type, "ready")

    evaluator_cfg = payload.get("evaluator", {}) if isinstance(payload.get("evaluator"), Mapping) else {}
    if used_evaluators.intersection({"seims", "spatiotemporal_seims", "spatial_seims"}):
        from .impl.solvers.seims_plugin import validate_seims_runtime

        for issue in validate_seims_runtime(payload):
            missing_paths.append({"component": "seims", "path": "", "reason": issue})
    if used_evaluators.intersection({"swat", "swatplus", "swat+"}):
        cfg = _first_config(evaluator_cfg, "swat", "swatplus", "swat+")
        _check_path(cfg, "template_dir", "swatplus", True, missing_paths)
        _check_path(cfg, "executable", "swatplus", False, missing_paths)
    if used_evaluators.intersection({"swmm", "swmm_lid", "epa_swmm", "swmm_lid_demo"}):
        cfg = _first_config(evaluator_cfg, "epa_swmm", "swmm_lid", "swmm")
        _check_path(cfg, "model_path", "swmm", False, missing_paths)
        _check_path(cfg, "executable", "swmm", False, missing_paths)

    for filename, content in files.items():
        if filename.endswith(".py") and ("NotImplementedError" in content or "TODO" in content):
            custom_files.add(filename)
            component("code", filename, "requires_custom_code", "File still contains TODO or NotImplementedError.", customization_file=filename)

    # A registry hit does not prove that the configured component can be
    # constructed in the current environment.  Build the runtime graph without
    # calling solve() so missing Python dependencies and invalid evaluator
    # configuration are reported before E4 starts.
    if not custom_files and not missing_paths and not any(item["status"] == "invalid" for item in gaps):
        try:
            from .factory import OptimizationFactory

            OptimizationFactory.create(payload)
            component("runtime", "factory_assembly", "ready")
        except Exception as exc:
            component(
                "runtime",
                "factory_assembly",
                "unsupported",
                f"Runtime assembly failed: {exc}",
            )

    if any(item["status"] == "invalid" for item in gaps):
        status = "invalid_payload"
    elif custom_files or any(item["status"] == "requires_custom_code" for item in gaps):
        status = "requires_custom_code"
    elif missing_paths or support_status == "requires_local_paths":
        status = "requires_local_paths"
    elif any(item["status"] == "unsupported" for item in gaps):
        status = "unsupported"
    else:
        status = "ready"
    return _report(status, components, gaps, missing_paths, custom_files)


def _first_config(configs: Mapping[str, Any], *names: str) -> Mapping[str, Any]:
    for name in names:
        value = configs.get(name)
        if isinstance(value, Mapping):
            return value
    return {}


def _check_path(config: Mapping[str, Any], key: str, component: str, directory: bool, missing: list[dict[str, Any]]) -> None:
    raw = str(config.get(key, "") or "")
    if not raw:
        missing.append({"component": component, "path": key, "reason": f"{key} is not configured."})
        return
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        missing.append({"component": component, "path": str(path), "reason": f"{key} does not exist."})


def _report(
    status: str,
    components: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    missing_paths: list[dict[str, Any]],
    custom_files: set[str],
) -> dict[str, Any]:
    return {
        "status": status,
        "ready": status == "ready",
        "scientific_execution_allowed": status == "ready",
        "components": components,
        "gaps": gaps,
        "missing_paths": missing_paths,
        "customization_files": sorted(custom_files),
        "message": {
            "ready": "Runtime capability check passed. The optimization can be executed.",
            "requires_local_paths": "Local model files or executables must be configured before execution.",
            "requires_custom_code": "Custom runtime components must be implemented before scientific execution.",
            "unsupported": "The requested runtime configuration is unsupported.",
            "invalid_payload": "The optimization payload is invalid.",
        }.get(status, "Runtime capability check failed."),
    }
