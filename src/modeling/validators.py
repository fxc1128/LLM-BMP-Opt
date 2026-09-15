from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:
    from src.knowledge.metrics.schema import MetricSpec, UnsupportedMetricSpec
    from src.knowledge.watershed_spatial_graph import WatershedSpatialGraph
except ImportError:  # The open release validates payload structure without private spatial knowledge assets.
    MetricSpec = None
    UnsupportedMetricSpec = ValueError
    WatershedSpatialGraph = Any


ALLOWED = {
    "stage": {
        "demand_unit",
        "domain_resolution",
        "optimization_model",
        "executable_payload",
        "runtime_result",
    },
    "demand_role": {"objective", "constraint"},
    "phenomenon": {"sediment", "runoff", "nutrient", "cost", "landuse", "implementation", "custom"},
    "variable": {"sediment_load", "sediment_intensity", "investment_cost", "custom"},
    "source": {"raster", "timeseries", "cost_model", "rule", "custom"},
    "spatial_aggregation": {"sum", "mean", "weighted_mean", "max"},
    "temporal_aggregation": {"window_total", "annualized"},
    "comparison": {"raw_value", "reduction_rate"},
    "peak_reference": {"none", "baseline_location", "scenario_peak"},
    "sense": {"maximize", "minimize", "<=", ">=", "=="},
    "domain_mode": {"within", "intersect", "upstream", "full_watershed", "explicit", "auto"},
    "domain_type": {"watershed", "subbasin", "field", "admin", "reach", "outlet", "custom", "mixed"},
    "optimizer": {"nsga2", "ga", "de", "random_search", "custom"},
    "status": {"completed", "failed", "needs_revision", "not_run"},
}


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str
    allowed: Optional[List[str]] = None


@dataclass
class ValidationReport:
    stage: str
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, path: str, message: str, allowed: Optional[Iterable[str]] = None) -> None:
        self.errors.append(ValidationIssue(path, message, _sorted_or_none(allowed)))

    def add_warning(self, path: str, message: str, allowed: Optional[Iterable[str]] = None) -> None:
        self.warnings.append(ValidationIssue(path, message, _sorted_or_none(allowed)))

    def feedback(self) -> str:
        if self.ok and not self.warnings:
            return "Validation passed. The artifact can be passed to the next expert."

        lines = [
            f"Validation feedback for stage '{self.stage}'.",
            "Revise only this stage output. Preserve valid fields unless they conflict with an error.",
        ]
        if self.errors:
            lines.append("Errors:")
            for issue in self.errors:
                lines.append(_format_issue(issue))
        if self.warnings:
            lines.append("Warnings:")
            for issue in self.warnings:
                lines.append(_format_issue(issue))
        return "\n".join(lines)


def validate_artifact(
    stage: str | None,
    artifact: Mapping[str, Any],
    spatial_graph_path: str | Path | None = None,
) -> ValidationReport:
    stage_name = stage or str(artifact.get("stage", ""))
    report = ValidationReport(stage=stage_name)
    if not isinstance(artifact, Mapping):
        report.add_error("$", "Artifact must be a JSON object.")
        return report

    if stage_name not in ALLOWED["stage"]:
        report.add_error("$.stage", f"Unsupported stage: {stage_name}", ALLOWED["stage"])
        return report

    graph = _load_graph(spatial_graph_path, report) if spatial_graph_path else None

    if stage_name == "demand_unit":
        _validate_demand_unit(artifact, report)
    elif stage_name == "domain_resolution":
        _validate_domain_resolution(artifact, report, graph)
    elif stage_name == "optimization_model":
        _validate_optimization_model(artifact, report)
    elif stage_name == "executable_payload":
        _validate_executable_payload(artifact, report)
    elif stage_name == "runtime_result":
        _validate_runtime_result(artifact, report)

    return report


def _validate_demand_unit(artifact: Mapping[str, Any], report: ValidationReport) -> None:
    _require_value(artifact, "stage", "$", report)
    units = _require_list(artifact, "demand_units", "$", report)
    for idx, unit in enumerate(units):
        path = f"$.demand_units[{idx}]"
        if not isinstance(unit, Mapping):
            report.add_error(path, "Demand unit must be an object.")
            continue

        _require_value(unit, "id", path, report)
        _enum(unit.get("role"), "demand_role", f"{path}.role", report)

        quantity = _require_object(unit, "quantity", path, report)
        if quantity is not None:
            _enum(quantity.get("phenomenon"), "phenomenon", f"{path}.quantity.phenomenon", report)
            _enum(quantity.get("variable"), "variable", f"{path}.quantity.variable", report)
            _enum(quantity.get("source"), "source", f"{path}.quantity.source", report)

        metric = _require_object(unit, "metric", path, report)
        if metric is not None:
            _validate_metric(metric, f"{path}.metric", report)

        criterion = _require_object(unit, "criterion", path, report)
        if criterion is not None:
            _enum(criterion.get("sense"), "sense", f"{path}.criterion.sense", report)

        scope = _require_object(unit, "scope", path, report)
        if scope is not None:
            _require_value(scope, "raw_spatial_text", f"{path}.scope", report)
            _require_value(scope, "raw_temporal_text", f"{path}.scope", report)


def _validate_domain_resolution(
    artifact: Mapping[str, Any],
    report: ValidationReport,
    graph: WatershedSpatialGraph | None,
) -> None:
    resolutions = _require_list(artifact, "resolutions", "$", report)
    for idx, item in enumerate(resolutions):
        path = f"$.resolutions[{idx}]"
        if not isinstance(item, Mapping):
            report.add_error(path, "Domain resolution must be an object.")
            continue
        _require_value(item, "demand_unit_id", path, report)
        for name in ("temporal_domain", "evaluation_domain", "action_domain", "simulation_domain"):
            domain = _require_object(item, name, path, report)
            if domain is None:
                continue
            if name in ("evaluation_domain", "simulation_domain"):
                _enum(domain.get("domain_type"), "domain_type", f"{path}.{name}.domain_type", report)
            if name == "action_domain":
                _enum(domain.get("mode"), "domain_mode", f"{path}.action_domain.mode", report)
                _enum(domain.get("domain_type"), "domain_type", f"{path}.action_domain.domain_type", report)
            if graph and name in ("evaluation_domain", "action_domain", "simulation_domain"):
                _validate_node_ids(domain, f"{path}.{name}", report, graph)


def _validate_optimization_model(artifact: Mapping[str, Any], report: ValidationReport) -> None:
    _require_value(artifact, "problem_type", "$", report)
    simulator = _require_object(artifact, "simulator", "$", report)
    if simulator is not None:
        _require_value(simulator, "name", "$.simulator", report)

    decision_variables = _require_object(artifact, "decision_variables", "$", report)
    if decision_variables is not None:
        _require_value(decision_variables, "unit_type", "$.decision_variables", report)
        _require_value(decision_variables, "encoding", "$.decision_variables", report)

    objectives = _require_list(artifact, "objectives", "$", report)
    constraints = artifact.get("constraints", [])
    if not isinstance(constraints, list):
        report.add_error("$.constraints", "constraints must be a list when present.")
        constraints = []
    for idx, objective in enumerate(objectives):
        _validate_model_term(objective, f"$.objectives[{idx}]", report, objective=True)
    for idx, constraint in enumerate(constraints):
        _validate_model_term(constraint, f"$.constraints[{idx}]", report, objective=False)

    optimizer = _require_object(artifact, "optimizer", "$", report)
    if optimizer is not None:
        _enum(optimizer.get("name"), "optimizer", "$.optimizer.name", report)


def _validate_executable_payload(artifact: Mapping[str, Any], report: ValidationReport) -> None:
    payload = _require_object(artifact, "agent_payload", "$", report)
    if payload is None:
        return
    execution_mode = payload.get("execution_mode", "runtime_registry")
    if execution_mode not in ("cached_spatiotemporal_optimizer", "runtime_registry"):
        report.add_error(
            "$.agent_payload.execution_mode",
            f"Unsupported execution mode: {execution_mode}",
            {"cached_spatiotemporal_optimizer", "runtime_registry"},
        )
    for key in ("optimization_problem", "evaluator", "solver"):
        _require_object(payload, key, "$.agent_payload", report)

    evaluator = payload.get("evaluator")
    if isinstance(evaluator, Mapping) and evaluator.get("name") in (
        "spatiotemporal_seims",
        "cached_spatiotemporal_seims",
    ):
        metric = evaluator.get("metric") or evaluator.get("objective_metric")
        if isinstance(metric, Mapping):
            _validate_metric(metric, "$.agent_payload.evaluator.metric", report)
        else:
            report.add_warning(
                "$.agent_payload.evaluator.metric",
                "spatiotemporal_seims payload should include an explicit finite metric spec.",
            )
        if evaluator.get("name") == "cached_spatiotemporal_seims":
            for field_name in ("baseline_raster", "spatial_graph_path", "spatial_scope", "temporal_scope"):
                _require_value(evaluator, field_name, "$.agent_payload.evaluator", report)

    solver = payload.get("solver")
    if isinstance(solver, Mapping):
        if "name" in solver:
            _enum(solver.get("name"), "optimizer", "$.agent_payload.solver.name", report)
        elif "Type" in solver:
            report.add_warning(
                "$.agent_payload.solver.name",
                "Legacy solver payload uses Type instead of framework optimizer.name.",
            )
        else:
            report.add_warning(
                "$.agent_payload.solver.name",
                "Solver object should include optimizer.name for strict framework validation.",
            )


def _validate_runtime_result(artifact: Mapping[str, Any], report: ValidationReport) -> None:
    _enum(artifact.get("status"), "status", "$.status", report)
    if artifact.get("status") == "completed" and "best_solution" not in artifact:
        report.add_warning("$.best_solution", "completed result should include best_solution.")


def _validate_model_term(term: Any, path: str, report: ValidationReport, objective: bool) -> None:
    if not isinstance(term, Mapping):
        report.add_error(path, "Model term must be an object.")
        return
    _require_value(term, "id", path, report)
    _require_value(term, "demand_unit_id", path, report)
    metric = _require_object(term, "metric", path, report)
    if metric is not None:
        _validate_metric(metric, f"{path}.metric", report)
    criterion = _require_object(term, "criterion", path, report)
    if criterion is not None:
        _enum(criterion.get("sense"), "sense", f"{path}.criterion.sense", report)
        if objective and criterion.get("sense") not in ("maximize", "minimize"):
            report.add_error(
                f"{path}.criterion.sense",
                "Objective criterion must use maximize or minimize.",
                {"maximize", "minimize"},
            )


def _validate_metric(metric: Mapping[str, Any], path: str, report: ValidationReport) -> None:
    _enum(metric.get("spatial_aggregation"), "spatial_aggregation", f"{path}.spatial_aggregation", report)
    _enum(metric.get("temporal_aggregation"), "temporal_aggregation", f"{path}.temporal_aggregation", report)
    _enum(metric.get("comparison"), "comparison", f"{path}.comparison", report)
    _enum(metric.get("peak_reference", "none"), "peak_reference", f"{path}.peak_reference", report)
    source = metric.get("source", "raster")
    _enum(source, "source", f"{path}.source", report)
    if source != "raster":
        if metric.get("peak_reference", "none") != "none":
            report.add_error(f"{path}.peak_reference", "Non-raster metrics cannot use peak references.")
        return
    if MetricSpec is not None:
        try:
            MetricSpec.from_config(dict(metric))
        except UnsupportedMetricSpec as exc:
            report.add_error(path, str(exc))


def _validate_node_ids(
    domain: Mapping[str, Any],
    path: str,
    report: ValidationReport,
    graph: WatershedSpatialGraph,
) -> None:
    node_ids = domain.get("node_ids", [])
    if node_ids is None:
        return
    if not isinstance(node_ids, list):
        report.add_error(f"{path}.node_ids", "node_ids must be a list.")
        return
    named_regions = set(getattr(graph, "named_regions", {}).keys())
    missing = [node_id for node_id in node_ids if node_id not in graph.nodes and node_id not in named_regions]
    if missing:
        report.add_error(f"{path}.node_ids", f"Unknown spatial node ids: {missing[:10]}")


def _load_graph(path: str | Path, report: ValidationReport) -> WatershedSpatialGraph | None:
    try:
        return WatershedSpatialGraph.load(path)
    except Exception as exc:  # pragma: no cover - defensive feedback path
        report.add_warning("$.spatial_graph", f"Unable to load spatial graph: {exc}")
        return None


def _require_list(obj: Mapping[str, Any], key: str, path: str, report: ValidationReport) -> List[Any]:
    value = obj.get(key)
    child_path = f"{path}.{key}" if path != "$" else f"$.{key}"
    if value is None:
        report.add_error(child_path, "Required list is missing.")
        return []
    if not isinstance(value, list):
        report.add_error(child_path, "Value must be a list.")
        return []
    return value


def _require_object(
    obj: Mapping[str, Any],
    key: str,
    path: str,
    report: ValidationReport,
) -> Mapping[str, Any] | None:
    value = obj.get(key)
    child_path = f"{path}.{key}" if path != "$" else f"$.{key}"
    if value is None:
        report.add_error(child_path, "Required object is missing.")
        return None
    if not isinstance(value, Mapping):
        report.add_error(child_path, "Value must be an object.")
        return None
    return value


def _require_value(obj: Mapping[str, Any], key: str, path: str, report: ValidationReport) -> None:
    value = obj.get(key)
    child_path = f"{path}.{key}" if path != "$" else f"$.{key}"
    if value in (None, ""):
        report.add_error(child_path, "Required value is missing.")


def _enum(value: Any, enum_name: str, path: str, report: ValidationReport) -> None:
    allowed = ALLOWED[enum_name]
    if value is None:
        report.add_error(path, "Required enum value is missing.", allowed)
        return
    if str(value) not in allowed:
        report.add_error(path, f"Unsupported value: {value}", allowed)


def _sorted_or_none(values: Optional[Iterable[str]]) -> Optional[List[str]]:
    if values is None:
        return None
    return sorted(str(value) for value in values)


def _format_issue(issue: ValidationIssue) -> str:
    line = f"- {issue.path}: {issue.message}"
    if issue.allowed:
        line += f" Allowed: {issue.allowed}"
    return line
