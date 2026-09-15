from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from src.modeling.validators import validate_artifact


@dataclass
class ContractReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("; ".join(self.errors))


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?\s*", "", str(text or "")).replace("```", "").strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def validate_bdu_artifact(artifact: Mapping[str, Any]) -> ContractReport:
    errors: list[str] = []
    warnings: list[str] = []
    if artifact.get("stage") != "bdu":
        errors.append("$.stage must be 'bdu'.")
    bdu_list = artifact.get("bdu_list")
    if not isinstance(bdu_list, list) or not bdu_list:
        errors.append("$.bdu_list must be a non-empty list.")
        return ContractReport(False, errors, warnings)
    for idx, bdu in enumerate(bdu_list):
        path = f"$.bdu_list[{idx}]"
        if not isinstance(bdu, Mapping):
            errors.append(f"{path} must be an object.")
            continue
        for key in ("id", "Sub", "Tar", "BMP_Pref", "Scope"):
            if key not in bdu:
                errors.append(f"{path}.{key} is required.")
        if isinstance(bdu.get("Tar"), Mapping):
            for key in ("variable", "metric", "operator"):
                if key not in bdu["Tar"]:
                    errors.append(f"{path}.Tar.{key} is required.")
        if isinstance(bdu.get("Scope"), Mapping):
            for key in ("spatial", "temporal"):
                if key not in bdu["Scope"]:
                    errors.append(f"{path}.Scope.{key} is required.")
    checklist = artifact.get("governance_checklist", [])
    if checklist is not None and not isinstance(checklist, list):
        warnings.append("$.governance_checklist should be a list.")
    return ContractReport(not errors, errors, warnings)


def bdu_to_demand_unit_artifact(bdu_artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Project BDU to the existing broad demand_unit schema for validation/logging."""
    demand_units = []
    for bdu in bdu_artifact.get("bdu_list", []):
        if not isinstance(bdu, Mapping):
            continue
        target = bdu.get("Tar", {}) if isinstance(bdu.get("Tar"), Mapping) else {}
        scope = bdu.get("Scope", {}) if isinstance(bdu.get("Scope"), Mapping) else {}
        variable = str(target.get("variable", "custom")).lower()
        operator = str(target.get("operator", "maximize")).lower()
        is_cost = any(token in variable for token in ("cost", "investment", "budget"))
        is_runoff = "runoff" in variable or "flow" in variable
        is_nutrient = any(token in variable for token in ("nitrogen", "phosphorus", "nutrient", "nitrate"))
        phenomenon = "cost" if is_cost else "runoff" if is_runoff else "nutrient" if is_nutrient else "sediment"
        source = "cost_model" if is_cost else "timeseries" if is_runoff else "custom" if is_nutrient else "raster"
        demand_units.append(
            {
                "id": str(bdu.get("id")),
                "role": "constraint" if operator in {"<", "<=", ">", ">=", "==", "lower", "bigger", "around"} else "objective",
                "user_expression": str(bdu.get("source_text", "")),
                "quantity": {
                    "phenomenon": phenomenon,
                    "variable": "investment_cost" if is_cost else "custom" if phenomenon in {"runoff", "nutrient"} else "sediment_load",
                    "source": source,
                    "simulator_output": str(target.get("variable", "")),
                },
                "metric": {
                    "target": str(target.get("variable", "")),
                    "spatial_aggregation": "sum",
                    "temporal_aggregation": "window_total",
                    "comparison": "raw_value" if is_cost else "reduction_rate",
                    "peak_reference": "none",
                    "source": source,
                    "value": str(target.get("metric", target.get("variable", ""))),
                },
                "criterion": {
                    "sense": _operator_to_sense(operator),
                    "threshold": target.get("threshold"),
                    "unit": str(target.get("unit", "native")),
                },
                "scope": {
                    "raw_spatial_text": str(scope.get("spatial", "")),
                    "raw_temporal_text": str(scope.get("temporal", "")),
                    "evaluation_domain_hint": str(scope.get("spatial", "")),
                    "action_domain_hint": "auto",
                    "simulation_domain_hint": "full_watershed",
                },
                "uncertainties": bdu.get("uncertainties", []),
            }
        )
    return {"stage": "demand_unit", "source": "bdu_projection", "demand_units": demand_units}


def validate_payload_artifact(stage2_output: Mapping[str, Any]) -> dict[str, Any]:
    payload = stage2_output.get("agent_payload", {})
    report = validate_artifact("executable_payload", {"stage": "executable_payload", "agent_payload": payload})
    return {
        "stage": report.stage,
        "ok": report.ok,
        "errors": [issue.__dict__ for issue in report.errors],
        "warnings": [issue.__dict__ for issue in report.warnings],
        "feedback": report.feedback(),
    }


def _operator_to_sense(operator: str) -> str:
    if operator in {"min", "minimize"}:
        return "minimize"
    if operator in {"<", "<=", "lower", "not exceed", "under"}:
        return "<="
    if operator in {">", ">=", "bigger", "at least"}:
        return ">="
    if operator in {"==", "equal"}:
        return "=="
    return "maximize"
