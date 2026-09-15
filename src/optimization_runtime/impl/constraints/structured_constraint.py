from __future__ import annotations

import re
from typing import Any, Iterable

from ...core.interfaces import AbstractConstraint, ConstraintMode


def normalize_bmp_types(value: Any, default: Iterable[int] = (1, 2, 3, 4)) -> list[int]:
    """Normalize E3 BMP type domains into an explicit integer list."""
    if value is None:
        return list(default)
    if isinstance(value, dict):
        if "values" in value:
            value = value["values"]
        elif "range" in value:
            value = value["range"]
        elif "min" in value or "max" in value:
            low = int(value.get("min", value.get("lower", 0)))
            high = int(value.get("max", value.get("upper", low)))
            step = int(value.get("step", 1) or 1)
            return list(range(low, high + 1, step))
        elif "lower" in value or "upper" in value:
            low = int(value.get("lower", 0))
            high = int(value.get("upper", low))
            step = int(value.get("step", 1) or 1)
            return list(range(low, high + 1, step))
    if isinstance(value, str):
        nums = [int(float(item)) for item in re.findall(r"[-+]?\d+(?:\.\d+)?", value)]
        if len(nums) >= 2 and ("-" in value or "to" in value.lower()):
            return list(range(nums[0], nums[1] + 1))
        return nums or list(default)
    if isinstance(value, Iterable):
        return sorted({int(item) for item in value})
    return list(default)


class StructuredConstraint(AbstractConstraint):
    """Executable form for the four in-KB constraint templates.

    Supported template families:
    - Resource_Limitation: post-evaluation objective/value upper bound.
    - Demand_Satisfaction: post-evaluation objective/value lower bound.
    - Spatial_Relation: pre-evaluation per-gene spatial/type rule.
    - Physical_feasibility: pre-evaluation per-gene feasibility rule.
    """

    def __init__(self, spec: dict, default_target_index: int = 0):
        self.spec = spec or {}
        self.ctype = str(self.spec.get("type", "")).lower()
        self.target_idx = int(
            self.spec.get("target_obj_index", self.spec.get("objective_index", default_target_index))
        )
        self.threshold = self._threshold()

    def validate(self, individual, context, mode: ConstraintMode) -> bool:
        if mode == ConstraintMode.POST_EVALUATION and self._is_post_bound():
            return self._validate_post_bound(individual)
        if mode == ConstraintMode.PRE_EVALUATION and self._is_pre_gene_rule():
            return self._validate_pre_gene_rule(individual, context or {})
        return True

    def _is_post_bound(self) -> bool:
        return self.ctype in {
            "resource_limitation",
            "budget_limitation",
            "demand_satisfaction",
            "objective_lower_bound",
            "objective_upper_bound",
        }

    def _is_pre_gene_rule(self) -> bool:
        return self.ctype in {
            "spatial_relation",
            "physical_feasibility",
            "bmp_type_limitation",
            "type_limitation",
            "value_range",
            "custom_constraint",
        }

    def _validate_post_bound(self, individual) -> bool:
        if self.threshold is None:
            return True
        value = getattr(individual, f"obj_{self.target_idx}", None)
        if value is None:
            if hasattr(individual, "fitness") and getattr(individual.fitness, "valid", False):
                values = getattr(individual.fitness, "values", ())
                if self.target_idx < len(values):
                    value = values[self.target_idx]
        if value is None:
            return True
        sense = str(self.spec.get("sense", "")).strip()
        if not sense:
            sense = ">=" if self.ctype in {"demand_satisfaction", "objective_lower_bound"} else "<="
        if sense in (">=", "min", "at_least"):
            return float(value) >= float(self.threshold)
        if sense in (">",):
            return float(value) > float(self.threshold)
        if sense in ("<",):
            return float(value) < float(self.threshold)
        return float(value) <= float(self.threshold)

    def _validate_pre_gene_rule(self, individual, context: dict) -> bool:
        indices = self._target_gene_indices(context, len(individual))
        if not indices:
            return True

        allowed = self.spec.get("allowed_bmp_types", self.spec.get("allowed_types"))
        forbidden = self.spec.get("forbidden_bmp_types", self.spec.get("forbidden_types"))
        required = self.spec.get("required_bmp_type", self.spec.get("fixed_value"))
        minimum = self.spec.get("min_value", self.spec.get("lower"))
        maximum = self.spec.get("max_value", self.spec.get("upper"))

        allowed_set = set(normalize_bmp_types(allowed, default=[])) if allowed is not None else None
        forbidden_set = set(normalize_bmp_types(forbidden, default=[])) if forbidden is not None else set()

        for idx in indices:
            if idx < 0 or idx >= len(individual):
                continue
            value = int(individual[idx])
            if required is not None and value != int(required):
                return False
            if allowed_set is not None and value not in allowed_set:
                return False
            if value in forbidden_set:
                return False
            if minimum is not None and value < int(minimum):
                return False
            if maximum is not None and value > int(maximum):
                return False
        return True

    def _target_gene_indices(self, context: dict, length: int) -> list[int]:
        explicit = self.spec.get("gene_indices")
        if explicit is not None:
            return [int(idx) for idx in explicit]

        if str(self.spec.get("scope", "")).lower() in {"action_domain", "evaluation_scope"}:
            return [int(idx) for idx in context.get("action_domain_allowed_gene_indices", [])]

        unit_ids = self.spec.get("unit_ids") or self.spec.get("spatial_unit_ids")
        if unit_ids is not None:
            unit_set = {int(item) for item in unit_ids}
            gene_to_unit = {int(k): int(v) for k, v in context.get("gene_to_unit", {}).items()}
            return sorted(idx for idx, unit_id in gene_to_unit.items() if unit_id in unit_set)

        return list(range(length))

    def _threshold(self) -> float | None:
        for key in ("threshold", "value", "budget", "limit", "rhs", "target"):
            if key in self.spec and self.spec[key] is not None:
                return float(self.spec[key])
        criterion = self.spec.get("criterion")
        if isinstance(criterion, dict):
            for key in ("threshold", "value", "rhs", "target"):
                if key in criterion and criterion[key] is not None:
                    return float(criterion[key])
        match = re.search(
            r"(?:<=|>=|<|>)\s*([-+]?\d+(?:\.\d+)?)",
            str(self.spec.get("mathematical_form", "")),
        )
        if match:
            return float(match.group(1))
        return None
