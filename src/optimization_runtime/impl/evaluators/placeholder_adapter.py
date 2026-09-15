from __future__ import annotations

import random
from typing import Any, Iterable

from ...core.interfaces import AbstractEvaluator


class PlaceholderEvaluator(AbstractEvaluator):
    """Range-bounded fallback evaluator for runnable gap cases.

    The value is deliberately local: it uses a reproducible pseudo-random draw
    within a configured plausible range, seeded by the objective and BMP
    decision vector, so generated E3 code can run without an external simulator.
    Payload metadata must still mark this as a placeholder, not as a scientific
    substitute for the missing model interface.
    """

    def __init__(
        self,
        raw_cfg: Any = None,
        is_temporal: bool = False,
        context: dict | None = None,
        eval_config: dict | None = None,
        **_: Any,
    ) -> None:
        cfg = eval_config or {}
        self.objective_name = str(cfg.get("objective_name") or cfg.get("name") or "placeholder")
        self.default_value = float(cfg.get("default_value", 0.0))
        self.scale = float(cfg.get("scale", 1.0))
        requested_mode = str(cfg.get("mode", "semantic_auto")).lower()
        self.mode = _semantic_mode(self.objective_name, requested_mode)
        self.seed_offset = _stable_offset(self.objective_name)
        self.random_min, self.random_max = _configured_range(cfg, self.objective_name, self.mode)

    def evaluate(self, individual: Iterable[Any], context: dict) -> float:
        genes = [int(x) for x in individual]
        if self.mode == "constant":
            return self.default_value
        if self.random_min == self.random_max:
            return self.random_min
        seed = _stable_seed(self.objective_name, genes)
        return random.Random(seed).uniform(self.random_min, self.random_max)


def _semantic_mode(objective_name: str, requested_mode: str) -> str:
    if requested_mode not in ("", "auto", "semantic_auto", "decision_density"):
        return requested_mode

    text = objective_name.lower()
    if any(token in text for token in ("cost", "expense", "budget", "investment", "economy")):
        return "cost_like"
    if any(
        token in text
        for token in (
            "reduction",
            "benefit",
            "service",
            "ecosystem",
            "nitrogen",
            "phosphorus",
            "pollut",
            "quality",
            "runoff",
            "sediment",
            "yield",
        )
    ):
        return "benefit_like"
    return "benefit_like"


def _stable_offset(text: str) -> float:
    return (sum(ord(ch) for ch in text) % 97) / 1000.0


def _configured_range(cfg: dict[str, Any], objective_name: str, mode: str) -> tuple[float, float]:
    raw_range = cfg.get("value_range") or cfg.get("random_range") or cfg.get("plausible_range")
    if isinstance(raw_range, (list, tuple)) and len(raw_range) >= 2:
        return _ordered(float(raw_range[0]), float(raw_range[1]))
    if isinstance(raw_range, dict):
        lower = raw_range.get("min", raw_range.get("lower", raw_range.get("low")))
        upper = raw_range.get("max", raw_range.get("upper", raw_range.get("high")))
        if lower is not None and upper is not None:
            return _ordered(float(lower), float(upper))
    if "min_value" in cfg and "max_value" in cfg:
        return _ordered(float(cfg["min_value"]), float(cfg["max_value"]))

    text = objective_name.lower()
    if mode == "cost_like":
        return 0.8, 1.3
    if "rate" in text or "reduction" in text:
        return 0.0, 1.0
    return 30.0, 90.0


def _ordered(left: float, right: float) -> tuple[float, float]:
    return (left, right) if left <= right else (right, left)


def _stable_seed(objective_name: str, genes: list[int]) -> int:
    value = sum((idx + 17) * (gene + 31) for idx, gene in enumerate(genes))
    value += sum((idx + 1) * ord(ch) for idx, ch in enumerate(objective_name))
    return value
