from __future__ import annotations

import json
import os
from typing import Any

try:
    from evaluators import *  # noqa: F401,F403
except ModuleNotFoundError:
    pass

from agent_payload import AGENT_PAYLOAD
from runtime.factory import OptimizationFactory


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)


def _fitness_values(individual: Any) -> list[float]:
    fitness = getattr(individual, "fitness", None)
    values = getattr(fitness, "values", None)
    if values is None:
        return []
    return [float(v) for v in values]


def _serialize_solution(individual: Any, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "genes": _jsonable(list(individual)),
        "fitness": _fitness_values(individual),
    }


def main() -> int:
    problem, solver = OptimizationFactory.create(AGENT_PAYLOAD)
    population = solver.solve(problem)
    rows = [_serialize_solution(ind, idx) for idx, ind in enumerate(population)]
    rows.sort(key=lambda item: item.get("fitness", []))

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "pareto_front.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Saved optimization results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
