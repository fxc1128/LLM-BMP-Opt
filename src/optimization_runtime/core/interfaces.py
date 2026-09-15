from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Tuple

class ConstraintMode(Enum):
    PRE_EVALUATION = 1  # 基因生成后，跑任何评估前 (如：检查拓扑)
    POST_EVALUATION = 2 # 跑完部分评估后 (如：算出成本后检查预算)

class AbstractConstraint(ABC):
    @abstractmethod
    def validate(self, individual: Any, context: Any, mode: ConstraintMode) -> bool:
        """返回 True 表示满足约束，False 表示违背"""
        pass

class AbstractEvaluator(ABC):
    @abstractmethod
    def evaluate(self, individual: Any, context: Any) -> float:
        """返回单项目标值"""
        pass


@dataclass
class DecisionSpace:
    """Stable mapping between optimizer genes and real spatial management units."""

    unit_ids: list[str]
    gene_domains: list[list[int]]
    gene_to_unit: dict[int, str]
    unit_to_gene: dict[str, int]
    bmp_catalog: dict[int, str] = field(default_factory=dict)
    unit_metadata: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    spatial_relations: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Raw model-run result shared by objective and post-simulation constraints."""

    outputs: dict[str, Any] = field(default_factory=dict)
    work_dir: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AbstractEncoder(ABC):
    @abstractmethod
    def build(self, agent_payload: Mapping[str, Any]) -> DecisionSpace:
        """Create ordered spatial units, admissible gene values, and model bindings."""
        pass


class AbstractSimulator(ABC):
    @abstractmethod
    def run(self, individual: Any, decision_space: DecisionSpace, context: dict[str, Any]) -> SimulationResult:
        """Materialize one scenario and return raw simulator outputs."""
        pass


class AbstractObjectiveEvaluator(AbstractEvaluator):
    @abstractmethod
    def evaluate_result(self, result: SimulationResult, individual: Any, decision_space: DecisionSpace) -> float:
        """Compute an objective from a shared simulation result."""
        pass

    def evaluate(self, individual: Any, context: Any) -> float:
        return self.evaluate_result(context["_simulation_result"], individual, context["decision_space"])


class AbstractConstraintEvaluator(AbstractConstraint):
    @abstractmethod
    def validate_result(self, individual: Any, decision_space: DecisionSpace, result: SimulationResult | None, mode: ConstraintMode) -> bool:
        """Validate spatial constraints before simulation or metric constraints after it."""
        pass

    def validate(self, individual: Any, context: Any, mode: ConstraintMode) -> bool:
        result = context.get("_simulation_result") if isinstance(context, dict) else None
        return self.validate_result(individual, context["decision_space"], result, mode)

class AbstractSolver(ABC):
    @abstractmethod
    def solve(self, problem: Any) -> List[Any]:
        """运行优化算法"""
        pass
