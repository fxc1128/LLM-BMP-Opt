"""
OptimizationProblem: 优化问题定义。

context 解耦：同时支持 dict（轻量版）和 SASlpPosConfig（SEIMS版）。
评估器签名统一：evaluate(individual, context) -> float
  - 轻量版：context 是普通 dict
  - SEIMS版：context 是 dict，但含 _raw_cfg 键供评估器取用
"""
from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional, Union

from .interfaces import AbstractConstraint, AbstractEvaluator, ConstraintMode


class ObjectiveDir(Enum):
    MINIMIZE = -1.0
    MAXIMIZE = 1.0


@dataclass
class VariableDefinition:
    name: str
    type: str
    lower: float
    upper: float


class OptimizationProblem:
    """
    优化问题定义。

    context 可以是：
      - dict：轻量版，包含 genes_num / bmp_types / units_infos 等
      - SASlpPosConfig：SEIMS版（向后兼容），内部自动包装为 dict 访问
    """

    def __init__(self,
                 name: str,
                 context: Any,
                 evaluators: List[AbstractEvaluator],
                 constraints: List[AbstractConstraint],
                 weights: Tuple[float, ...],
                 simulator=None,
                 decision_space=None,
                 variables: Optional[List[VariableDefinition]] = None):
        self.name = name
        self.ctx = context
        self.evaluators = evaluators
        self.constraints = constraints
        self.weights = weights
        self.simulator = simulator
        self.decision_space = decision_space
        self.variables = variables or []

    @property
    def n_vars(self) -> int:
        """变量数量：优先从 context 获取"""
        if isinstance(self.ctx, dict):
            return self.ctx.get("genes_num", len(self.variables))
        # 向后兼容 SASlpPosConfig
        if hasattr(self.ctx, "genes_num"):
            return self.ctx.genes_num
        return len(self.variables)

    @property
    def n_objs(self) -> int:
        return len(self.evaluators)

    def evaluate(self, individual) -> Tuple[float, ...]:
        """
        核心评估流：Pre-Check → Evaluators（顺序执行）→ Post-Check

        约束短路：算完成本后若超预算，直接返回惩罚值，跳过后续慢评估器。
        """
        # Pre-evaluation 约束（拓扑检查等）
        for cons in self.constraints:
            if not cons.validate(individual, self.ctx, ConstraintMode.PRE_EVALUATION):
                return self._penalty()

        if self.simulator is not None:
            self.ctx["_simulation_result"] = self.simulator.run(individual, self.decision_space, self.ctx)

        results = []
        try:
            for i, ev in enumerate(self.evaluators):
                val = ev.evaluate(individual, self.ctx)
                try:
                    setattr(individual, f"obj_{i}", val)
                except Exception:
                    pass
                results.append(val)

                # Post-evaluation 约束（预算检查等）
                for cons in self.constraints:
                    if not cons.validate(individual, self.ctx, ConstraintMode.POST_EVALUATION):
                        return self._penalty()

        except Exception as e:
            print(f"[OptimizationProblem] evaluate error: {e}")
            return self._penalty()

        return tuple(results)

    def _penalty(self) -> Tuple[float, ...]:
        """根据权重方向返回极差惩罚值"""
        return tuple(1e15 if w < 0 else -1e15 for w in self.weights)
