"""
UnifiedSolver: 轻量NSGA-II主循环，算子通过SolverOperators注入。

设计原则：
- 核心进化循环不依赖任何SEIMS/MongoDB/pygeoc
- 算子（初始化/交叉/变异）通过SolverOperators注入，可替换为SEIMS空间感知版本
- context统一为dict，评估器签名 evaluate(individual, context) -> float 不变
- SEIMS插件只替换算子，不改变主循环逻辑
"""
from __future__ import annotations

import array
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Any

from deap import base, creator, tools, algorithms

from ...core.interfaces import AbstractSolver


# ─── 默认轻量算子（无外部依赖）────────────────────────────────────────────────

def _default_initializer(context: dict) -> List[int]:
    """随机整数初始化，从context读取genes_num和bmp_types"""
    n = context.get("genes_num", 10)
    bmp_types = context.get("bmp_types", [1, 2])
    bmp_times = context.get("bmp_time_values", [])
    zero_probability = float(context.get("zero_probability", 0.0))
    if bmp_times:
        values = [int(bmp) * 1000 + int(period) for bmp in bmp_types for period in bmp_times if int(bmp) > 0]
        return [0 if random.random() < zero_probability else random.choice(values) for _ in range(n)]
    bmp_min = min(bmp_types) if bmp_types else 1
    bmp_max = max(bmp_types) if bmp_types else 2
    return [random.randint(bmp_min, bmp_max) for _ in range(n)]


def _default_crossover(ind1, ind2, context: dict):
    """两点交叉（DEAP内置）"""
    if min(len(ind1), len(ind2)) >= 2:
        tools.cxTwoPoint(ind1, ind2)
    return ind1, ind2


def _default_mutation(ind, context: dict):
    """均匀整数变异"""
    bmp_types = context.get("bmp_types", [1, 2])
    bmp_times = context.get("bmp_time_values", [])
    bmp_min = min(bmp_types) if bmp_types else 1
    bmp_max = max(bmp_types) if bmp_types else 2
    indpb = context.get("mut_indpb", 0.1)
    if bmp_times:
        values = [int(bmp) * 1000 + int(period) for bmp in bmp_types for period in bmp_times if int(bmp) > 0]
        zero_probability = float(context.get("zero_probability", 0.0))
        for idx in range(len(ind)):
            if random.random() < indpb:
                ind[idx] = 0 if random.random() < zero_probability else random.choice(values)
        return ind
    tools.mutUniformInt(ind, low=bmp_min, up=bmp_max, indpb=indpb)
    return ind


# ─── SolverOperators 数据类 ───────────────────────────────────────────────────

@dataclass
class SolverOperators:
    """
    求解器算子配置。

    轻量默认：随机初始化 + 两点交叉 + 均匀整数变异
    SEIMS替换：initialize_scenario + crossover_slppos + mutate_rule_s

    所有算子签名：
      initializer(context: dict) -> List[int]
      crossover(ind1, ind2, context: dict) -> (ind1, ind2)
      mutation(ind, context: dict) -> ind
    """
    initializer: Callable = field(default=_default_initializer)
    crossover:   Callable = field(default=_default_crossover)
    mutation:    Callable = field(default=_default_mutation)

    # 标记来源，供日志/E4识别
    source: str = "lightweight"  # "lightweight" | "seims"


# ─── UnifiedSolver ────────────────────────────────────────────────────────────

class UnifiedSolver(AbstractSolver):
    """
    统一NSGA-II求解器。

    轻量版和SEIMS版共用同一个进化循环，只有SolverOperators不同。
    支持 on_generation_callback 钩子，供E4实时监控每代结果。
    """

    def __init__(self, operators: SolverOperators, cfg: dict):
        """
        Parameters
        ----------
        operators: SolverOperators，算子配置（轻量默认或SEIMS替换）
        cfg: solver配置字典，包含 n_pop, n_gen, cx_prob, mut_prob
        """
        self.ops = operators
        self.cfg = cfg
        self._toolbox: Optional[base.Toolbox] = None

    def _setup_deap(self, problem) -> base.Toolbox:
        """配置DEAP工具箱"""
        # 清理全局定义（避免重复导入冲突）
        for attr in ("FitnessMulti", "Individual"):
            if hasattr(creator, attr):
                delattr(creator, attr)

        creator.create("FitnessMulti", base.Fitness, weights=problem.weights)
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()

        # 初始化：通过算子注入
        ops = self.ops
        ctx = problem.ctx

        toolbox.register("gene_values", ops.initializer, ctx)
        toolbox.register("individual", tools.initIterate,
                         creator.Individual, toolbox.gene_values)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # 评估：委托给Problem
        toolbox.register("evaluate", problem.evaluate)

        # 交叉/变异：通过算子注入（包装为DEAP期望的签名）
        def _cx(ind1, ind2):
            return ops.crossover(ind1, ind2, ctx)

        def _mut(ind):
            result = ops.mutation(ind, ctx)
            return (result,)

        toolbox.register("mate", _cx)
        toolbox.register("mutate", _mut)
        toolbox.register("select", tools.selNSGA2)

        return toolbox

    def solve(self, problem, on_generation_callback=None) -> List[Any]:
        """
        执行NSGA-II优化主循环。

        Parameters
        ----------
        problem: OptimizationProblem
        on_generation_callback: 可选回调 func(pop, gen)，E4监控钩子

        Returns
        -------
        最终种群列表
        """
        n_pop   = self.cfg.get("n_pop", 80)
        n_gen   = self.cfg.get("n_gen", 100)
        cx_prob = self.cfg.get("cx_prob", 0.8)
        mut_prob = self.cfg.get("mut_prob", 0.1)

        toolbox = self._setup_deap(problem)

        # 初始化种群
        pop = toolbox.population(n=n_pop)

        # 初始评估
        invalid = [ind for ind in pop if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        if on_generation_callback:
            on_generation_callback(pop, 0)

        print(f"[UnifiedSolver/{self.ops.source}] Pop={n_pop}, Gen={n_gen}")

        # 进化主循环
        for gen in range(1, n_gen + 1):
            offspring = algorithms.varOr(pop, toolbox,
                                         lambda_=n_pop,
                                         cxpb=cx_prob,
                                         mutpb=mut_prob)
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
                ind.fitness.values = fit

            pop = toolbox.select(pop + offspring, n_pop)

            if on_generation_callback:
                on_generation_callback(pop, gen)

        return pop
