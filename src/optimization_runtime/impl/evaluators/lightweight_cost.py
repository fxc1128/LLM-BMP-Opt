"""
LightweightCostEvaluator: 不依赖SEIMS/MongoDB的简化成本评估器。

从 context["bmp_cost_table"] 读取单位成本，计算总成本（万元）。
用于轻量路径，或作为 CostEvaluator 不可用时的降级方案。

接口与 CostEvaluator 完全一致：evaluate(individual, context) -> float
"""
from __future__ import annotations
from ...core.interfaces import AbstractEvaluator


class LightweightCostEvaluator(AbstractEvaluator):
    """
    轻量成本评估器。

    成本 = sum(bmp_cost_table[bmp_code] * unit_area) / 10000  (万元)

    若 context 中无 unit_areas，则每个单元面积默认为 1.0（相对成本）。
    """

    def __init__(self, context: dict = None, **kwargs):
        # bmp_cost_table 可在构造时传入，也可在 evaluate 时从 context 读取
        self._cost_table = (context or {}).get("bmp_cost_table", {})

    def evaluate(self, individual, context) -> float:
        cost_table = context.get("bmp_cost_table", self._cost_table)
        unit_areas = context.get("unit_areas", {})
        gene_to_unit = context.get("gene_to_unit", {})

        total = 0.0
        for i, bmp_code in enumerate(individual):
            bmp_code = int(bmp_code)
            if bmp_code >= 1000:
                bmp_code = bmp_code // 1000
            unit_id = gene_to_unit.get(i, i)
            area = unit_areas.get(unit_id, 1.0)
            unit_cost = cost_table.get(bmp_code, 0)
            total += unit_cost * area

        return total / 10000.0  # 转换为万元
