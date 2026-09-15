from __future__ import annotations

from src.optimization_runtime.core.interfaces import AbstractEvaluator
from src.optimization_runtime.registry import EvaluatorRegistry


class CustomEvaluator(AbstractEvaluator):
    """Template evaluator for an unsupported external model/profile."""

    def __init__(self, raw_cfg=None, is_temporal: bool = False, **kwargs):
        self.raw_cfg = raw_cfg
        self.context = kwargs.get("context", {})
        self.eval_config = kwargs.get("eval_config", {})

    def evaluate(self, individual, context) -> float:
        raise NotImplementedError("Implement model execution and metric parsing here.")


EvaluatorRegistry.register("custom_external_evaluator", CustomEvaluator)
