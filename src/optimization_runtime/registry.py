from __future__ import annotations

from typing import Any, Dict, Optional, Type

from .core.interfaces import AbstractEvaluator


def _load_seims_evaluator():
    from .impl.evaluators.seims_adapter import SEIMSEvaluator
    return SEIMSEvaluator


def _load_spatiotemporal_seims_evaluator():
    from .impl.evaluators.spatiotemporal_seims_adapter import SpatioTemporalSEIMSEvaluator
    return SpatioTemporalSEIMSEvaluator


def _load_cost_evaluator():
    from .impl.evaluators.cost_adapter import CostEvaluator
    return CostEvaluator


def _load_surrogate_evaluator():
    from .impl.evaluators.surrogate_adapter import SurrogateEvaluator
    return SurrogateEvaluator


def _load_custom_evaluator():
    from .impl.evaluators.func_adapter import CustomFunctionEvaluator
    return CustomFunctionEvaluator


def _load_lightweight_cost_evaluator():
    from .impl.evaluators.lightweight_cost import LightweightCostEvaluator
    return LightweightCostEvaluator


def _load_swatplus_evaluator():
    from .impl.evaluators.swatplus_adapter import SwatPlusEvaluator
    return SwatPlusEvaluator


def _load_swmm_lid_evaluator():
    from .impl.evaluators.swmm_lid_adapter import SwmmLidEvaluator
    return SwmmLidEvaluator


def _load_epa_swmm_evaluator():
    from .impl.evaluators.epa_swmm_adapter import EpaSwmmEvaluator
    return EpaSwmmEvaluator


def _load_placeholder_evaluator():
    from .impl.evaluators.placeholder_adapter import PlaceholderEvaluator
    return PlaceholderEvaluator


_BUILTIN_LOADERS: Dict[str, Any] = {
    "bmp_net_cost_model": _load_cost_evaluator,
    "bmp_net_cost": _load_cost_evaluator,
    "bmp_cost_model": _load_cost_evaluator,
    "bmp_cost": _load_cost_evaluator,
    "lightweight_cost": _load_lightweight_cost_evaluator,
    "seims": _load_seims_evaluator,
    "spatiotemporal_seims": _load_spatiotemporal_seims_evaluator,
    "spatial_seims": _load_spatiotemporal_seims_evaluator,
    "surrogate": _load_surrogate_evaluator,
    "sklearn_surrogate": _load_surrogate_evaluator,
    "surrogate_model": _load_surrogate_evaluator,
    "swat": _load_swatplus_evaluator,
    "swatplus": _load_swatplus_evaluator,
    "swat+": _load_swatplus_evaluator,
    "swmm": _load_swmm_lid_evaluator,
    "swmm_lid": _load_swmm_lid_evaluator,
    "epa_swmm": _load_swmm_lid_evaluator,
    "swmm_lid_demo": _load_swmm_lid_evaluator,
    "epa_swmm": _load_epa_swmm_evaluator,
    "custom_python": _load_custom_evaluator,
    "placeholder_metric": _load_placeholder_evaluator,
    "placeholder": _load_placeholder_evaluator,
    "custom_needed": _load_placeholder_evaluator,
}

_BUILTIN_STUBS: Dict[str, Type[AbstractEvaluator]] = {}

_USER_REGISTRY: Dict[str, Type[AbstractEvaluator]] = {}
_ENCODERS: Dict[str, Any] = {}
_SIMULATORS: Dict[str, Any] = {}
_OPTIMIZERS: Dict[str, Any] = {}


class EvaluatorRegistry:
    """Registry for builtin and user-defined objective evaluators."""

    @classmethod
    def register(cls, name: str, evaluator_cls: Type[AbstractEvaluator]):
        if not issubclass(evaluator_cls, AbstractEvaluator):
            raise TypeError(
                f"[EvaluatorRegistry] {evaluator_cls.__name__} must subclass AbstractEvaluator"
            )
        _USER_REGISTRY[name.lower()] = evaluator_cls
        print(f"[EvaluatorRegistry] registered user evaluator: '{name}' -> {evaluator_cls.__name__}")

    @classmethod
    def create(
        cls,
        evaluator_type: str,
        raw_cfg=None,
        is_temporal: bool = False,
        model_dir: Optional[str] = None,
        code: Optional[str] = None,
        context: Optional[dict] = None,
        **extra_kwargs,
    ) -> AbstractEvaluator:
        key = evaluator_type.lower().strip()

        if key in _USER_REGISTRY:
            try:
                return _USER_REGISTRY[key](
                    raw_cfg,
                    is_temporal=is_temporal,
                    context=context,
                    **extra_kwargs,
                )
            except TypeError:
                try:
                    return _USER_REGISTRY[key](raw_cfg, is_temporal=is_temporal, **extra_kwargs)
                except TypeError:
                    return _USER_REGISTRY[key]()

        if key in _BUILTIN_STUBS:
            return _BUILTIN_STUBS[key](raw_cfg, is_temporal=is_temporal, **extra_kwargs)

        loader = _BUILTIN_LOADERS.get(key)
        if loader is None:
            available = sorted(set(_BUILTIN_LOADERS.keys()) | set(_USER_REGISTRY.keys()))
            raise ValueError(
                f"[EvaluatorRegistry] unknown evaluator type: '{evaluator_type}'\n"
                f"  available: {available}\n"
                f"  for custom types, call EvaluatorRegistry.register(name, cls) first"
            )

        evaluator_cls = loader()

        if key in ("surrogate", "sklearn_surrogate", "surrogate_model"):
            return evaluator_cls(
                raw_cfg,
                is_temporal=is_temporal,
                model_dir=model_dir,
                context=context,
                **extra_kwargs,
            )
        if key == "custom_python":
            if not code:
                raise ValueError("[EvaluatorRegistry] custom_python evaluator requires 'code' parameter")
            return evaluator_cls(code)
        if key == "lightweight_cost":
            return evaluator_cls(context=context)
        if key in ("placeholder_metric", "placeholder", "custom_needed"):
            eval_config = extra_kwargs.pop("eval_config", None)
            return evaluator_cls(
                raw_cfg,
                is_temporal=is_temporal,
                context=context,
                eval_config=eval_config,
                **extra_kwargs,
            )
        if key in ("spatiotemporal_seims", "spatial_seims"):
            return evaluator_cls(raw_cfg, is_temporal=is_temporal, context=context, **extra_kwargs)

        return evaluator_cls(raw_cfg, is_temporal=is_temporal, **extra_kwargs)

    @classmethod
    def list_available(cls) -> Dict[str, str]:
        result = {}
        for key, loader in _BUILTIN_LOADERS.items():
            result[key] = "builtin" if loader is not None else "stub (not implemented)"
        for key in _BUILTIN_STUBS:
            result[key] = "stub (not implemented)"
        for key, evaluator_cls in _USER_REGISTRY.items():
            result[key] = f"user-defined ({evaluator_cls.__name__})"
        return result


class EncoderRegistry:
    @classmethod
    def register(cls, name: str, encoder_cls):
        _ENCODERS[name.lower()] = encoder_cls

    @classmethod
    def create(cls, name: str = "payload", **kwargs):
        key = name.lower()
        if key not in _ENCODERS:
            raise ValueError(f"Unknown encoder '{name}'. Available: {sorted(_ENCODERS)}")
        return _ENCODERS[key](**kwargs)


class SimulatorRegistry:
    @classmethod
    def register(cls, name: str, simulator_cls):
        _SIMULATORS[name.lower()] = simulator_cls

    @classmethod
    def create(cls, name: str = "legacy", **kwargs):
        key = name.lower()
        if key not in _SIMULATORS:
            raise ValueError(f"Unknown simulator '{name}'. Available: {sorted(_SIMULATORS)}")
        return _SIMULATORS[key](**kwargs)


class OptimizerRegistry:
    @classmethod
    def register(cls, name: str, optimizer_cls):
        _OPTIMIZERS[name.lower()] = optimizer_cls

    @classmethod
    def create(cls, name: str, **kwargs):
        key = name.lower()
        if key not in _OPTIMIZERS:
            raise ValueError(f"Unknown optimizer '{name}'. Available: {sorted(_OPTIMIZERS)}")
        return _OPTIMIZERS[key](**kwargs)


def _register_builtin_execution_components() -> None:
    from .impl.encoders.payload_encoder import PayloadEncoder
    from .impl.simulators.legacy_simulator import LegacySimulator
    from .impl.solvers.nsga2_optimizer import NSGA2Optimizer
    for name in ("payload", "generic", "seims_slppos", "swatplus_hru", "swmm_subcatchment"):
        EncoderRegistry.register(name, PayloadEncoder)
    for name in ("legacy", "seims", "swatplus", "epa_swmm", "swmm_lid"):
        SimulatorRegistry.register(name, LegacySimulator)
    for name in ("nsga2", "nsga-ii", "nsga2 (spatial)"):
        OptimizerRegistry.register(name, NSGA2Optimizer)


_register_builtin_execution_components()
