# Minimal __init__ - avoid eager SEIMS imports
# Import specific modules directly when needed:
#   from app.geo_opt.optimization_framework.factory import OptimizationFactory
#   from app.geo_opt.optimization_framework.registry import EvaluatorRegistry
from .core.interfaces import AbstractEvaluator, AbstractSolver, AbstractConstraint
