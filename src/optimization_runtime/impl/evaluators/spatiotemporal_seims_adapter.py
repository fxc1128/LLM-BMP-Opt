from __future__ import annotations

import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ...core.interfaces import AbstractEvaluator

seims_path = os.environ.get("SEIMS_PATH", "")
if seims_path and seims_path not in sys.path:
    sys.path.insert(0, seims_path)

try:
    from scenario_analysis.spatialunits.scenario import SUScenario
except ImportError:  # pragma: no cover - exercised only without SEIMS-dev
    SUScenario = None

from src.knowledge.spatial_graph_builder import SEIMSSpatialGraphBuilder
from src.knowledge.spatiotemporal_metric import SedimentReductionMetric
from src.knowledge.watershed_spatial_graph import WatershedSpatialGraph


class SpatioTemporalSEIMSEvaluator(AbstractEvaluator):
    """First-version spatial-window SEIMS evaluator.

    SEIMS still runs at the model's valid simulation scope, usually the full
    watershed. Local objectives are computed afterward from output rasters using
    a resolved spatial mask.
    """

    def __init__(
        self,
        raw_cfg,
        is_temporal: bool = False,
        context: Optional[dict] = None,
        eval_config: Optional[dict] = None,
        **_: Any,
    ):
        if SUScenario is None:
            raise ImportError("SEIMS scenario_analysis is required for SpatioTemporalSEIMSEvaluator.")

        self.is_temporal = is_temporal
        self.context = context or {}
        self.eval_config = eval_config or {}
        self.metric_cfg = self.eval_config.get("metric", {})
        self.temporal_scope = self._temporal_scope(raw_cfg)
        if self._use_period_sed_metric():
            self.raw_cfg = deepcopy(raw_cfg)
        else:
            self.raw_cfg = self._scoped_raw_cfg(raw_cfg, self.temporal_scope)
        self.env_file = self.metric_cfg.get(
            "env_file",
            self.eval_config.get("env_file", self.raw_cfg.eval_info.get("ENVEVAL", "SED_OL_SUM.tif")),
        )

        graph_path = self._spatial_graph_path()
        spatial_scope = self.eval_config.get(
            "spatial_scope",
            self.context.get("_agent_payload", {}).get("basic_info", {}).get("spatial_scope", "full_watershed"),
        )
        self.spatial_graph = WatershedSpatialGraph.load(graph_path)
        self.resolved_scope = self.spatial_graph.resolve(spatial_scope)
        self.metric = SedimentReductionMetric()

        self.baseline_raster = self.eval_config.get("baseline_raster")
        self.base_value = None
        self.base_period_values = None
        self._check_and_run_base_scenario()

    def _temporal_scope(self, raw_cfg) -> Dict[str, Any]:
        scope = dict(self.eval_config.get("temporal_scope", {}))
        scope.setdefault("type", "continuous")
        if scope["type"] != "continuous":
            raise ValueError("SpatioTemporalSEIMSEvaluator v1 only supports continuous temporal windows.")
        if "start" not in scope and getattr(raw_cfg, "eval_stime", None):
            scope["start"] = raw_cfg.eval_stime.strftime("%Y-%m-%d %H:%M:%S")
        if "end" not in scope and getattr(raw_cfg, "eval_etime", None):
            scope["end"] = raw_cfg.eval_etime.strftime("%Y-%m-%d %H:%M:%S")
        scope.setdefault("annualize", True)
        return scope

    def _scoped_raw_cfg(self, raw_cfg, temporal_scope: Dict[str, Any]):
        start = temporal_scope.get("start")
        end = temporal_scope.get("end")
        if not start or not end:
            return raw_cfg

        scoped = deepcopy(raw_cfg)
        start_dt = _parse_datetime(start, end_of_day=False)
        end_dt = _parse_datetime(end, end_of_day=True)
        scoped.eval_stime = start_dt
        scoped.eval_etime = end_dt
        scoped.runtime_years = max(1.0, (end_dt.year - start_dt.year + 1.0))

        model = getattr(scoped, "model", None)
        if model is not None:
            for attr, value in (
                ("simu_stime", start_dt),
                ("simu_etime", end_dt),
                ("out_stime", start_dt),
                ("out_etime", end_dt),
            ):
                if hasattr(model, attr):
                    setattr(model, attr, value)
            config_dict = getattr(model, "config_dict", None)
            if isinstance(config_dict, dict):
                config_dict["simu_stime"] = start_dt
                config_dict["simu_etime"] = end_dt
                config_dict["out_stime"] = start_dt
                config_dict["out_etime"] = end_dt
                seims_model = config_dict.get("SEIMS_Model")
                if isinstance(seims_model, dict):
                    seims_model["Sim_Time_start"] = start_dt.strftime("%Y-%m-%d %H:%M:%S")
                    seims_model["Sim_Time_end"] = end_dt.strftime("%Y-%m-%d %H:%M:%S")
                    seims_model["Output_Time_start"] = start_dt.strftime("%Y-%m-%d %H:%M:%S")
                    seims_model["Output_Time_end"] = end_dt.strftime("%Y-%m-%d %H:%M:%S")
                common = config_dict.get("Scenario_Common")
                if isinstance(common, dict):
                    common["eval_time_start"] = start_dt.strftime("%Y-%m-%d %H:%M:%S")
                    common["eval_time_end"] = end_dt.strftime("%Y-%m-%d %H:%M:%S")
                    common["runtime_years"] = int(scoped.runtime_years)
        return scoped

    def _spatial_graph_path(self) -> str:
        configured = self.eval_config.get("spatial_graph_path")
        auto_build = self.eval_config.get("spatial_graph_auto_build", False)
        if configured and Path(configured).is_file() and not auto_build:
            return configured

        model_dir = Path(getattr(self.raw_cfg.model, "model_dir", ""))
        graph_path = (
            Path(configured)
            if configured
            else Path.cwd() / "src" / "knowledge" / "data" / f"{model_dir.name}_spatial_graph.json"
        )
        if auto_build or not graph_path.is_file():
            SEIMSSpatialGraphBuilder(model_dir).write(graph_path)
        return str(graph_path)

    def _check_and_run_base_scenario(self) -> None:
        configured_base = self.raw_cfg.eval_info.get("BASE_ENV", -9999)
        run_base_each_time = self.raw_cfg.eval_info.get("RUN_BASE_EACH_TIME", True)
        use_precomputed = (
            self.eval_config.get("use_precomputed_base_env", False)
            or self.raw_cfg.eval_info.get("USE_PRECOMPUTED_BASE_ENV", False)
        )
        configured_periods = self.raw_cfg.eval_info.get("BASE_SED_PERIODS")
        if isinstance(configured_periods, (list, tuple)) and configured_periods:
            self.base_period_values = [float(value) for value in configured_periods]

        if (
            use_precomputed
            and configured_base
            and float(configured_base) > 0
            and (not self._use_period_sed_metric() or self.base_period_values)
        ):
            self.base_value = float(configured_base)
            return

        if use_precomputed and self.baseline_raster:
            self.base_value = self._masked_value(self.baseline_raster)
            self.raw_cfg.eval_info["BASE_ENV"] = self.base_value
            return

        cached = self.raw_cfg.eval_info.get("SPATIOTEMP_BASE_ENV")
        cached_raster = self.raw_cfg.eval_info.get("SPATIOTEMP_BASE_RASTER")
        if not run_base_each_time and cached and cached_raster and os.path.isfile(cached_raster):
            self.base_value = float(cached)
            self.baseline_raster = cached_raster
            return

        print("[SpatioTemporalSEIMSEvaluator] Running masked baseline scenario...")
        sce = SUScenario(self.raw_cfg)
        zero_genes = [0] * self.raw_cfg.genes_num
        sce.initialize(input_genes=zero_genes)
        sce.set_unique_id(int(self.eval_config.get("baseline_scenario_id", 0)))
        if self.is_temporal:
            sce.decoding_with_bmps_order()
        else:
            sce.decoding()
        sce.export_to_mongodb()
        self._reset_model_period(sce)

        if not sce.execute_seims_model():
            raise RuntimeError("SEIMS baseline scenario execution failed.")

        if self.is_temporal:
            sce.calculate_environment_bmps_order()
        else:
            sce.calculate_environment()
        self._export_scenario_gtiff_if_possible(sce)

        baseline_raster = self._output_raster(sce)
        self.baseline_raster = str(baseline_raster)
        self.base_value = sce.sed_sum
        self.raw_cfg.eval_info["BASE_ENV"] = self.base_value
        if self.is_temporal and getattr(sce, "sed_per_period", None):
            self.raw_cfg.eval_info["BASE_SED_PERIODS"] = sce.sed_per_period
            self.base_period_values = [float(value) for value in sce.sed_per_period]
        self.raw_cfg.eval_info["SPATIOTEMP_BASE_ENV"] = self.base_value
        self.raw_cfg.eval_info["SPATIOTEMP_BASE_RASTER"] = str(baseline_raster)
        print(f"[SpatioTemporalSEIMSEvaluator] Native BASE_ENV = {self.base_value}")

    def evaluate(self, individual, context) -> float:
        try:
            sce = self._run_scenario(individual)
            individual.scenario = sce
            if self._use_period_sed_metric():
                env = self._period_sediment_reduction(sce)
                individual.sed_sum = getattr(sce, "sed_sum", None)
                individual.environment = env
                individual.spatiotemporal_metric = {
                    "objective_name": self.eval_config.get("objective_name"),
                    "temporal_scope": self.temporal_scope,
                    "baseline_periods": self.base_period_values,
                    "scenario_periods": list(getattr(sce, "sed_per_period", []) or []),
                }
                return env
            if self._use_native_seims_metric():
                individual.sed_sum = sce.sed_sum
                env = max(0.0, float(sce.environment))
                individual.environment = env
                return env

            scenario_raster = self._output_raster(sce)
            result = self.metric.calculate(
                self.baseline_raster,
                scenario_raster,
                self.resolved_scope,
                self.temporal_scope,
                self.metric_cfg,
            )
            individual.sed_sum = result.scenario_value
            individual.environment = result.objective_value
            individual.spatiotemporal_metric = result
            return result.objective_value
        except Exception as exc:
            print(f"[SpatioTemporalSEIMSEvaluator] Evaluate failed: {exc}")
            return getattr(self.raw_cfg, "worst_env", 0.0)

    def _run_scenario(self, individual):
        sce = SUScenario(self.raw_cfg)
        individual.id = sce.set_unique_id()
        setattr(sce, "gene_values", individual)
        if hasattr(sce, "boundary_adjustment") and not self.is_temporal:
            sce.boundary_adjustment()
        if self.is_temporal:
            sce.decoding_with_bmps_order()
        else:
            sce.decoding()
        sce.export_to_mongodb()
        self._reset_model_period(sce)

        constraint_attr = "satisfy_investment_constraints" if self.is_temporal else "satisfy_investment_constraints_spatial"
        if hasattr(sce, constraint_attr):
            constraint_result = getattr(sce, constraint_attr)
            if callable(constraint_result):
                constraint_result = constraint_result()
            satisfied = constraint_result[0]
            if not satisfied:
                individual.environment = getattr(sce, "worst_env", 0.0)
                individual.economy = getattr(sce, "worst_econ", 1e20)
                return sce

        if not sce.execute_seims_model():
            raise RuntimeError(f"SEIMS scenario execution failed for individual {individual.id}.")

        if self.is_temporal:
            satisfied, cost_parts = sce.satisfy_investment_constraints
            if satisfied and hasattr(sce, "calculate_economy_bmps_order"):
                sce.calculate_economy_bmps_order(*cost_parts)
                individual.economy = sce.economy
        elif hasattr(sce, "calculate_economy"):
            sce.calculate_economy()
            individual.economy = sce.economy
        if self.is_temporal:
            sce.calculate_environment_bmps_order()
        else:
            sce.calculate_environment()
        if hasattr(sce, "export_scenario_to_txt"):
            sce.export_scenario_to_txt()
        self._export_scenario_gtiff_if_possible(sce)
        return sce

    @staticmethod
    def _reset_model_period(sce) -> None:
        model = getattr(sce, "model", None)
        if model is None or not hasattr(model, "ResetSimulationPeriod"):
            return
        try:
            model.ResetSimulationPeriod()
        except Exception as exc:
            print(f"[SpatioTemporalSEIMSEvaluator] Simulation period reset skipped: {exc}")

    @staticmethod
    def _export_scenario_gtiff_if_possible(sce) -> None:
        if not hasattr(sce, "export_scenario_to_gtiff"):
            return
        try:
            sce.export_scenario_to_gtiff()
        except Exception as exc:
            print(f"[SpatioTemporalSEIMSEvaluator] Scenario GeoTIFF export skipped: {exc}")

    def _output_raster(self, sce) -> Path:
        raster = Path(sce.modelout_dir) / self.env_file
        if not raster.is_file():
            raise FileNotFoundError(f"Expected SEIMS raster output not found: {raster}")
        return raster

    def _masked_value(self, raster_path: str | Path) -> float:
        return self.metric.calculate(
            raster_path,
            raster_path,
            self.resolved_scope,
            self.temporal_scope,
            self.metric_cfg,
        ).baseline_value

    def _use_native_seims_metric(self) -> bool:
        if self.eval_config.get("use_native_seims_metric") is False:
            return False
        if self.eval_config.get("use_native_seims_metric", False):
            return True
        metric_name = str(self.metric_cfg.get("name", "")).lower()
        spatial_agg = str(self.metric_cfg.get("spatial_aggregation", "")).lower()
        comparison = str(self.metric_cfg.get("comparison", "reduction_rate")).lower()
        env_file = str(
            self.metric_cfg.get("env_file") or self.metric_cfg.get("value") or self.env_file
        ).lower()
        if comparison == "reduction_rate" and "sed_ol_sum" in env_file:
            return True
        full_scope = str(self.resolved_scope.raw).lower() in {"full_watershed", "watershed", "full watershed"}
        sum_reduction = (
            metric_name in {"", "sum_reduction_rate", "annualized_sum_reduction_rate"}
            or spatial_agg in {"", "sum"}
        ) and comparison == "reduction_rate"
        return full_scope and sum_reduction

    def _use_period_sed_metric(self) -> bool:
        return bool(self.is_temporal and self.eval_config.get("use_period_sed_metric", False))

    def _period_sediment_reduction(self, sce) -> float:
        scenario_periods = [float(value) for value in getattr(sce, "sed_per_period", []) or []]
        if not self.base_period_values or not scenario_periods:
            return max(0.0, float(getattr(sce, "environment", self.raw_cfg.worst_env)))

        indices = self._period_indices(len(scenario_periods))
        if not indices:
            indices = list(range(min(len(self.base_period_values), len(scenario_periods))))
        base = sum(self.base_period_values[idx] for idx in indices if idx < len(self.base_period_values))
        scenario = sum(scenario_periods[idx] for idx in indices if idx < len(scenario_periods))
        if base <= 0:
            return 0.0
        return max(0.0, (base - scenario) * 100.0 / base)

    def _period_indices(self, period_count: int) -> list[int]:
        start = _parse_datetime(self.temporal_scope.get("start"), end_of_day=False)
        end = _parse_datetime(self.temporal_scope.get("end"), end_of_day=True)
        raw_start = getattr(self.raw_cfg, "eval_stime", None)
        if raw_start is not None:
            sim_start_year = raw_start.year
        else:
            sim_start = self.eval_config.get("sim_time_start") or self.eval_config.get("output_time_start")
            if sim_start:
                sim_start_year = _parse_datetime(str(sim_start), end_of_day=False).year
            else:
                sim_start_year = start.year
        return [
            year - sim_start_year
            for year in range(start.year, end.year + 1)
            if 0 <= year - sim_start_year < period_count
        ]


def _parse_datetime(value: str, end_of_day: bool) -> datetime:
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            if fmt == "%Y-%m-%d" and end_of_day:
                return parsed.replace(hour=23, minute=59, second=59)
            return parsed
        except ValueError:
            continue
    raise ValueError(f"Unsupported temporal_scope datetime: {value}")
