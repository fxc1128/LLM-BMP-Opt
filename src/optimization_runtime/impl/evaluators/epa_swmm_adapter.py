from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from ...core.interfaces import AbstractEvaluator


class EpaSwmmEvaluator(AbstractEvaluator):
    """Run the EPA SWMM command-line engine for each candidate LID decision."""

    _simulation_cache: dict[tuple[Any, ...], dict[str, float]] = {}

    SUPPORTED_METRICS = {
        "runoff_volume",
        "peak_flow",
        "outlet_runoff_volume",
        "outlet_peak_flow",
        "total_cost",
    }

    def __init__(self, raw_cfg=None, is_temporal: bool = False, eval_config=None, **_: Any):
        cfg = dict(raw_cfg or {})
        cfg.update(eval_config or {})
        self.model_path = Path(cfg["model_path"]).resolve()
        self.executable = Path(cfg["executable"]).resolve()
        self.work_root = Path(cfg.get("work_root", "runs/swmm")).resolve()
        self.metric = str(cfg.get("metric", "runoff_volume")).lower()
        self.lid_unit_cost = float(cfg.get("lid_unit_cost", 95.0))
        self.lid_area_m2 = float(cfg.get("lid_area_m2", 102.0))
        self.target_subcatchment = str(cfg.get("target_subcatchment", "NC_Bioretention"))
        self.enabled_lid_count = int(cfg.get("enabled_lid_count", 1))
        decision_binding = cfg.get("decision_binding", {})
        if not isinstance(decision_binding, Mapping):
            decision_binding = {}
        self.decision_subcatchments = [
            str(value)
            for value in decision_binding.get("subcatchments", [])
        ]
        self.outlet_node = str(cfg.get("outlet_node", "Outlet"))
        self.unit_capex_usd = float(cfg.get("unit_capex_usd", 0.0))
        self.unit_installation_usd = float(cfg.get("unit_installation_usd", 0.0))
        self.annual_om_per_unit_usd = float(cfg.get("annual_om_per_unit_usd", 0.0))
        self.analysis_years = int(cfg.get("analysis_years", 0))
        self.discount_rate = float(cfg.get("discount_rate", 0.0))
        self.timeout_seconds = int(cfg.get("timeout_seconds", 60))
        if self.metric not in self.SUPPORTED_METRICS:
            raise ValueError(f"Unsupported EPA SWMM metric: {self.metric}")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"SWMM model not found: {self.model_path}")
        if not self.executable.is_file():
            raise FileNotFoundError(f"SWMM executable not found: {self.executable}")

    def evaluate(self, individual: Iterable[Any], context: Any) -> float:
        return float(self.evaluate_all(individual)[self.metric])

    def evaluate_all(self, individual: Iterable[Any]) -> dict[str, float]:
        decisions = self._decisions(individual)
        if self.metric == "total_cost":
            return {"total_cost": self._total_cost(decisions)}
        cache_key = self._cache_key(decisions)
        cached = self._simulation_cache.get(cache_key)
        if cached is not None:
            return {**cached, "total_cost": self._total_cost(decisions)}
        self.work_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="swmm_", dir=self.work_root) as temp_dir:
            workspace = Path(temp_dir)
            inp = workspace / "model.inp"
            rpt = workspace / "model.rpt"
            out = workspace / "model.out"
            inp.write_text(self._materialize_lid(decisions), encoding="utf-8")
            completed = subprocess.run(
                [str(self.executable), str(inp), str(rpt), str(out)],
                cwd=workspace,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
            if completed.returncode != 0 or not rpt.is_file():
                raise RuntimeError(f"EPA SWMM failed: {completed.stdout}\n{completed.stderr}")
            metrics = self._parse_report(rpt)
        self._simulation_cache[cache_key] = metrics
        return {
            **metrics,
            "total_cost": self._total_cost(decisions),
        }

    def _decisions(self, individual: Iterable[Any]) -> dict[str, int]:
        values = [max(0, int(value)) for value in individual]
        if self.decision_subcatchments:
            return {
                subcatchment: values[index] if index < len(values) else 0
                for index, subcatchment in enumerate(self.decision_subcatchments)
            }
        return {self.target_subcatchment: self.enabled_lid_count if values and values[0] > 0 else 0}

    def _cache_key(self, decisions: Mapping[str, int]) -> tuple[Any, ...]:
        return (
            str(self.model_path),
            str(self.executable),
            self.target_subcatchment,
            self.outlet_node,
            tuple(sorted(decisions.items())),
        )

    def _total_cost(self, decisions: Mapping[str, int]) -> float:
        count = sum(decisions.values())
        capital_cost = count * (self.unit_capex_usd + self.unit_installation_usd)
        if self.analysis_years <= 0 or self.annual_om_per_unit_usd <= 0:
            return capital_cost
        if self.discount_rate <= 0:
            return capital_cost + count * self.annual_om_per_unit_usd * self.analysis_years
        annuity_factor = sum(
            1.0 / ((1.0 + self.discount_rate) ** year)
            for year in range(1, self.analysis_years + 1)
        )
        return capital_cost + count * self.annual_om_per_unit_usd * annuity_factor

    def _materialize_lid(self, decisions: Mapping[str, int]) -> str:
        lines = self.model_path.read_text(encoding="utf-8").splitlines()
        in_lid_usage = False
        result: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.upper() == "[LID_USAGE]":
                in_lid_usage = True
            elif stripped.startswith("[") and stripped.endswith("]"):
                in_lid_usage = False
            if in_lid_usage and stripped and not stripped.startswith(";"):
                parts = line.split()
                if len(parts) >= 3 and parts[0] in decisions:
                    parts[2] = str(decisions[parts[0]])
                    line = "  ".join(parts)
            result.append(line)
        return "\n".join(result) + "\n"

    def _parse_report(self, report_path: Path) -> dict[str, float]:
        lines = report_path.read_text(encoding="utf-8", errors="replace").splitlines()
        in_runoff_summary = False
        in_outfall_summary = False
        metrics: dict[str, float] = {}
        for line in lines:
            if "Subcatchment Runoff Summary" in line:
                in_runoff_summary = True
                continue
            if "Outfall Loading Summary" in line:
                in_outfall_summary = True
                continue
            if (
                in_runoff_summary
                and "runoff_volume" not in metrics
                and line.lstrip().startswith(self.target_subcatchment)
            ):
                values = line.split()
                if len(values) >= 5:
                    metrics["runoff_volume"] = float(values[-3])
                    metrics["peak_flow"] = float(values[-2])
            if in_outfall_summary and line.lstrip().startswith(self.outlet_node):
                values = line.split()
                if len(values) >= 5 and "outlet_peak_flow" not in metrics:
                    # [OUTFALL LOADING SUMMARY]: name, flow frequency, average
                    # flow, maximum flow, total volume, then pollutant loads.
                    metrics["outlet_peak_flow"] = float(values[3])
                    metrics["outlet_runoff_volume"] = float(values[4])
        required = {"runoff_volume", "peak_flow", "outlet_runoff_volume", "outlet_peak_flow"}
        if not required.issubset(metrics):
            raise RuntimeError("EPA SWMM report did not contain the configured runoff and outlet summaries.")
        return metrics

