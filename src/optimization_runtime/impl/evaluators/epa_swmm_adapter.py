from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

from ...core.interfaces import AbstractEvaluator


class EpaSwmmEvaluator(AbstractEvaluator):
    """Run the EPA SWMM command-line engine for each candidate LID decision."""

    SUPPORTED_METRICS = {"runoff_volume", "peak_flow", "total_cost"}

    def __init__(self, raw_cfg=None, is_temporal: bool = False, eval_config=None, **_: Any):
        cfg = dict(raw_cfg or {})
        cfg.update(eval_config or {})
        self.model_path = Path(cfg["model_path"]).resolve()
        self.executable = Path(cfg["executable"]).resolve()
        self.work_root = Path(cfg.get("work_root", "runs/swmm")).resolve()
        self.metric = str(cfg.get("metric", "runoff_volume")).lower()
        self.lid_unit_cost = float(cfg.get("lid_unit_cost", 95.0))
        self.lid_area_m2 = float(cfg.get("lid_area_m2", 102.0))
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
        enabled = bool(list(individual) and int(list(individual)[0]) > 0)
        self.work_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="swmm_", dir=self.work_root) as temp_dir:
            workspace = Path(temp_dir)
            inp = workspace / "model.inp"
            rpt = workspace / "model.rpt"
            out = workspace / "model.out"
            inp.write_text(self._materialize_lid(enabled), encoding="utf-8")
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
            runoff, peak = self._parse_report(rpt)
        return {
            "runoff_volume": runoff,
            "peak_flow": peak,
            "total_cost": self.lid_area_m2 * self.lid_unit_cost if enabled else 0.0,
        }

    def _materialize_lid(self, enabled: bool) -> str:
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
                if len(parts) >= 3 and parts[0] != "[LID_USAGE]":
                    parts[2] = "1" if enabled else "0"
                    line = "  ".join(parts)
            result.append(line)
        return "\n".join(result) + "\n"

    @staticmethod
    def _parse_report(report_path: Path) -> tuple[float, float]:
        lines = report_path.read_text(encoding="utf-8", errors="replace").splitlines()
        in_runoff_summary = False
        for line in lines:
            if "Subcatchment Runoff Summary" in line:
                in_runoff_summary = True
                continue
            if in_runoff_summary and line.lstrip().startswith("NC_Bioretention"):
                values = line.split()
                if len(values) >= 5:
                    return float(values[-3]), float(values[-2])
        raise RuntimeError("EPA SWMM report did not contain a subcatchment runoff summary.")
