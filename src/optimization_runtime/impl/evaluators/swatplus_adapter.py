from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ...core.interfaces import AbstractEvaluator


DEFAULT_FTCREEK_MODEL_DIR = Path(".")


@dataclass(frozen=True)
class SwatPlusRunResult:
    value: float
    workspace: Path
    output_file: Path
    run_returncode: int
    metric: str
    aggregation: str


class SwatPlusEvaluator(AbstractEvaluator):
    """Real SWAT+ evaluator.

    The evaluator copies a SWAT+ TxtInOut model, materializes BMP choices into
    management input files, runs the SWAT+ executable, and parses model outputs.
    It intentionally fails on missing files or failed runs; no placeholder value
    is returned for an unavailable SWAT+ model.
    """

    def __init__(self, raw_cfg=None, is_temporal: bool = False, **kwargs):
        cfg = _as_dict(kwargs.get("eval_config")) or _as_dict(raw_cfg)
        context_cfg = _as_dict(kwargs.get("context"))
        swat_cfg = _as_dict(context_cfg.get("swatplus")) or _as_dict(context_cfg.get("swat"))
        cfg = {**swat_cfg, **cfg}

        self.template_dir = Path(
            cfg.get("template_dir")
            or cfg.get("model_dir")
            or cfg.get("MODEL_DIR")
            or DEFAULT_FTCREEK_MODEL_DIR
        )
        self.executable = Path(
            cfg.get("executable")
            or cfg.get("EXE")
            or cfg.get("bin_path")
            or self.template_dir / "SWAT+.exe"
        )
        self.work_root = Path(
            cfg.get("work_root")
            or cfg.get("workspace_root")
            or Path("sandbox_results") / "swatplus_runs"
        )
        self.timeout_seconds = int(cfg.get("timeout_seconds", 300))
        self.keep_workspaces = bool(cfg.get("keep_workspaces", True))
        self.copy_existing_outputs = bool(cfg.get("copy_existing_outputs", False))

        self.output_file_name = str(cfg.get("output_file", "basin_ls_aa.txt"))
        self.metric = str(cfg.get("metric", cfg.get("column", "sedyld"))).lower()
        self.aggregation = str(cfg.get("aggregation", "mean")).lower()
        self.unit_ids = [str(x) for x in cfg.get("unit_ids", cfg.get("hru_ids", []))]
        self.year_range = cfg.get("year_range")

        binding = _as_dict(cfg.get("decision_binding"))
        self.binding_file = str(binding.get("file", cfg.get("binding_file", "landuse.lum")))
        self.binding_key_column = str(binding.get("key_column", cfg.get("key_column", "NAME")))
        self.binding_target_column = str(
            binding.get("target_column", cfg.get("target_column", "CONS_PRAC"))
        )
        self.action_values = list(
            binding.get("actions")
            or cfg.get("action_values")
            or ["up_down_slope", "contour", "filter_strip"]
        )
        self.target_lu_mgt_names = [
            str(x) for x in (
                binding.get("lu_mgt_names")
                or cfg.get("lu_mgt_names")
                or cfg.get("target_lu_mgt_names")
                or []
            )
        ]
        self.default_action_index = int(cfg.get("default_action_index", 0))
        self.last_result: SwatPlusRunResult | None = None

        self._validate_static_inputs()

    def evaluate(self, individual: Any, context: Any) -> float:
        decisions = _individual_to_list(individual)
        workspace = self._workspace_for(decisions)
        if not (workspace / "success.fin").is_file():
            self._prepare_workspace(workspace)
            self._materialize_decisions(workspace, decisions)
            run = self._run_swatplus(workspace)
        else:
            run = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        value, output_path = self._parse_metric(workspace)
        self.last_result = SwatPlusRunResult(
            value=value,
            workspace=workspace,
            output_file=output_path,
            run_returncode=run.returncode,
            metric=self.metric,
            aggregation=self.aggregation,
        )
        return value

    def _validate_static_inputs(self) -> None:
        missing = []
        for path in [
            self.template_dir,
            self.executable,
            self.template_dir / "file.cio",
            self.template_dir / "time.sim",
            self.template_dir / self.binding_file,
        ]:
            if not path.exists():
                missing.append(str(path))
        if missing:
            raise FileNotFoundError("SWAT+ evaluator missing required inputs: " + "; ".join(missing))

    def _workspace_for(self, decisions: Sequence[Any]) -> Path:
        material = "|".join(map(str, decisions)) or "baseline"
        digest = hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]
        return self.work_root / f"swatplus_{digest}"

    def _prepare_workspace(self, workspace: Path) -> None:
        workspace.parent.mkdir(parents=True, exist_ok=True)
        if workspace.exists():
            shutil.rmtree(workspace)
        if self.copy_existing_outputs:
            shutil.copytree(self.template_dir, workspace)
        else:
            shutil.copytree(self.template_dir, workspace, ignore=_ignore_swat_outputs)

    def _materialize_decisions(self, workspace: Path, decisions: Sequence[Any]) -> None:
        if not decisions:
            decisions = [self.default_action_index]
        path = workspace / self.binding_file
        table = _read_swat_table(path)
        key_idx = _column_index(table.headers, self.binding_key_column)
        target_idx = _column_index(table.headers, self.binding_target_column)

        target_list = list(self.target_lu_mgt_names)
        if not target_list:
            target_list = [row[key_idx] for row in table.rows[: len(decisions)]]
        targets = set(target_list)

        decision_by_target: dict[str, Any] = {}
        for idx, target in enumerate(target_list):
            decision_by_target[target] = decisions[idx] if idx < len(decisions) else decisions[-1]

        changed = 0
        for row in table.rows:
            key = row[key_idx]
            if key not in targets:
                continue
            row[target_idx] = self._decision_to_action(decision_by_target[key])
            changed += 1
        if changed == 0:
            raise ValueError(
                f"No SWAT+ management rows matched {self.binding_key_column} values: {sorted(targets)}"
            )
        _write_swat_table(path, table)

    def _decision_to_action(self, decision: Any) -> str:
        if isinstance(decision, str):
            if decision.isdigit():
                return self.action_values[int(decision)]
            return decision
        index = int(decision)
        if index < 0 or index >= len(self.action_values):
            raise ValueError(f"SWAT+ decision index {index} outside action_values={self.action_values}")
        return str(self.action_values[index])

    def _run_swatplus(self, workspace: Path) -> subprocess.CompletedProcess[str]:
        exe = workspace / self.executable.name
        if not exe.exists():
            shutil.copy2(self.executable, exe)
        run = subprocess.run(
            [str(exe)],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        if run.returncode != 0:
            raise RuntimeError(
                f"SWAT+ failed with exit code {run.returncode} in {workspace}\n"
                f"stdout:\n{run.stdout[-2000:]}\n"
                f"stderr:\n{run.stderr[-2000:]}"
            )
        if not (workspace / "success.fin").is_file():
            raise RuntimeError(f"SWAT+ run did not produce success.fin in {workspace}")
        return run

    def _parse_metric(self, workspace: Path) -> tuple[float, Path]:
        output_path = workspace / self.output_file_name
        if not output_path.is_file():
            raise FileNotFoundError(f"SWAT+ output file missing: {output_path}")
        table = _read_swat_output(output_path)
        metric_idx = _column_index(table.headers, self.metric)
        yr_idx = _optional_column_index(table.headers, "yr")
        unit_idx = _optional_column_index(table.headers, "unit")
        gis_idx = _optional_column_index(table.headers, "gis_id")

        values: list[float] = []
        for row in table.rows:
            if self.year_range and yr_idx is not None:
                year = int(float(row[yr_idx]))
                start, end = self.year_range
                if year < int(start) or year > int(end):
                    continue
            if self.unit_ids:
                unit_texts = {row[i] for i in (unit_idx, gis_idx) if i is not None}
                if not unit_texts.intersection(self.unit_ids):
                    continue
            values.append(float(row[metric_idx]))

        if not values:
            raise ValueError(
                f"No rows matched metric={self.metric}, units={self.unit_ids}, years={self.year_range}"
            )
        if self.aggregation == "sum":
            value = sum(values)
        elif self.aggregation in ("max", "peak"):
            value = max(values)
        elif self.aggregation == "min":
            value = min(values)
        elif self.aggregation in ("mean", "average", "annual_average"):
            value = sum(values) / len(values)
        else:
            raise ValueError(f"Unsupported SWAT+ aggregation: {self.aggregation}")
        if not math.isfinite(value):
            raise ValueError(f"SWAT+ parsed non-finite value: {value}")
        return value, output_path


@dataclass
class _SwatTable:
    title: str
    headers: list[str]
    rows: list[list[str]]


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _individual_to_list(individual: Any) -> list[Any]:
    if individual is None:
        return []
    if isinstance(individual, (str, bytes)):
        return [individual.decode() if isinstance(individual, bytes) else individual]
    try:
        return list(individual)
    except TypeError:
        return [individual]


def _column_index(headers: Sequence[str], name: str) -> int:
    lname = name.lower()
    for idx, header in enumerate(headers):
        if header.lower() == lname:
            return idx
    raise KeyError(f"Column '{name}' not found in SWAT+ table headers: {headers}")


def _optional_column_index(headers: Sequence[str], name: str) -> int | None:
    try:
        return _column_index(headers, name)
    except KeyError:
        return None


def _read_swat_table(path: Path) -> _SwatTable:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        raise ValueError(f"SWAT+ table has fewer than two lines: {path}")
    title = lines[0]
    headers = lines[1].split()
    rows = [line.split() for line in lines[2:] if line.strip()]
    for row in rows:
        if len(row) != len(headers):
            raise ValueError(f"Malformed SWAT+ row in {path}: {row}")
    return _SwatTable(title=title, headers=headers, rows=rows)


def _write_swat_table(path: Path, table: _SwatTable) -> None:
    lines = [table.title, " ".join(table.headers)]
    lines.extend(" ".join(row) for row in table.rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_swat_output(path: Path) -> _SwatTable:
    lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if len(lines) < 4:
        raise ValueError(f"SWAT+ output has no data rows: {path}")
    title = lines[0]
    headers = lines[1].split()
    rows = [line.split() for line in lines[3:]]
    rows = [row[: len(headers)] for row in rows if len(row) >= len(headers)]
    if not rows:
        raise ValueError(f"SWAT+ output has no parseable rows: {path}")
    return _SwatTable(title=title, headers=headers, rows=rows)


def _ignore_swat_outputs(directory: str, names: Iterable[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        lower = name.lower()
        if lower in {"success.fin", "simulation.out", "diagnostics.out", "simulation.time"}:
            ignored.add(name)
        elif lower.startswith("fort."):
            ignored.add(name)
        elif lower.endswith(("_day.txt", "_mon.txt", "_yr.txt", "_aa.txt")):
            ignored.add(name)
        elif lower.startswith(("cs_", "gwflow_balance_", "gwflow_flux_", "gwflow_state_")):
            ignored.add(name)
    return ignored
