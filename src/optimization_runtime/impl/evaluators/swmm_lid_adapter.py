from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ...core.interfaces import AbstractEvaluator


DEFAULT_SWMM_LID_MODEL = (
    Path(__file__).resolve().parents[4]
    / "data"
    / "swmm_lid_urban_demo"
    / "model"
    / "urban_lid_demo.inp"
)


@dataclass(frozen=True)
class Subcatchment:
    name: str
    area_ha: float
    imperv_pct: float
    width_m: float
    slope_pct: float


@dataclass(frozen=True)
class LidControl:
    name: str
    runoff_reduction: float
    peak_reduction: float
    tss_reduction: float
    cost_per_m2: float


class SwmmLidEvaluator(AbstractEvaluator):
    """Deterministic lightweight SWMM/LID evaluator for P11.

    This adapter parses a small SWMM-style INP file and evaluates LID decisions
    with transparent event-scale equations. It is intended for executable case
    validation and optimization plumbing, not as a replacement for the EPA
    SWMM hydraulic engine.
    """

    SUPPORTED_METRICS = {"runoff_volume", "peak_flow", "tss_load", "total_cost"}

    def __init__(
        self,
        raw_cfg: Optional[dict] = None,
        is_temporal: bool = False,
        eval_config: Optional[dict] = None,
        **_: Any,
    ):
        cfg = dict(raw_cfg or {})
        cfg.update(eval_config or {})
        self.config = cfg
        self.is_temporal = is_temporal
        self.model_path = Path(cfg.get("model_path") or cfg.get("inp_path") or DEFAULT_SWMM_LID_MODEL)
        self.metric = str(cfg.get("metric", "runoff_volume")).strip().lower()
        if self.metric not in self.SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported SWMM/LID metric '{self.metric}'. "
                f"Supported metrics: {sorted(self.SUPPORTED_METRICS)}"
            )

        binding = cfg.get("decision_binding", {})
        self.actions: List[str] = list(
            binding.get(
                "actions",
                ["none", "bioretention", "permeable_pavement", "infiltration_trench", "vegetative_swale"],
            )
        )
        self.bound_subcatchments: Optional[List[str]] = binding.get("subcatchments")
        self.area_fraction_by_action = {
            str(k): float(v)
            for k, v in binding.get("area_fraction_by_action", {}).items()
        }
        self.default_area_fraction = float(binding.get("area_fraction", cfg.get("area_fraction", 0.35)))

        parsed = _parse_swmm_lid_inp(self.model_path)
        subcatchments = parsed["subcatchments"]
        if self.bound_subcatchments:
            wanted = set(self.bound_subcatchments)
            subcatchments = [s for s in subcatchments if s.name in wanted]
        if not subcatchments:
            raise ValueError(f"No subcatchments available in {self.model_path}")
        self.subcatchments: List[Subcatchment] = subcatchments
        self.lid_controls: Dict[str, LidControl] = parsed["lid_controls"]
        self.rainfall_mm = float(cfg.get("rainfall_mm", parsed["rainfall_mm"]))
        self.event_hours = max(float(cfg.get("event_hours", parsed["event_hours"])), 0.25)
        self.tss_emc_mg_l = float(cfg.get("tss_emc_mg_l", parsed["tss_emc_mg_l"]))
        self.last_result: Dict[str, float] = {}

    def evaluate(self, individual: Iterable[Any], context: Any) -> float:
        result = self.evaluate_all(individual)
        return float(result[self.metric])

    def evaluate_all(self, individual: Iterable[Any]) -> Dict[str, float]:
        genes = list(individual)
        if len(genes) < len(self.subcatchments):
            genes = genes + [0] * (len(self.subcatchments) - len(genes))

        runoff_volume = 0.0
        peak_flow = 0.0
        tss_load = 0.0
        total_cost = 0.0
        baseline_runoff = 0.0
        baseline_peak = 0.0
        baseline_tss = 0.0

        for idx, subcatchment in enumerate(self.subcatchments):
            action_name = self._action_name(genes[idx])
            lid = self.lid_controls.get(action_name)
            area_fraction = self._area_fraction(action_name)

            coeff = min(0.95, max(0.05, 0.18 + 0.0065 * subcatchment.imperv_pct))
            base_volume = self.rainfall_mm * subcatchment.area_ha * 10.0 * coeff
            base_peak = (base_volume / (self.event_hours * 3600.0)) * (1.15 + subcatchment.imperv_pct / 100.0)
            base_tss = base_volume * self.tss_emc_mg_l / 1000.0

            runoff_reduction = lid.runoff_reduction * area_fraction if lid else 0.0
            peak_reduction = lid.peak_reduction * area_fraction if lid else 0.0
            tss_reduction = lid.tss_reduction * area_fraction if lid else 0.0

            volume = base_volume * (1.0 - runoff_reduction)
            peak = base_peak * (1.0 - peak_reduction)
            tss = base_tss * (1.0 - runoff_reduction) * (1.0 - tss_reduction)
            cost = (
                subcatchment.area_ha * 10000.0 * area_fraction * lid.cost_per_m2
                if lid
                else 0.0
            )

            baseline_runoff += base_volume
            baseline_peak += base_peak
            baseline_tss += base_tss
            runoff_volume += volume
            peak_flow += peak
            tss_load += tss
            total_cost += cost

        self.last_result = {
            "runoff_volume": runoff_volume,
            "peak_flow": peak_flow,
            "tss_load": tss_load,
            "total_cost": total_cost,
            "baseline_runoff_volume": baseline_runoff,
            "baseline_peak_flow": baseline_peak,
            "baseline_tss_load": baseline_tss,
            "subcatchment_count": float(len(self.subcatchments)),
            "rainfall_mm": self.rainfall_mm,
        }
        return dict(self.last_result)

    def _action_name(self, gene: Any) -> str:
        try:
            action_idx = int(gene)
        except Exception:
            action_idx = 0
        if action_idx < 0 or action_idx >= len(self.actions):
            action_idx = 0
        return self.actions[action_idx]

    def _area_fraction(self, action_name: str) -> float:
        if action_name == "none":
            return 0.0
        return max(
            0.0,
            min(
                1.0,
                self.area_fraction_by_action.get(action_name, self.default_area_fraction),
            ),
        )


def _parse_swmm_lid_inp(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"SWMM/LID INP file not found: {path}")

    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";;"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line.strip("[]").upper()
            sections.setdefault(current, [])
            continue
        if current:
            sections[current].append(line)

    subcatchments = _parse_subcatchments(sections.get("SUBCATCHMENTS", []))
    lid_controls = _parse_lid_controls(sections.get("LID_CONTROLS", []))
    rainfall_values = _parse_timeseries_values(sections.get("TIMESERIES", []))
    rainfall_mm = sum(v for v in rainfall_values if v > 0.0) or 50.0
    event_hours = max((len(rainfall_values) - 1) * (5.0 / 60.0), 0.25)
    tss_emc = _parse_tss_emc(sections.get("POLLUTANTS", []))
    return {
        "subcatchments": subcatchments,
        "lid_controls": lid_controls,
        "rainfall_mm": rainfall_mm,
        "event_hours": event_hours,
        "tss_emc_mg_l": tss_emc,
    }


def _parse_subcatchments(lines: Iterable[str]) -> List[Subcatchment]:
    result: List[Subcatchment] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 7:
            continue
        result.append(
            Subcatchment(
                name=parts[0],
                area_ha=float(parts[3]),
                imperv_pct=float(parts[4]),
                width_m=float(parts[5]),
                slope_pct=float(parts[6]),
            )
        )
    return result


def _parse_lid_controls(lines: Iterable[str]) -> Dict[str, LidControl]:
    controls: Dict[str, LidControl] = {}
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        controls[parts[0]] = LidControl(
            name=parts[0],
            runoff_reduction=float(parts[1]),
            peak_reduction=float(parts[2]),
            tss_reduction=float(parts[3]),
            cost_per_m2=float(parts[4]),
        )
    return controls


def _parse_timeseries_values(lines: Iterable[str]) -> List[float]:
    values: List[float] = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            try:
                values.append(float(parts[2]))
            except ValueError:
                continue
    return values


def _parse_tss_emc(lines: Iterable[str]) -> float:
    for line in lines:
        parts = line.split()
        if len(parts) >= 3 and parts[0].upper() == "TSS":
            try:
                return float(parts[2])
            except ValueError:
                return 165.0
    return 165.0
