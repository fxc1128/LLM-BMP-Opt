from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_DIR = PROJECT_ROOT / "profiles"


@dataclass(frozen=True)
class RuntimeProfile:
    profile_id: str
    data: dict[str, Any]
    path: Path

    @property
    def model_family(self) -> str:
        return str(self.data.get("model_family", "unknown"))

    @property
    def support_status(self) -> str:
        return str(self.data.get("support_status", "unknown"))

    def default_payload_parts(self) -> dict[str, Any]:
        return {
            "decision_variable": deepcopy(self.data.get("default_decision_variable", {})),
            "evaluator": deepcopy(self.data.get("default_evaluator_config", {})),
            "solver": deepcopy(self.data.get("default_solver", {})),
        }

    def prompt_summary(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "display_name": self.data.get("display_name"),
            "model_family": self.model_family,
            "support_status": self.support_status,
            "aliases": self.data.get("aliases", []),
            "supported_targets": self.data.get("supported_targets", []),
            "supported_units": self.data.get("supported_units", []),
            "supported_measures": self.data.get("supported_measures", []),
            "default_decision_variable": self.data.get("default_decision_variable", {}),
            "default_evaluator_config": self.data.get("default_evaluator_config", {}),
            "default_solver": self.data.get("default_solver", {}),
            "payload_notes": self.data.get("payload_notes", []),
        }


class ProfileRegistry:
    """Read-only registry for external model/runtime profiles.

    Profiles are intentionally data files. Stage 2 may choose one and generate a
    complete payload from it, but the registry does not infer demand semantics.
    """

    def __init__(self, profile_dir: str | Path = DEFAULT_PROFILE_DIR):
        self.profile_dir = Path(profile_dir)
        self._profiles = self._load_profiles(self.profile_dir)

    def list_profiles(self) -> list[RuntimeProfile]:
        return list(self._profiles.values())

    def get(self, profile_id: str) -> RuntimeProfile:
        key = str(profile_id)
        if key not in self._profiles:
            available = ", ".join(sorted(self._profiles))
            raise KeyError(f"Unknown profile_id '{profile_id}'. Available profiles: {available}")
        return self._profiles[key]

    def prompt_context(self) -> str:
        summaries = [profile.prompt_summary() for profile in self.list_profiles()]
        return json.dumps({"runtime_profiles": summaries}, ensure_ascii=False, indent=2)

    def find_candidates(self, text: str, limit: int = 3) -> list[RuntimeProfile]:
        lowered = str(text or "").lower()
        scored: list[tuple[int, RuntimeProfile]] = []
        for profile in self.list_profiles():
            score = 0
            haystacks: Iterable[Any] = (
                [profile.profile_id, profile.model_family]
                + list(profile.data.get("aliases", []))
                + list(profile.data.get("supported_targets", []))
                + list(profile.data.get("supported_units", []))
            )
            for item in haystacks:
                item_text = str(item).lower()
                if item_text and item_text in lowered:
                    score += 1
            if score:
                scored.append((score, profile))
        scored.sort(key=lambda item: (-item[0], item[1].profile_id))
        return [profile for _, profile in scored[:limit]]

    @staticmethod
    def _load_profiles(profile_dir: Path) -> dict[str, RuntimeProfile]:
        profiles: dict[str, RuntimeProfile] = {}
        if not profile_dir.is_dir():
            return profiles
        for path in sorted(profile_dir.rglob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, Mapping):
                continue
            profile_id = str(data.get("profile_id") or path.stem)
            profiles[profile_id] = RuntimeProfile(profile_id, dict(data), path)
        return profiles


def merge_profile_defaults(agent_payload: dict[str, Any], profile: RuntimeProfile) -> dict[str, Any]:
    """Merge explicit Stage 2 payload with profile defaults without semantic guessing."""
    payload = deepcopy(agent_payload)
    defaults = profile.default_payload_parts()

    opt = payload.setdefault("optimization_problem", {})
    if isinstance(opt, dict):
        opt.setdefault("decision_variable", defaults["decision_variable"])
        opt.setdefault("constraints", [])
        opt.setdefault("objectives", [])

    evaluator = payload.setdefault("evaluator", {})
    if isinstance(evaluator, dict):
        for key, value in defaults["evaluator"].items():
            if key not in evaluator:
                evaluator[key] = deepcopy(value)
            elif isinstance(evaluator[key], dict) and isinstance(value, dict):
                evaluator[key] = _deep_default_merge(evaluator[key], value)

    payload.setdefault("solver", defaults["solver"])
    payload.setdefault("profile_id", profile.profile_id)
    payload.setdefault("support_status", profile.support_status)
    return payload


def _deep_default_merge(current: dict[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(current)
    for key, value in defaults.items():
        if key not in merged:
            merged[key] = deepcopy(value)
        elif isinstance(merged[key], dict) and isinstance(value, Mapping):
            merged[key] = _deep_default_merge(merged[key], value)
    return merged
