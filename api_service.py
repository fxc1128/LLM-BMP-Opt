from __future__ import annotations

import os
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.codegen.template_codegen import TemplateCodeGenerator
from src.open_pipeline import OpenPipeline
from src.optimization_runtime.capability import inspect_runtime_capability


app = FastAPI(title="LLM-BMP-Opt Open Pipeline", version="0.1.0")
pipeline: OpenPipeline | None = None


class InteractionRequest(BaseModel):
    user_input: str = Field(min_length=1)
    session_id: str | None = None
    execution_mode: Literal[1, 2, 3] = 3
    requirement_template: dict | None = None
    bdu: dict | None = None
    agent_payload: dict | None = None
    # Compatibility aliases used by the existing WPS proxy during migration.
    template: dict | None = None
    math_model: dict | None = None


class RuntimePreflightRequest(BaseModel):
    session_id: str | None = None
    agent_payload: dict | None = None
    code_files: dict[str, str] | None = None


class RunRequest(RuntimePreflightRequest):
    timeout_seconds: int = Field(default=600, ge=1, le=3600)


def get_pipeline() -> OpenPipeline:
    global pipeline
    if pipeline is None:
        pipeline = OpenPipeline(generated_root=os.getenv("GENERATED_CODE_DIR", "generated_code"))
    return pipeline


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "llm-bmp-opt-open"}


@app.post("/interact")
async def interact(request: InteractionRequest) -> dict:
    try:
        data = request.model_dump()
        requirement_template = data.pop("requirement_template")
        legacy_template = data.pop("template")
        data["requirement_template"] = requirement_template or legacy_template
        legacy_model = data.pop("math_model")
        if not data.get("agent_payload") and isinstance(legacy_model, dict):
            data["agent_payload"] = legacy_model.get("agent_payload", legacy_model)
        result = await get_pipeline().interact(**data)
        # Keep aliases while the WPS frontend migrates to the open contract.
        if "requirement_template" in result:
            result.setdefault("template", result["requirement_template"])
        if "agent_payload" in result:
            result.setdefault("math_model", {
                "problem_formulation": result.get("problem_formulation", {}),
                "agent_payload": result["agent_payload"],
            })
        return result
    except (RuntimeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM provider request failed: {exc}") from exc


@app.post("/runtime/preflight")
def runtime_preflight(request: RuntimePreflightRequest) -> dict:
    payload = request.agent_payload
    if payload is None and request.session_id:
        payload = (get_pipeline().session_artifacts(request.session_id).get("stage2") or {}).get("agent_payload")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="agent_payload is required or must exist in the session.")
    return inspect_runtime_capability(payload, request.code_files)


@app.post("/runE4")
def run_e4(request: RunRequest) -> dict:
    pipeline_instance = get_pipeline()
    artifacts = pipeline_instance.session_artifacts(request.session_id or "")
    payload = request.agent_payload or (artifacts.get("stage2") or {}).get("agent_payload")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="agent_payload is required or must exist in the session.")

    capability = inspect_runtime_capability(payload, request.code_files)
    if not capability["ready"]:
        status = capability["status"]
        return {
            "e4_status": "gap_skipped" if status in {"requires_custom_code", "requires_local_paths"} else status,
            "runtime_capability": capability,
            "stdout": "",
            "stderr": capability["message"],
            "pareto": [],
            "scientific_result": False,
        }

    package_dir = _resolve_package_dir(pipeline_instance, request.session_id, artifacts)
    if not (package_dir / "main.py").is_file() or not (package_dir / "runtime").is_dir():
        TemplateCodeGenerator().render_package(payload, output_dir=package_dir)
    if request.code_files:
        _write_code_files(package_dir, request.code_files)
    main_path = package_dir / "main.py"
    if not main_path.is_file():
        raise HTTPException(status_code=400, detail=f"Generated package has no main.py: {package_dir}")

    try:
        completed = subprocess.run(
            [sys.executable, "main.py"],
            cwd=package_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=request.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "e4_status": "timeout",
            "runtime_capability": capability,
            "stdout": exc.stdout or "",
            "stderr": f"Execution exceeded {request.timeout_seconds} seconds.",
            "pareto": [],
            "scientific_result": False,
        }

    pareto = _read_pareto(package_dir)
    success = completed.returncode == 0 and bool(pareto)
    return {
        "e4_status": "success" if success else "runtime_error",
        "runtime_capability": capability,
        "stdout": completed.stdout,
        "stderr": completed.stderr if completed.returncode else ("No Pareto result was produced." if not pareto else ""),
        "pareto": pareto,
        "diff_data": {},
        "scientific_result": success,
        "result_path": str(package_dir / "results" / "pareto_front.json") if success else None,
    }


def _resolve_package_dir(pipeline_instance: OpenPipeline, session_id: str | None, artifacts: dict) -> Path:
    raw = artifacts.get("package_dir")
    if raw:
        path = Path(raw)
    else:
        safe_session = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id or "default")
        path = pipeline_instance.generated_root / safe_session
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _write_code_files(package_dir: Path, code_files: dict[str, str]) -> None:
    allowed = {
        "agent_payload.py", "main.py", "requirements.txt", "encoder.py",
        "simulator.py", "evaluators.py", "constraints.py", "README.md",
    }
    for name, content in code_files.items():
        if name not in allowed:
            raise HTTPException(status_code=400, detail=f"Unsupported generated file: {name}")
        (package_dir / name).write_text(str(content), encoding="utf-8")


def _read_pareto(package_dir: Path) -> list[dict]:
    path = package_dir / "results" / "pareto_front.json"
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        return []
    rows = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        row = dict(item)
        row.setdefault("id", index + 1)
        row.setdefault("objectives", row.get("fitness", []))
        row.setdefault("bmp_config", row.get("genes", []))
        rows.append(row)
    return rows
