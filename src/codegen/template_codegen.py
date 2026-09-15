from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
import shutil
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


@dataclass
class CodePackage:
    files: list[str]
    contents: dict[str, str]
    output_dir: Path | None = None

    def as_fenced_blocks(self) -> str:
        blocks = []
        for filename in self.files:
            if filename not in self.contents:
                continue
            blocks.append(f"```{filename}\n{self.contents[filename].rstrip()}\n```")
        return "\n\n".join(blocks) + "\n"


class TemplateCodeGenerator:
    """Bundle generated code with the public optimization runtime."""

    def __init__(
        self,
        template_dir: str | Path = DEFAULT_TEMPLATE_DIR,
        project_root: str | Path = PROJECT_ROOT,
    ):
        self.template_dir = Path(template_dir)
        self.project_root = Path(project_root)

    def render_package(
        self,
        agent_payload: dict[str, Any],
        output_dir: str | Path | None = None,
        include_custom_evaluators: bool = False,
    ) -> CodePackage:
        context = {
            "payload_literal": pformat(agent_payload, width=100, sort_dicts=False),
            "project_root": str(self.project_root).replace("\\", "\\\\"),
            "requirements": self._requirements(agent_payload),
        }
        files = ["agent_payload.py"]
        contents = {
            "agent_payload.py": self._render("agent_payload.py.tpl", context),
            "main.py": self._render("main.py.tpl", context),
            "requirements.txt": self._render("requirements.txt.tpl", context),
        }
        if include_custom_evaluators:
            contents["evaluators.py"] = self._render("evaluators.py.tpl", context)
            files.append("evaluators.py")
        files.extend(["main.py", "requirements.txt", "runtime/"])

        out_path = Path(output_dir) if output_dir is not None else None
        if out_path is not None:
            out_path.mkdir(parents=True, exist_ok=True)
            for filename in files:
                if filename in contents:
                    (out_path / filename).write_text(contents[filename], encoding="utf-8")
            runtime_source = self.project_root / "src" / "optimization_runtime"
            if not runtime_source.is_dir():
                raise FileNotFoundError(f"Bundled runtime source was not found: {runtime_source}")
            shutil.copytree(
                runtime_source,
                out_path / "runtime",
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        return CodePackage(files=files, contents=contents, output_dir=out_path)

    def write_llm_package(self, contents: dict[str, str], output_dir: str | Path | None = None) -> CodePackage:
        """Persist Stage 3 LLM code and copy the shared runtime without rewriting it."""
        required = {"agent_payload.py", "main.py", "requirements.txt"}
        missing = required.difference(contents)
        if missing:
            raise ValueError(f"LLM package is missing required files: {', '.join(sorted(missing))}")
        allowed = required | {"encoder.py", "simulator.py", "evaluators.py", "constraints.py", "README.md"}
        unexpected = set(contents).difference(allowed)
        if unexpected:
            raise ValueError(f"LLM package contains unsupported files: {', '.join(sorted(unexpected))}")
        files = [name for name in ("agent_payload.py", "main.py", "requirements.txt", "encoder.py", "simulator.py", "evaluators.py", "constraints.py", "README.md") if name in contents]
        files.append("runtime/")
        out_path = Path(output_dir) if output_dir is not None else None
        if out_path is not None:
            out_path.mkdir(parents=True, exist_ok=True)
            for filename, content in contents.items():
                (out_path / filename).write_text(content, encoding="utf-8")
            self._copy_runtime(out_path)
        return CodePackage(files=files, contents=contents, output_dir=out_path)

    def _copy_runtime(self, out_path: Path) -> None:
        runtime_source = self.project_root / "src" / "optimization_runtime"
        if not runtime_source.is_dir():
            raise FileNotFoundError(f"Bundled runtime source was not found: {runtime_source}")
        shutil.copytree(runtime_source, out_path / "runtime", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    def _render(self, template_name: str, context: dict[str, str]) -> str:
        text = (self.template_dir / template_name).read_text(encoding="utf-8")
        for key, value in context.items():
            text = text.replace("{{ " + key + " }}", value)
        return text

    @staticmethod
    def _requirements(agent_payload: dict[str, Any]) -> str:
        evaluators = agent_payload.get("evaluator", {})
        evaluator_keys = set(evaluators.keys()) if isinstance(evaluators, dict) else set()
        lines = ["deap>=1.3.0", "numpy>=1.21.0"]
        if evaluator_keys.intersection({"seims", "spatiotemporal_seims", "spatial_seims"}):
            lines.extend(["# pymongo>=3.12.0", "# pygeoc>=0.3.0"])
        if evaluator_keys.intersection({"swat", "swatplus", "swat+"}):
            lines.append("# SWAT+ executable and TxtInOut files are required locally.")
        return "\n".join(lines) + "\n"
