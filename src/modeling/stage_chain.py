from __future__ import annotations

from pathlib import Path
from typing import Any

from src.experts.stage1_bdu_parser import Stage1BDUParser
from src.experts.stage2_payload_modeler import Stage2PayloadModeler
from src.experts.stage3_code_generator import Stage3LLMCodeGenerator
from src.codegen.template_codegen import TemplateCodeGenerator
from src.modeling.profile_registry import ProfileRegistry


class OpenSourceStageChain:
    """Open-source three-stage chain: BDU -> payload -> code.

    This is additive and keeps the existing legacy pipeline/runtime untouched.
    """

    def __init__(
        self,
        llm,
        profile_registry: ProfileRegistry | None = None,
        code_generator: TemplateCodeGenerator | None = None,
    ):
        self.profile_registry = profile_registry or ProfileRegistry()
        self.stage1 = Stage1BDUParser(llm)
        self.stage2 = Stage2PayloadModeler(llm, self.profile_registry)
        self.codegen = Stage3LLMCodeGenerator(llm, package_writer=code_generator or TemplateCodeGenerator())

    async def run(
        self,
        user_input: str,
        execution_mode: int = 3,
        output_dir: str | Path | None = None,
        history: list[Any] | None = None,
        requirement_template: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        bdu = await self.stage1.parse(user_input, history=history, requirement_template=requirement_template)
        if bdu.get("status") == "needs_clarification" or execution_mode == 1:
            return {
                "step": "NEEDS_CLARIFICATION" if bdu.get("status") == "needs_clarification" else "FINISH_STAGE1",
                "bdu": bdu,
                "governance_checklist": bdu.get("governance_checklist", []),
                "requirement_template": bdu.get("requirement_template", {}),
                "missing_fields": bdu.get("missing_fields", []),
                "response": bdu.get("user_message", ""),
            }

        stage2 = await self.stage2.build_payload(bdu)
        if execution_mode == 2:
            return {
                "step": "FINISH_STAGE2",
                "bdu": bdu,
                "governance_checklist": bdu.get("governance_checklist", []),
                "requirement_template": bdu.get("requirement_template", {}),
                "problem_formulation": stage2.get("problem_formulation", {}),
                "agent_payload": stage2.get("agent_payload", {}),
                "payload_validation": stage2.get("payload_validation", {}),
                "response": stage2.get("user_report", ""),
            }

        package = await self.codegen.generate(stage2, output_dir=output_dir)
        return {
            "step": "FINISH_STAGE3",
            "bdu": bdu,
            "governance_checklist": bdu.get("governance_checklist", []),
            "requirement_template": bdu.get("requirement_template", {}),
            "problem_formulation": stage2.get("problem_formulation", {}),
            "agent_payload": stage2.get("agent_payload", {}),
            "payload_validation": stage2.get("payload_validation", {}),
            "code_files": package.files,
            "code_file_path": str(package.output_dir) if package.output_dir else None,
            "generated_code": package.as_fenced_blocks(),
            "response": stage2.get("user_report", ""),
        }
