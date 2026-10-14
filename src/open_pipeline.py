from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src.modeling.stage_chain import OpenSourceStageChain
from src.modeling.stage_contracts import validate_payload_artifact


class OpenPipeline:
    """Session-aware public API for the BDU -> payload -> code workflow."""

    def __init__(self, llm: Any | None = None, generated_root: str | Path = "generated_code"):
        # Runtime preflight/E4 do not need an LLM.  Initialize it lazily when an
        # interaction actually enters E1-E3 so execution remains available
        # after a service restart or in an offline runtime container.
        self.llm = llm
        self.generated_root = Path(generated_root)
        self._history: dict[str, list[Any]] = {}
        self._requirement_templates: dict[str, dict[str, Any]] = {}
        self._artifacts: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _make_llm_from_env() -> ChatOpenAI:
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        model = os.getenv("LLM_MODEL") or os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"
        base_url = os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "180"))
        max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
        if not api_key:
            raise RuntimeError("Set LLM_API_KEY (or OPENAI_API_KEY / DEEPSEEK_API_KEY) before starting the API.")
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    async def interact(
        self,
        user_input: str,
        session_id: str | None = None,
        execution_mode: int = 3,
        requirement_template: dict[str, Any] | None = None,
        bdu: dict[str, Any] | None = None,
        agent_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.llm is None:
            self.llm = self._make_llm_from_env()
        session_id = session_id or str(uuid4())
        history = self._history.setdefault(session_id, [])
        state = self._artifacts.setdefault(session_id, {})
        if requirement_template:
            self._requirement_templates[session_id] = requirement_template
        if bdu:
            state["bdu"] = bdu
        if agent_payload:
            restored_stage2 = state.setdefault("stage2", {})
            restored_stage2["agent_payload"] = agent_payload
            restored_stage2.setdefault("profile_id", agent_payload.get("profile_id"))
            restored_stage2.setdefault("support_status", agent_payload.get("support_status", "unknown"))
            restored_stage2["payload_validation"] = validate_payload_artifact(restored_stage2)
        output_dir = self.generated_root / session_id if execution_mode >= 3 else None
        result = await OpenSourceStageChain(self.llm).run(
            user_input=user_input,
            execution_mode=execution_mode,
            output_dir=output_dir,
            history=history,
            requirement_template=self._requirement_templates.get(session_id),
            bdu_artifact=state.get("bdu") if execution_mode >= 2 else None,
            stage2_output=state.get("stage2") if execution_mode >= 3 else None,
        )
        template = result.get("requirement_template")
        if isinstance(template, dict):
            self._requirement_templates[session_id] = template
        if isinstance(result.get("bdu"), dict):
            state["bdu"] = result["bdu"]
        if result.get("step") in {"FINISH_E2", "FINISH_E3"}:
            state["stage2"] = {
                "profile_id": (result.get("agent_payload") or {}).get("profile_id"),
                "support_status": (result.get("agent_payload") or {}).get("support_status"),
                "problem_formulation": result.get("problem_formulation", {}),
                "agent_payload": result.get("agent_payload", {}),
                "payload_validation": result.get("payload_validation", {}),
                "user_report": result.get("response", ""),
            }
        if result.get("step") == "FINISH_E3":
            state["package_dir"] = result.get("code_file_path")
        history.extend([HumanMessage(content=user_input), AIMessage(content=str(result.get("response", "")))])
        result["session_id"] = session_id
        return result

    def session_artifacts(self, session_id: str) -> dict[str, Any]:
        return dict(self._artifacts.get(session_id, {}))
