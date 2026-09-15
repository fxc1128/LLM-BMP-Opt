from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src.modeling.stage_chain import OpenSourceStageChain


class OpenPipeline:
    """Session-aware public API for the BDU -> payload -> code workflow."""

    def __init__(self, llm: Any | None = None, generated_root: str | Path = "generated_code"):
        self.llm = llm or self._make_llm_from_env()
        self.generated_root = Path(generated_root)
        self._history: dict[str, list[Any]] = {}
        self._requirement_templates: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _make_llm_from_env() -> ChatOpenAI:
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        model = os.getenv("LLM_MODEL", "deepseek-chat")
        base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
        if not api_key:
            raise RuntimeError("Set LLM_API_KEY (or OPENAI_API_KEY / DEEPSEEK_API_KEY) before starting the API.")
        return ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0)

    async def interact(self, user_input: str, session_id: str | None = None, execution_mode: int = 3) -> dict[str, Any]:
        session_id = session_id or str(uuid4())
        history = self._history.setdefault(session_id, [])
        output_dir = self.generated_root / session_id if execution_mode >= 3 else None
        result = await OpenSourceStageChain(self.llm).run(
            user_input=user_input,
            execution_mode=execution_mode,
            output_dir=output_dir,
            history=history,
            requirement_template=self._requirement_templates.get(session_id),
        )
        template = result.get("requirement_template")
        if isinstance(template, dict):
            self._requirement_templates[session_id] = template
        history.extend([HumanMessage(content=user_input), AIMessage(content=str(result.get("response", "")))])
        result["session_id"] = session_id
        return result
