from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"


class OpenKnowledgeBase:
    """Read-only Chinese knowledge-base loader for the public pipeline."""

    _FILES = {
        "stage1": ("geo_opt_kgb.json",),
        "stage2": ("geo_opt_math_kb.json",),
        "stage3": ("geo_opt_code_kb.json",),
    }

    def __init__(self, knowledge_dir: str | Path = KNOWLEDGE_DIR):
        self.knowledge_dir = Path(knowledge_dir)

    @lru_cache(maxsize=8)
    def load(self, stage: str) -> Mapping[str, Any]:
        try:
            filenames = self._FILES[stage]
        except KeyError as exc:
            raise ValueError(f"Unknown knowledge stage: {stage}") from exc
        sources: dict[str, Any] = {}
        for filename in filenames:
            path = self.knowledge_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"知识库文件不存在：{path}")
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, Mapping):
                raise ValueError(f"知识库根节点必须是对象：{path}")
            sources[path.stem] = value
        return {"knowledge_sources": sources}

    def render(self, stage: str) -> str:
        return json.dumps(self.load(stage), ensure_ascii=False, indent=2)


def build_chinese_stage_prompt(
    *,
    stage_role: str,
    template_intro: str,
    knowledge_context: str,
    constraints: str,
    examples: str,
    output_format: str,
    current_state: str = "无历史状态。",
) -> str:
    """Keep every stage prompt in the legacy role/template/KB/rules/example form."""

    return f"""# 阶段角色（stageRole）
{stage_role}

# 待填写模板介绍
{template_intro}

# 外部知识库（只读注入）
以下为项目提供的中文知识库。它用于术语、建模和代码约束的解释；不得臆造其中不存在的模型能力。
{knowledge_context}

# 约束规则
{constraints}

# 标准示例
{examples}

# 当前会话状态
{current_state}

# 输出格式
{output_format}
"""
