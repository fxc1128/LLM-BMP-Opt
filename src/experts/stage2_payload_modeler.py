from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.experts.stage1_bdu_parser import format_bdu_for_prompt
from src.modeling.profile_registry import ProfileRegistry, merge_profile_defaults
from src.modeling.stage_contracts import extract_json_object, validate_payload_artifact
from src.prompting import OpenKnowledgeBase, build_chinese_stage_prompt


class Stage2PayloadModeler:
    """Stage 2 modeler that lets the LLM generate the payload contract."""

    def __init__(self, llm, profile_registry: ProfileRegistry | None = None, knowledge_base: OpenKnowledgeBase | None = None):
        self.llm = llm
        self.profile_registry = profile_registry or ProfileRegistry()
        self.knowledge_base = knowledge_base or OpenKnowledgeBase()

    async def build_payload(self, bdu_artifact: dict[str, Any]) -> dict[str, Any]:
        system_content = build_chinese_stage_prompt(
            stage_role="你是流域 BMP 优化建模专家（Stage 2）。只根据已确认的 BDU，将需求形式化为 Table S3 风格的 F、C、X，并生成运行时可验证的 AGENT_PAYLOAD。",
            template_intro="F 是目标列表（指标、方向、评价模型、空间和时间）；C 是约束；X 是空间单元上的 BMP 决策变量。AGENT_PAYLOAD 必须显式写出 evaluator、solver、objectives、decision_variable 与 constraints。",
            knowledge_context=self.knowledge_base.render("stage2"),
            constraints="1. 只能从下方运行时 profile 注册表选择 profile_id。\n2. 无适配 profile 时 support_status=unsupported；仍须生成 agent_payload，所有未支持目标使用 placeholder_metric，并逐项给出 placeholder=true、reason、unit、value_range、adapter_todo。绝不可捏造真实模型结果。\n3. 不重做 Stage 1 解释；BDU 是唯一的需求事实来源。\n4. 所有目标、约束、决策变量、评价器和求解器参数必须在 JSON 中显式给出。\n5. 保持中文用户报告，数学符号可使用通用英文缩写。\n\n运行时 profile 注册表：\n" + self.profile_registry.prompt_context(),
            examples="示例：F=['最小化总成本', '最大化泥沙削减率']；C=['总成本 <= 预算']；X='坡位单元上的 BMP 类型或面积比例'。",
            current_state=format_bdu_for_prompt(bdu_artifact),
            output_format="仅输出一个 JSON 对象：{stage:'payload', profile_id:'...', support_status:'executable|unsupported|requires_local_paths', problem_formulation:{F:[],C:[],X:'...'}, agent_payload:{...}, unsupported_reason:null, user_report:'中文说明'}。",
        )
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=format_bdu_for_prompt(bdu_artifact)),
        ]
        output = None
        for attempt in range(3):
            response = await self.llm.ainvoke(messages)
            output = extract_json_object(response.content)
            output = self._merge_declared_profile_defaults(output)
            output["payload_validation"] = validate_payload_artifact(output)
            if output["payload_validation"]["ok"]:
                return output
            if attempt < 2:
                messages.extend([
                    AIMessage(content=str(response.content)),
                    HumanMessage(content=(
                        "上一个 payload 未通过结构校验。请只修正 JSON 结构并保留需求语义。\n"
                        + output["payload_validation"]["feedback"]
                    )),
                ])
        return output or {
            "payload_validation": {
                "ok": False,
                "errors": [{"path": "$", "message": "Stage 2 returned no payload.", "allowed": None}],
                "warnings": [],
                "feedback": "Stage 2 returned no payload.",
            }
        }

    def _merge_declared_profile_defaults(self, output: dict[str, Any]) -> dict[str, Any]:
        profile_id = output.get("profile_id")
        if not profile_id:
            payload = output.get("agent_payload")
            if isinstance(payload, dict):
                payload.setdefault("support_status", output.get("support_status", "unsupported"))
            return output
        profile = self.profile_registry.get(str(profile_id))
        payload = output.get("agent_payload")
        if isinstance(payload, dict):
            output["agent_payload"] = merge_profile_defaults(payload, profile)
        output.setdefault("support_status", profile.support_status)
        if isinstance(output.get("agent_payload"), dict):
            output["agent_payload"].setdefault("support_status", output["support_status"])
        return output


def format_stage2_for_user(stage2_output: dict[str, Any]) -> str:
    formulation = stage2_output.get("problem_formulation", {})
    return json.dumps(
        {
            "support_status": stage2_output.get("support_status"),
            "profile_id": stage2_output.get("profile_id"),
            "F": formulation.get("F", []),
            "C": formulation.get("C", []),
            "X": formulation.get("X"),
            "unsupported_reason": stage2_output.get("unsupported_reason"),
        },
        ensure_ascii=False,
        indent=2,
    )
