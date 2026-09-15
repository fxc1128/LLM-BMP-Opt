from __future__ import annotations

import json
from typing import Any, Mapping

from langchain_core.messages import HumanMessage, SystemMessage

from src.modeling.stage_contracts import bdu_to_demand_unit_artifact, extract_json_object, validate_bdu_artifact
from src.prompting import OpenKnowledgeBase, build_chinese_stage_prompt


REQUIREMENT_FIELDS = {
    "spatial_scope": {"label": "治理区域", "hint": "请说明治理区域或研究区，例如整个流域、指定子流域或地块。"},
    "time_span": {"label": "评价时间", "hint": "请说明评价时段或降雨事件，例如多年平均、某场暴雨或某个规划期。"},
    "objectives": {"label": "治理目标", "hint": "请说明希望改善或约束的指标，例如削减泥沙、降低径流峰值或控制成本。"},
    "candidate_measures": {"label": "可选措施", "hint": "请说明可选 BMP；也可以确认采用当地常用措施。"},
    "discretization": {"label": "空间决策单元", "hint": "请说明决策单元，例如子流域、HRU、地块或坡位单元；也可以确认由模型推荐。"},
}


def _empty_template() -> dict[str, dict[str, str]]:
    return {key: {"value": "", "status": "EMPTY"} for key in REQUIREMENT_FIELDS}


class Stage1BDUParser:
    """Chinese Stage 1 parser with persistent missing-field and follow-up support."""

    def __init__(self, llm, knowledge_base: OpenKnowledgeBase | None = None):
        self.llm = llm
        self.knowledge_base = knowledge_base or OpenKnowledgeBase()

    async def parse(self, user_input: str, history: list[Any] | None = None, requirement_template: Mapping[str, Any] | None = None) -> dict[str, Any]:
        prior_template = self._normalise_template(requirement_template)
        prompt = build_chinese_stage_prompt(
            stage_role="你是流域 BMP 优化的需求分析专家（Stage 1）。将用户自然语言转换为基本需求单元 BDU，并维护可多轮确认的需求字段。不要生成数学模型、优化载荷或 Python 代码。",
            template_intro="BDU_i<Sub=利益相关方, Tar=<变量、指标、方向/阈值>, BMP_Pref=措施与决策单元, Scope=<空间范围、时间范围>>。同时填充 requirement_template；每个字段只可使用 EMPTY、FILLED、CONFIRMED 三种状态。",
            knowledge_context=self.knowledge_base.render("stage1"),
            constraints="1. 已 CONFIRMED 的字段不可改写；仅可在用户明确纠正时标记 correction_requested。\n2. spatial_scope、time_span、objectives、candidate_measures、discretization 均为进入 Stage 2 前必须确认的字段。\n3. 用户确认“常用/默认/当地 BMP”或“由模型推荐”时，允许填入该默认值并标记 CONFIRMED。\n4. 信息不足时仍输出已知 BDU，不得虚构模型、阈值、流域数据或措施效果。",
            examples="用户：在 Youwuzhen 流域以坡位单元布设当地常用 BMP，评价多年平均，减少泥沙且成本最低。\n字段：治理区域=Youwuzhen（CONFIRMED）；评价时间=多年平均（CONFIRMED）；治理目标=减少泥沙且成本最低（CONFIRMED）；可选措施=当地常用 BMP（CONFIRMED）；空间决策单元=坡位单元（CONFIRMED）。",
            current_state=json.dumps(prior_template, ensure_ascii=False, indent=2),
            output_format="仅输出一个 JSON 对象：{stage:'bdu', status:'complete|needs_clarification', bdu_list:[...], governance_checklist:[...], requirement_template:{字段:{value,status}}, user_message:'...' }。",
        )
        messages = [SystemMessage(content=prompt), *(history or [])[-6:], HumanMessage(content=user_input)]
        response = await self.llm.ainvoke(messages)
        artifact = extract_json_object(response.content)
        report = validate_bdu_artifact(artifact)
        report.raise_for_errors()
        template = self._merge_template(prior_template, artifact.get("requirement_template"), artifact)
        missing_fields = [key for key, value in template.items() if value["status"] != "CONFIRMED"]
        artifact["requirement_template"] = template
        artifact["missing_fields"] = missing_fields
        artifact["status"] = "needs_clarification" if missing_fields else "complete"
        artifact["demand_unit_artifact"] = bdu_to_demand_unit_artifact(artifact)
        if missing_fields:
            artifact["user_message"] = await self._make_followup(template, artifact)
        return artifact

    async def _make_followup(self, template: Mapping[str, Any], artifact: Mapping[str, Any]) -> str:
        missing = [REQUIREMENT_FIELDS[key] for key, value in template.items() if value["status"] != "CONFIRMED"]
        prompt = build_chinese_stage_prompt(
            stage_role="你是 Stage 1 的追问助手，帮助用户补齐流域 BMP 优化需求。",
            template_intro="根据尚未填写的字段，提出一轮简洁、通俗、可直接回答的中文追问。",
            knowledge_context=self.knowledge_base.render("stage1"),
            constraints="只追问缺失字段；不重复已确认信息；不输出 JSON、数学公式或代码；最多三句话。",
            examples="缺少评价时间和措施时：请问希望按多年平均还是一次具体降雨事件评价？可选 BMP 有指定类型吗，还是采用当地常用措施？",
            current_state=json.dumps({"missing": missing, "bdu_list": artifact.get("bdu_list", [])}, ensure_ascii=False),
            output_format="只输出面向用户的一段中文追问。",
        )
        response = await self.llm.ainvoke([SystemMessage(content=prompt)])
        text = str(response.content).strip()
        return text if text and not text.startswith("{") else "\n".join(item["hint"] for item in missing[:3])

    @staticmethod
    def _normalise_template(template: Mapping[str, Any] | None) -> dict[str, dict[str, str]]:
        normalised = _empty_template()
        if not isinstance(template, Mapping):
            return normalised
        for key in normalised:
            candidate = template.get(key)
            if isinstance(candidate, Mapping):
                value = str(candidate.get("value", "")).strip()
                status = str(candidate.get("status", "EMPTY")).upper()
                normalised[key] = {"value": value, "status": status if status in {"EMPTY", "FILLED", "CONFIRMED"} else "EMPTY"}
        return normalised

    def _merge_template(self, prior: Mapping[str, Any], proposed: Any, artifact: Mapping[str, Any]) -> dict[str, dict[str, str]]:
        merged = self._normalise_template(prior)
        proposed_template = self._normalise_template(proposed if isinstance(proposed, Mapping) else None)
        derived = self._derive_fields(artifact)
        for key, current in merged.items():
            if current["status"] == "CONFIRMED":
                continue
            candidate = proposed_template[key]
            if candidate["status"] != "EMPTY" and candidate["value"]:
                merged[key] = candidate
            elif derived.get(key):
                merged[key] = {"value": derived[key], "status": "FILLED"}
        return merged

    @staticmethod
    def _derive_fields(artifact: Mapping[str, Any]) -> dict[str, str]:
        bdu = next((item for item in artifact.get("bdu_list", []) if isinstance(item, Mapping)), {})
        scope = bdu.get("Scope", {}) if isinstance(bdu.get("Scope"), Mapping) else {}
        target = bdu.get("Tar", {}) if isinstance(bdu.get("Tar"), Mapping) else {}
        preference = bdu.get("BMP_Pref", {}) if isinstance(bdu.get("BMP_Pref"), Mapping) else {}
        measures = preference.get("measures", [])
        return {
            "spatial_scope": str(scope.get("spatial", "")).strip(),
            "time_span": str(scope.get("temporal", "")).strip(),
            "objectives": " ".join(str(target.get(key, "")) for key in ("variable", "metric", "operator")).strip(),
            "candidate_measures": "、".join(str(item) for item in measures) if isinstance(measures, list) else str(measures or ""),
            "discretization": str(preference.get("unit_hint", "")).strip(),
        }


def format_bdu_for_prompt(bdu_artifact: Mapping[str, Any]) -> str:
    return json.dumps({"bdu_list": bdu_artifact.get("bdu_list", []), "governance_checklist": bdu_artifact.get("governance_checklist", []), "requirement_template": bdu_artifact.get("requirement_template", {})}, ensure_ascii=False, indent=2)
