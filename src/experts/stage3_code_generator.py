from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from langchain_core.messages import HumanMessage, SystemMessage

from src.codegen.template_codegen import CodePackage, TemplateCodeGenerator
from src.prompting import OpenKnowledgeBase, build_chinese_stage_prompt


_FILE_BLOCK = re.compile(r"```(?P<name>[A-Za-z0-9_.-]+)\s*\n(?P<content>.*?)```", re.DOTALL)
_REQUIRED_FILES = {"agent_payload.py", "main.py", "requirements.txt"}
_EXTENSION_FILES = {"encoder.py", "simulator.py", "evaluators.py", "constraints.py", "README.md"}
_ALLOWED_FILES = _REQUIRED_FILES | _EXTENSION_FILES


class Stage3LLMCodeGenerator:
    """Ask the LLM for a constrained, self-contained public code package."""

    def __init__(self, llm, package_writer: TemplateCodeGenerator | None = None, knowledge_base: OpenKnowledgeBase | None = None):
        self.llm = llm
        self.package_writer = package_writer or TemplateCodeGenerator()
        self.knowledge_base = knowledge_base or OpenKnowledgeBase()

    async def generate(self, stage2_output: Mapping[str, Any], output_dir: str | Path | None = None) -> CodePackage:
        payload = stage2_output.get("agent_payload")
        if not isinstance(payload, Mapping):
            raise ValueError("Stage 3 requires a valid agent_payload from Stage 2.")
        requires_extension_skeletons = stage2_output.get("support_status") not in {"executable", "requires_local_paths"} or self._uses_placeholder(payload)
        prompt = build_chinese_stage_prompt(
            stage_role="你是流域 BMP 优化代码生成专家（Stage 3）。根据已确认的 AGENT_PAYLOAD 生成可下载、可审查的 Python 代码包。代码内容必须由你生成，不得把模板渲染器当作代码生成结果。",
            template_intro="代码包固定包含 agent_payload.py、main.py、requirements.txt。未支持需求必须额外包含 encoder.py、simulator.py、evaluators.py、constraints.py、README.md，作为用户接入真实研究区模型的骨架。运行时公共库将在打包时作为 runtime/ 目录附带。main.py 必须从 runtime.factory 导入 OptimizationFactory，并以 agent_payload.py 中的 AGENT_PAYLOAD 为输入。",
            knowledge_context=self.knowledge_base.render("stage3"),
            constraints="1. 只生成允许的相对文件名，禁止绝对路径、目录穿越、二进制文件或网络下载代码。\n2. 不得改变 Stage 2 给出的目标、约束、变量、profile_id 和评价器语义。\n3. main.py 必须提供 main()，在运行 Factory 前注册 MyEncoder、MySimulator、目标评价器和约束评价器；不得依赖本项目根目录。\n4. requirements.txt 只列 Python 依赖；外部流域可执行文件与数据路径必须在 README.md 说明，而非伪装成 pip 包。\n5. 使用 placeholder_metric 或 support_status=unsupported 时，必须生成 encoder.py、simulator.py、evaluators.py、constraints.py、README.md。README 首段明确“占位评估，不是科学模拟”，并列出 reason、单位、value_range、adapter_todo。\n6. encoder.py 的 MyEncoder.build 必须返回 DecisionSpace；simulator.py 的 MySimulator.run 必须返回 SimulationResult；其他骨架应保留清晰 TODO。\n7. 所有 Python 代码应可通过 py_compile；输出只包含文件代码块，不加解释文字。",
            examples="```agent_payload.py\nAGENT_PAYLOAD = {...}\n```\n\n```main.py\nfrom agent_payload import AGENT_PAYLOAD\nfrom runtime.factory import OptimizationFactory\n\ndef main():\n    return OptimizationFactory().run(AGENT_PAYLOAD)\n\nif __name__ == '__main__':\n    main()\n```\n\n```requirements.txt\ndeap>=1.3.0\nnumpy>=1.21.0\n```",
            current_state=str({"profile_id": stage2_output.get("profile_id"), "agent_payload": payload}),
            output_format="严格输出带文件名的 fenced code blocks，例如 ```main.py ... ```。必须恰好包含 agent_payload.py、main.py、requirements.txt。",
        )
        response = await self.llm.ainvoke([SystemMessage(content=prompt), HumanMessage(content="请生成 Stage 3 代码包。")])
        contents = self.parse_file_blocks(str(response.content), require_extension_skeletons=requires_extension_skeletons)
        return self.package_writer.write_llm_package(contents, output_dir=output_dir)

    @staticmethod
    def parse_file_blocks(text: str, require_extension_skeletons: bool = False) -> dict[str, str]:
        files: dict[str, str] = {}
        for match in _FILE_BLOCK.finditer(text):
            name = match.group("name")
            if name not in _ALLOWED_FILES:
                raise ValueError(f"Stage 3 returned a forbidden file name: {name}")
            if name in files:
                raise ValueError(f"Stage 3 returned duplicate file: {name}")
            files[name] = match.group("content").strip() + "\n"
        missing = _REQUIRED_FILES.difference(files)
        if missing:
            raise ValueError(f"Stage 3 response missed required files: {', '.join(sorted(missing))}")
        if require_extension_skeletons:
            missing_extensions = _EXTENSION_FILES.difference(files)
            if missing_extensions:
                raise ValueError(f"Stage 3 placeholder package missed extension files: {', '.join(sorted(missing_extensions))}")
        for name, content in files.items():
            if name.endswith(".py"):
                try:
                    compile(content, name, "exec")
                except SyntaxError as exc:
                    raise ValueError(f"Stage 3 returned invalid Python in {name}: {exc.msg}") from exc
        return files

    @staticmethod
    def _uses_placeholder(payload: Mapping[str, Any]) -> bool:
        problem = payload.get("optimization_problem", {})
        objectives = problem.get("objectives", []) if isinstance(problem, Mapping) else []
        if not isinstance(objectives, list):
            return False
        return any(isinstance(item, Mapping) and str(item.get("evaluator", "")).lower() in {"placeholder", "placeholder_metric", "custom_needed"} for item in objectives)
