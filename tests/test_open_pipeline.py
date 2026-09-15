from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.codegen.template_codegen import TemplateCodeGenerator
from src.optimization_runtime.registry import EncoderRegistry, OptimizerRegistry, SimulatorRegistry

try:
    from src.modeling.stage_chain import OpenSourceStageChain
except ModuleNotFoundError:
    OpenSourceStageChain = None


class _Response:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, _messages):
        self.calls += 1
        if self.calls == 1:
            return _Response(json.dumps({
                "stage": "bdu", "status": "complete",
                "bdu_list": [{
                    "id": "BDU_1", "Sub": "watershed manager",
                    "Tar": {"variable": "implementation cost", "metric": "sum", "operator": "minimize", "threshold": None, "unit": "CNY"},
                    "BMP_Pref": {"unit_hint": "candidate unit", "measures": ["common BMPs"], "implementation_mode": "type_selection"},
                    "Scope": {"spatial": "whole watershed", "temporal": "annual"},
                    "source_text": "Minimize annual BMP implementation cost.", "uncertainties": []
                }],
                "governance_checklist": [],
                "requirement_template": {
                    "spatial_scope": {"value": "whole watershed", "status": "CONFIRMED"},
                    "time_span": {"value": "annual", "status": "CONFIRMED"},
                    "objectives": {"value": "minimize implementation cost", "status": "CONFIRMED"},
                    "candidate_measures": {"value": "common BMPs", "status": "CONFIRMED"},
                    "discretization": {"value": "candidate unit", "status": "CONFIRMED"}
                },
                "user_message": "已确认。"
            }))
        if self.calls == 2:
            return _Response(json.dumps({
            "stage": "payload", "profile_id": "generic_lightweight", "support_status": "executable",
            "problem_formulation": {"F": ["minimize implementation cost"], "C": [], "X": "candidate unit, BMP type"},
            "agent_payload": {
                "profile_id": "generic_lightweight", "force_lightweight_runtime": True,
                "basic_info": {"spatial_scope": "whole watershed", "time_span": "annual", "spatial_resolution": "candidate unit"},
                "optimization_problem": {
                    "objectives": [{"name": "implementation_cost", "type": "min", "indicator": "BMP cost", "evaluator": "bmp_net_cost_model"}],
                    "decision_variable": {"spatial_discretization": "candidate unit", "candidate_unit_count": 4, "BMP_value": {"BMP_type": [0, 1, 2]}},
                    "constraints": []
                },
                "evaluator": {"bmp_net_cost_model": {}},
                "solver": {"PopulationSize": 6, "GenerationsNum": 2, "CrossoverRate": 0.8, "MutateRate": 0.1}
            },
            "unsupported_reason": None, "user_report": "A runnable lightweight cost package was generated."
            }))
        return _Response("""```agent_payload.py
AGENT_PAYLOAD = {'profile_id': 'generic_lightweight', 'force_lightweight_runtime': True, 'optimization_problem': {'objectives': [{'name': 'implementation_cost', 'type': 'min', 'indicator': 'BMP cost', 'evaluator': 'bmp_net_cost_model'}], 'decision_variable': {'spatial_discretization': 'candidate unit', 'candidate_unit_count': 4, 'BMP_value': {'BMP_type': [0, 1, 2]}}, 'constraints': []}, 'evaluator': {'bmp_net_cost_model': {}}, 'solver': {'PopulationSize': 6, 'GenerationsNum': 2, 'CrossoverRate': 0.8, 'MutateRate': 0.1}}
```
```main.py
from agent_payload import AGENT_PAYLOAD
from runtime.factory import OptimizationFactory

def main():
    return OptimizationFactory().run(AGENT_PAYLOAD)

if __name__ == '__main__':
    main()
```
```requirements.txt
deap>=1.3.0
numpy>=1.21.0
```""")


class GeneratedPackageTest(unittest.TestCase):
    def test_execution_registries_build_a_real_decision_space(self):
        payload = {
            "optimization_problem": {"decision_variable": {"candidate_unit_count": 3, "BMP_value": {"BMP_type": [0, 1, 2]}}},
            "evaluator": {"swmm_lid": {"decision_binding": {"subcatchments": ["S1", "S2", "S3"], "actions": ["none", "bio", "permeable"]}}},
        }
        space = EncoderRegistry.create("swmm_subcatchment").build(payload)
        self.assertEqual(space.unit_ids, ["S1", "S2", "S3"])
        self.assertEqual(space.gene_to_unit[1], "S2")
        self.assertEqual(space.bmp_catalog[2], "permeable")
        self.assertEqual(SimulatorRegistry.create("swmm_lid").run([0, 1, 2], space, {}).metadata["mode"], "legacy_evaluator_executes_simulator")
        self.assertIsNotNone(OptimizerRegistry.create("nsga2", operators=None, cfg={}))

    def test_generated_package_is_independent_and_runnable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "package"
            payload = {
                "profile_id": "generic_lightweight", "force_lightweight_runtime": True,
                "optimization_problem": {
                    "objectives": [{"name": "implementation_cost", "type": "min", "indicator": "BMP cost", "evaluator": "bmp_net_cost_model"}],
                    "decision_variable": {"spatial_discretization": "candidate unit", "candidate_unit_count": 4, "BMP_value": {"BMP_type": [0, 1, 2]}},
                    "constraints": []
                },
                "evaluator": {"bmp_net_cost_model": {}},
                "solver": {"PopulationSize": 6, "GenerationsNum": 2, "CrossoverRate": 0.8, "MutateRate": 0.1}
            }
            package = TemplateCodeGenerator().render_package(payload, output_dir=package_dir)
            self.assertIn("```main.py", package.as_fenced_blocks())
            self.assertTrue((package_dir / "runtime" / "factory.py").is_file())
            self.assertNotIn("D:" + "/EGC", (package_dir / "main.py").read_text(encoding="utf-8"))
            completed = subprocess.run(
                [sys.executable, "main.py"], cwd=package_dir, capture_output=True,
                text=True, encoding="utf-8", errors="replace", check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            results = json.loads((package_dir / "results" / "pareto_front.json").read_text(encoding="utf-8"))
            self.assertEqual(len(results), 6)


@unittest.skipUnless(OpenSourceStageChain is not None, "langchain-core is required for Stage 1/Stage 2 contract test")
class OpenChainContractTest(unittest.TestCase):
    def test_fake_llm_runs_three_stages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = asyncio.run(OpenSourceStageChain(FakeLLM()).run("Minimize BMP cost annually.", output_dir=Path(temp_dir) / "package"))
            self.assertEqual(result["step"], "FINISH_STAGE3")


if __name__ == "__main__":
    unittest.main()
