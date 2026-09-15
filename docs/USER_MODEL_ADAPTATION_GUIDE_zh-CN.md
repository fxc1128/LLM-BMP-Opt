# 用户模型接入指南

本指南面向需要把自己的流域模型接入下载代码包的用户。最重要的一点是：不能只修改评价器。优化器生成的 `individual` 必须先有真实的空间含义，模型也必须知道如何把它转换为输入文件。

## 运行链路

```text
Encoder 建立空间决策域
  -> Optimizer 生成 individual
  -> 预模拟约束检查
  -> Simulator 写入模型并运行
  -> ObjectiveEvaluator 计算目标值
  -> 后模拟约束检查
  -> Optimizer 交叉、变异和选择
```

`individual[i]` 是第 `i` 个空间单元的决策值。例如 `[0, 2, 1]` 表示三个单元分别采用编码为 0、2、1 的措施。编码的含义、单元顺序和模型输入位置必须由用户明确提供。

## 需要配置的文件

Stage 3 对未支持需求会生成：

```text
agent_payload.py   需求、模型路径、变量域和求解参数
encoder.py         空间单元和基因编码
simulator.py       模型输入写入、执行和原始输出
evaluators.py      目标值公式
constraints.py     预模拟/后模拟约束
main.py            注册组件并启动优化
README.md          数据、单位、模型版本和运行说明
requirements.txt
runtime/           公共运行时
```

## 1. 编写 Encoder

Encoder 的输入是 `agent_payload`，输出是 `DecisionSpace`。它至少要提供真实单元 ID、稳定顺序、每个基因允许的值域和 BMP 编码表。

```python
# encoder.py
from runtime.core.interfaces import AbstractEncoder, DecisionSpace


class MyEncoder(AbstractEncoder):
    def build(self, agent_payload):
        unit_ids = ["HRU_001", "HRU_002", "HRU_003"]
        bmp_catalog = {0: "不实施", 1: "等高耕作", 2: "缓冲带"}
        return DecisionSpace(
            unit_ids=unit_ids,
            gene_domains=[[0, 1, 2] for _ in unit_ids],
            gene_to_unit={i: unit_id for i, unit_id in enumerate(unit_ids)},
            unit_to_gene={unit_id: i for i, unit_id in enumerate(unit_ids)},
            bmp_catalog=bmp_catalog,
            unit_metadata={unit_id: {} for unit_id in unit_ids},
        )
```

实际项目中，`unit_ids` 应从 HRU、坡位、地块、子汇水区或设施数据读取，而不是手写。空间单元的排序必须固定，否则同一个 `individual` 在不同运行中会代表不同方案。

## 2. 编写 Simulator

Simulator 接收 `individual` 和 `DecisionSpace`，复制模型模板，将决策写入副本，运行模型，然后返回 `SimulationResult`。它不负责决定目标方向，也不应把一个模型指标直接伪装成另一个指标。

```python
# simulator.py
from pathlib import Path
import subprocess
from runtime.core.interfaces import AbstractSimulator, SimulationResult


class MySimulator(AbstractSimulator):
    def run(self, individual, decision_space, context):
        run_dir = Path(context["work_root"]) / context["scenario_id"]
        # 1. 复制 template_dir 到 run_dir
        # 2. 对每个 index：unit_id = decision_space.gene_to_unit[index]
        # 3. 用 decision_space.bmp_catalog[individual[index]] 修改模型输入
        # 4. 执行模型并检查 returncode
        subprocess.run([context["executable"]], cwd=run_dir, check=True, timeout=600)
        return SimulationResult(
            outputs={"output_dir": str(run_dir), "raw_output": str(run_dir / "output.csv")},
            work_dir=str(run_dir),
            metadata={"model": context.get("model_name"), "unit_count": len(decision_space.unit_ids)},
        )
```

模型输入文件、可执行文件、输出文件和参数应通过 `agent_payload['simulator']` 配置。原始模板不能被直接覆盖，每个候选解应使用独立情景目录。

## 3. 编写目标评价器

目标评价器只从 `SimulationResult` 计算数值。总量、年均、峰值、削减量和削减率可以复用公共公式；只有模型输出解析是模型专用的。

```python
# evaluators.py
from runtime.core.interfaces import AbstractObjectiveEvaluator


class SedimentReduction(AbstractObjectiveEvaluator):
    def evaluate_result(self, result, individual, decision_space):
        baseline = float(result.outputs["baseline_sediment"])
        scenario = float(result.outputs["scenario_sediment"])
        return (baseline - scenario) / baseline
```

返回值必须是一个 `float`，并且单位、方向和聚合方式要在 `agent_payload.py` 中写明。

## 4. 编写约束评价器

无需模型输出的约束应在模拟前检查，例如禁配区域、每个单元只能选一种措施、上下游不能同时配置、最小/最大实施面积。需要模型输出的约束在模拟后检查，例如成本上限、断面浓度和洪峰阈值。

```python
# constraints.py
from runtime.core.interfaces import AbstractConstraintEvaluator, ConstraintMode


class UpstreamDownstreamRule(AbstractConstraintEvaluator):
    def validate_result(self, individual, decision_space, result, mode):
        if mode != ConstraintMode.PRE_EVALUATION:
            return True
        upstream = decision_space.spatial_relations.get("upstream", {})
        for unit_id, downstream_ids in upstream.items():
            i = decision_space.unit_to_gene[unit_id]
            if individual[i] == 2 and any(individual[decision_space.unit_to_gene[x]] == 2 for x in downstream_ids):
                return False
        return True
```

## 5. 在 main.py 注册

```python
from agent_payload import AGENT_PAYLOAD
from encoder import MyEncoder
from simulator import MySimulator
from evaluators import SedimentReduction
from constraints import UpstreamDownstreamRule
from runtime.registry import EncoderRegistry, SimulatorRegistry, EvaluatorRegistry

EncoderRegistry.register("my_encoder", MyEncoder)
SimulatorRegistry.register("my_simulator", MySimulator)
EvaluatorRegistry.register("sediment_reduction", SedimentReduction)

AGENT_PAYLOAD["encoder"] = {"type": "my_encoder"}
AGENT_PAYLOAD["simulator"] = {"type": "my_simulator"}
AGENT_PAYLOAD["optimization_problem"]["objectives"][0]["evaluator"] = "sediment_reduction"
```

约束类也必须由 Factory 的约束配置引用；若使用自定义约束代码，应明确它是在模拟前还是模拟后执行。

## Placeholder 的正确用途

`placeholder_metric` 只表示当前没有真实模型接口。它可以用于检查代码包和优化流程，但返回值不是水文、水质、生态或经济模拟结果。完成 Encoder、Simulator、输出解析、基线验证和 BMP 情景验证后，才能把 placeholder 替换为真实组件。

## 验证顺序

先用一个已知单元和两个 BMP 做单元映射检查，再运行零措施基线；随后运行一个已知措施情景，确认模型输入确实变化、输出单位正确。最后用 `PopulationSize=2`、`GenerationsNum=1` 做冒烟测试，再恢复正式参数，并保存模型版本、数据版本、配置、日志和失败情景。
