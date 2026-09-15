"""
SurrogateEvaluator: sklearn MLPRegressor 代理模型的环境效益评估器适配器。

用途:
  当不需要运行完整 SEIMS 过程模型时（如调试、快速原型、长周期训练），
  使用已训练的代理模型预测泥沙削减率，作为 SEIMSEvaluator 的替代。

  成本计算仍使用真实的 CostEvaluator（SUScenario.calculate_economy()），
  与本评估器组合使用即可获得完整的双目标评估能力。

接入方式:
  AGENT_PAYLOAD 的 objectives 中，将 sediment_reduction 的 evaluator
  指定为 "surrogate" 或 "sklearn_surrogate"，并在 evaluator 配置块中提供：
    {
      "surrogate": {
        "model_dir": "D:/path/to/surrogate_training_xxx/models",
        "baseline_genes": null   // 可选，默认 [0]*n_genes 表示无措施基准
      }
    }
"""
from __future__ import annotations

import json
import os
import pickle
import warnings
from typing import List, Optional

import numpy as np

from ...core.interfaces import AbstractEvaluator

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)


class SurrogateEvaluator(AbstractEvaluator):
    """
    基于 sklearn pkl 代理模型的环境效益评估器。

    返回值: 泥沙削减率 = (baseline_SED - pred_SED) / baseline_SED  (0~1浮点数)
    目标方向: MAXIMIZE（值越大越好，与 SEIMSEvaluator 一致）
    """

    def __init__(self, raw_cfg=None, is_temporal: bool = False,
                 model_dir: Optional[str] = None,
                 context: Optional[dict] = None, **kwargs):
        """
        Parameters
        ----------
        raw_cfg:     SASlpPosConfig（SEIMS路径）或 None（轻量路径）
        is_temporal: 是否为时空优化模式
        model_dir:   代理模型目录，包含 surrogate_model.pkl 等文件。
                     若为 None，尝试从 raw_cfg 或 context["_surrogate_dir"] 推导。
        context:     dict context（轻量路径使用，用于推导 model_dir / genes_num）
        """
        self.raw_cfg = raw_cfg
        self.is_temporal = is_temporal
        # 轻量路径：model_dir 也可从 context 中取
        if model_dir is None and isinstance(context, dict):
            model_dir = context.get("_surrogate_dir", None)

        # ── 确定模型目录 ─────────────────────────────────────────────────────
        if model_dir is None:
            model_dir = self._infer_model_dir(raw_cfg)
        if model_dir is None:
            raise ValueError(
                "[SurrogateEvaluator] model_dir 未提供，也无法从 raw_cfg 推导。\n"
                "请在 AGENT_PAYLOAD 的 evaluator.surrogate.model_dir 中指定代理模型目录。"
            )
        self.model_dir = model_dir

        # ── 加载模型、缩放器、元数据 ─────────────────────────────────────────
        self._model = None
        self._scaler_X = None
        self._scaler_y = None
        self._metadata: dict = {}
        self._n_input: int = 0
        self._output_vars: List[str] = []

        self._load_artifacts()

        # ── 基准 SED 延迟计算（首次 evaluate 时计算）─────────────────────────
        self._baseline_sed: Optional[float] = None

    # ──────────────────────────────────────────────────────────────────────────
    # 私有辅助方法
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _infer_model_dir(raw_cfg) -> Optional[str]:
        """从 raw_cfg 推导模型目录（约定：surrogate_model_dir 属性或 eval_info 字段）。"""
        # 方式1: 配置对象有 surrogate_model_dir 属性
        if hasattr(raw_cfg, 'surrogate_model_dir') and raw_cfg.surrogate_model_dir:
            return raw_cfg.surrogate_model_dir
        # 方式2: eval_info 字典中的 SURROGATE_DIR 键
        eval_info = getattr(raw_cfg, 'eval_info', {}) or {}
        if isinstance(eval_info, str):
            try:
                eval_info = json.loads(eval_info)
            except Exception:
                eval_info = {}
        return eval_info.get('SURROGATE_DIR', None)

    def _load_artifacts(self):
        """加载 pkl 模型文件和缩放器。"""
        model_path = os.path.join(self.model_dir, "surrogate_model.pkl")
        scaler_x_path = os.path.join(self.model_dir, "scaler_X.pkl")
        scaler_y_path = os.path.join(self.model_dir, "scaler_y.pkl")
        meta_path = os.path.join(self.model_dir, "model_metadata.json")

        for p, label in [(model_path, "surrogate_model.pkl"),
                         (scaler_x_path, "scaler_X.pkl"),
                         (scaler_y_path, "scaler_y.pkl")]:
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"[SurrogateEvaluator] 缺少文件: {p}\n"
                    f"  请确认 model_dir='{self.model_dir}' 下包含 {label}"
                )

        with open(model_path, "rb") as f:
            self._model = pickle.load(f)

        if not _JOBLIB_AVAILABLE:
            raise ImportError(
                "[SurrogateEvaluator] 需要 joblib 库，请运行: pip install joblib"
            )
        self._scaler_X = joblib.load(scaler_x_path)
        self._scaler_y = joblib.load(scaler_y_path)

        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
        genes_num_fallback = (getattr(self.raw_cfg, 'genes_num', 0)
                              if self.raw_cfg is not None else 0)
        self._n_input = self._metadata.get("input_dim", genes_num_fallback)
        self._output_vars = self._metadata.get("output_vars", ["SED"])

        r2 = (self._metadata.get("performance", {})
                             .get("SED", {})
                             .get("r2", float("nan")))
        print(f"[SurrogateEvaluator] loaded | "
              f"model_type={self._metadata.get('model_type', 'unknown')} | "
              f"R2(SED)={r2:.4f} | input_dim={self._n_input}")

    def _predict_sed(self, genes: List[int]) -> float:
        """对单个基因序列预测 SED (kg/a)。"""
        X = np.array(genes, dtype=float).reshape(1, -1)
        X_s = self._scaler_X.transform(X)
        y_n = self._model.predict(X_s)
        y_pred = self._scaler_y.inverse_transform(y_n.reshape(-1, 1))
        sed_idx = (self._output_vars.index("SED")
                   if "SED" in self._output_vars else 0)
        return float(y_pred[0][sed_idx])

    def _get_baseline_sed(self) -> float:
        """惰性计算基准 SED（全0基因 = 无措施方案）。"""
        if self._baseline_sed is None:
            genes_num_fallback = (getattr(self.raw_cfg, 'genes_num', 105)
                                  if self.raw_cfg is not None else 105)
            n = self._n_input if self._n_input > 0 else genes_num_fallback
            self._baseline_sed = self._predict_sed([0] * n)
            print(f"[SurrogateEvaluator] baseline SED (no-BMP) = {self._baseline_sed:.2f} kg/a")
        return self._baseline_sed

    # ──────────────────────────────────────────────────────────────────────────
    # AbstractEvaluator 接口
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate(self, individual, context) -> float:
        """
        返回泥沙削减率（越大越好，范围通常 0~1）。
        与 SEIMSEvaluator 返回值语义一致，可直接替换。
        """
        try:
            genes = list(individual)
            baseline = self._get_baseline_sed()
            if baseline <= 0:
                return 0.0
            pred_sed = self._predict_sed(genes)
            return (baseline - pred_sed) / baseline
        except Exception as e:
            print(f"[SurrogateEvaluator] evaluate 出错: {e}")
            return 0.0
