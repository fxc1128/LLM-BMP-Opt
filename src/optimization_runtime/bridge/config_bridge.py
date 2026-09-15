import json
import os
import re
import sys

from ..utils.config_proxy import DictConfigProxy
from ..impl.constraints.structured_constraint import normalize_bmp_types

seims_path = os.environ.get("SEIMS_PATH", "")
if seims_path and seims_path not in sys.path:
    sys.path.insert(0, seims_path)


class ConfigBridge:
    @staticmethod
    def _construct_config_dict(json_data):
        """Convert AGENT_PAYLOAD into the legacy SEIMS scenario_analysis config."""
        input_prob = json_data["optimization_problem"]
        evaluator_block = json_data.get("evaluator", {})
        seims_cfg = (
            evaluator_block.get("seims")
            or evaluator_block.get("spatiotemporal_seims")
            or evaluator_block.get("spatial_seims")
            or evaluator_block.get("bmp_net_cost_model")
            or {}
        )

        raw_model_dir = seims_cfg.get("MODEL_DIR") or seims_cfg.get("model_dir") or r"D:\Default"
        clean_model_dir = os.path.normpath(raw_model_dir)
        db_name = seims_cfg.get("db_name", os.path.basename(clean_model_dir))
        scenario_db = seims_cfg.get("scenario_db")
        if not scenario_db:
            if db_name.endswith("_longterm_model"):
                scenario_db = db_name[: -len("_longterm_model")] + "_Scenario"
            else:
                scenario_db = db_name + "_Scenario"

        time_span = (
            seims_cfg.get("time_span")
            or json_data.get("basic_info", {}).get("time_span", "2013-2017")
        )
        start_year, end_year = time_span.split("-") if "-" in time_span else ("2013", "2017")
        temporal_scope = seims_cfg.get("temporal_scope", {})
        eval_start = temporal_scope.get("start") or f"{start_year}-01-01 00:00:00"
        eval_end = temporal_scope.get("end") or f"{end_year}-12-31 23:59:59"

        config_dict = {
            "SEIMS_Model": {
                "HOSTNAME": seims_cfg.get("HOSTNAME", "127.0.0.1"),
                "PORT": seims_cfg.get("PORT", "27017"),
                "VERSION": "OMP",
                "NTHREAD": str(seims_cfg.get("NTHREAD", seims_cfg.get("threadsNum", "4"))),
                "FDIRMTD": str(seims_cfg.get("FDIRMTD", "0")),
                "LYRMTD": str(seims_cfg.get("LYRMTD", seims_cfg.get("layeringMethod", "1"))),
                "SCENARIO_ID": str(seims_cfg.get("SCENARIO_ID", seims_cfg.get("scenarioID", "0"))),
                "Sim_Time_start": seims_cfg.get("sim_time_start", eval_start),
                "Sim_Time_end": seims_cfg.get("sim_time_end", eval_end),
                "Output_Time_start": seims_cfg.get("output_time_start", eval_start),
                "Output_Time_end": seims_cfg.get("output_time_end", eval_end),
                "MODEL_DIR": clean_model_dir,
                "BIN_DIR": seims_cfg.get("BIN_DIR", r"D:\Default\Bin"),
                "db_name": db_name,
                "SCENARIODB": scenario_db,
            }
        }

        bmp_times = input_prob["decision_variable"].get("BMP_value", {}).get("BMP_time", [])
        is_temporal = bool(bmp_times)
        runtime_years = int(end_year) - int(start_year) + 1
        implementation_period = max([runtime_years] + [int(t) for t in bmp_times]) if is_temporal else 5

        budget_wan = None
        for cons in input_prob.get("constraints", []):
            if cons.get("type", "").lower() == "budget_limitation":
                budget_wan = cons.get("budget") or cons.get("value")
                if budget_wan is None:
                    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(cons.get("mathematical_form", "")))
                    if match:
                        budget_wan = float(match.group(0))
        has_budget = budget_wan is not None

        config_dict["Scenario_Common"] = {
            "runtime_years": runtime_years,
            "eval_time_start": eval_start,
            "eval_time_end": eval_end,
            "worst_economy": 99999999.0,
            "worst_environment": 0.0,
            "enable_implementation_order": is_temporal,
            "implementation_period": implementation_period,
            "effectiveness_changeable": is_temporal,
            "change_frequency": 1,
            "change_times": implementation_period if is_temporal else 1,
            "years_first_period": implementation_period if is_temporal else 5,
            "enable_investment_quota": has_budget,
            "investment_float_range": 1.0 if has_budget else 0.2,
            "investment_each_period": [float(budget_wan) / 2.0] if has_budget else [99999999],
            "investment_aver_constrain": False,
            "discount_rate": 0.1,
            "selected_scenario_file": "",
            "export_scenario_txt": True,
            "export_scenario_tif": True,
            "prioritize_key_bmps": False,
            "pareto_front_scenarios": "[]",
        }

        spatial_unit = input_prob["decision_variable"].get("spatial_discretization", "SLPPOS")
        bmp_types = normalize_bmp_types(
            input_prob["decision_variable"].get("BMP_value", {}).get("BMP_type", [1, 2, 3, 4])
        )

        bmps_info = {
            "17": {
                "COLLECTION": "AREAL_STRUCT_MANAGEMENT",
                "SUBSCENARIO": bmp_types,
                "SLPPOS_TAG_NAME": {"1": "summit", "4": "backslope", "16": "valley"},
                "SLPPOS_GFS_NAME": {"1": "rdgInf", "4": "bksInf", "16": "vlyInf"},
            }
        }
        bmps_retain = {
            "12": {
                "COLLECTION": "PLANT_MANAGEMENT",
                "DISTRIBUTION": "RASTER|LANDUSE",
                "LOCATION": "33",
                "SUBSCENARIO": 0,
            }
        }
        unit_json_map = {
            "SLPPOS": "slppos_3cls_units_updown.json",
            "CONNFIELD": "connected_field_units_updown_15.json",
        }
        distribution_name_map = {
            "SLPPOS": "SLPPOS_UNITS",
            "CONNFIELD": "FIELDS_15",
        }
        cfg_units = {
            spatial_unit: {
                "DISTRIBUTION": f"RASTER|{distribution_name_map.get(spatial_unit, f'{spatial_unit}_UNITS')}",
                "UNITJSON": unit_json_map.get(spatial_unit, "units.json"),
            }
        }

        env_file = (
            seims_cfg.get("metric", {}).get("env_file")
            or seims_cfg.get("env_file")
            or "SED_OL_SUM.tif"
        )
        eval_info = {
            "OUTPUTID": "SED_OL",
            "ENVEVAL": env_file,
            "BASE_ENV": -9999,
            "RUN_BASE_EACH_TIME": True,
        }
        if seims_cfg.get("use_precomputed_base_env", False):
            eval_info["RUN_BASE_EACH_TIME"] = False
            eval_info["USE_PRECOMPUTED_BASE_ENV"] = True
            for key in ("BASE_ENV", "base_env", "baseline_value"):
                if key in seims_cfg:
                    eval_info["BASE_ENV"] = seims_cfg[key]
                    break
            if "BASE_SED_PERIODS" in seims_cfg:
                eval_info["BASE_SED_PERIODS"] = seims_cfg["BASE_SED_PERIODS"]

        config_dict["BMPs"] = {
            "bmps_info": json.dumps(bmps_info),
            "bmps_retain": json.dumps(bmps_retain),
            "eval_info": json.dumps(eval_info),
            "bmps_cfg_units": json.dumps(cfg_units),
            "bmps_cfg_method": "HILLSLP" if spatial_unit == "SLPPOS" else "UPDOWN",
            "bmps_cfg_units_opt": False,
        }

        solver_cfg = json_data.get("solver", {})
        config_dict["NSGA2"] = {
            "GenerationsNum": solver_cfg.get("GenerationsNum", 4),
            "PopulationSize": solver_cfg.get("PopulationSize", 12),
            "CrossoverRate": solver_cfg.get("CrossoverRate", 0.8),
            "MutateRate": solver_cfg.get("MutateRate", 0.1),
            "MaxMutatePerc": 0.2,
            "InputPopulation": False,
        }

        return config_dict, spatial_unit, is_temporal

    @staticmethod
    def create_legacy_config_object(json_data):
        if not seims_path:
            raise RuntimeError("SEIMS_PATH must point to a separately installed SEIMS Python runtime.")
        from scenario_analysis.config import SAConfig
        from scenario_analysis.spatialunits.config import SASlpPosConfig, SAConnFieldConfig

        cfg_dict, unit_type, is_temporal = ConfigBridge._construct_config_dict(json_data)
        proxy = DictConfigProxy(cfg_dict)

        if unit_type == "SLPPOS":
            raw_cfg = SASlpPosConfig(proxy)
        elif unit_type == "CONNFIELD":
            raw_cfg = SAConnFieldConfig(proxy)
        else:
            raw_cfg = SAConfig(proxy)

        try:
            raw_cfg.construct_indexes_units_gene()
        except Exception as exc:
            print(f"Warning during SAConfig init (expected when testing without data): {exc}")
        raw_cfg.fast_native_operators = bool(json_data.get("fast_native_operators", False))
        raw_cfg.fast_native_nonzero_prob = float(json_data.get("fast_native_nonzero_prob", 0.05))
        if not hasattr(raw_cfg, "key_bmps"):
            raw_cfg.key_bmps = {}
        raw_cfg.allowed_bmp_time_values = [
            int(item)
            for item in json_data.get("optimization_problem", {})
            .get("decision_variable", {})
            .get("BMP_value", {})
            .get("BMP_time", [])
        ]

        return raw_cfg, is_temporal
