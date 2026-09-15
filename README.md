# LLM-BMP-Opt Open Pipeline

This standalone release provides the open BDU-to-optimization workflow:

1. Stage 1 uses the LLM to turn a user request into Basic Demand Units (BDUs), retains field states (`EMPTY`, `FILLED`, `CONFIRMED`), and asks follow-up questions until the required items are confirmed.
2. Stage 2 uses the LLM to produce a structured BMP optimization formulation and executable payload.
3. Stage 3 uses the LLM to generate the downloadable Python files; the package writer only bundles those files with its own `runtime/` directory.

Every stage follows the same Chinese prompt structure: `stageRole`, template explanation,
read-only external knowledge base, constraints, examples, current session state, and an
explicit output contract. The shipped Chinese knowledge bases live in `knowledge/`:
`geo_opt_kgb.json` (Stage 1), `geo_opt_math_kb.json` (Stage 2), and
`geo_opt_code_kb.json` (Stage 3). Each file retains the legacy primary structure and
contains its comprehensive catalog internally; it is injected at request time rather
than copied into hard-coded prompt strings.

Stage 3's knowledge base contains the registered research-area/model profiles, evaluator
and metric bindings, solver boundary, and the `placeholder_metric` contract for unsupported
models or indicators. The Chinese user guide for replacing placeholders with a real model
adapter is [docs/USER_MODEL_ADAPTATION_GUIDE_zh-CN.md](docs/USER_MODEL_ADAPTATION_GUIDE_zh-CN.md).
When Stage 2 marks a demand as unsupported, execution mode `3` still generates a clearly
labelled placeholder package and its adaptation README; execution mode `2` returns only
the formulation and support diagnosis.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set LLM_API_KEY.
.\.venv\Scripts\python -m uvicorn api_service:app --host 0.0.0.0 --port 5001
```

Call `POST /interact` with `user_input`, optional `session_id`, and `execution_mode`
(`1` = BDU dialogue, `2` = formulation, `3` = code package). Generated packages
are placed in `generated_code/<session_id>/` and can be run independently:

```powershell
cd generated_code\<session_id>
python -m pip install -r requirements.txt
python main.py
```

The `generic_lightweight` profile is a runnable software demonstration only. It does
not represent hydrologic or water-quality simulation results. A SEIMS, SWAT+, or
SWMM result requires separately acquired, licensed model executables and study data;
set the relevant local paths and validate the model configuration before use.

## Container

```powershell
Copy-Item .env.example .env
# Add LLM_API_KEY to .env.
docker compose up --build
```

The container serves the API but deliberately does not bundle external watershed
model executables or research-area data.

## Public-release boundary

The release includes public SEIMS, SWAT+, and EPA SWMM example assets. See
`DATA_MANIFEST.md` for their runtime boundary and Git LFS requirements. Do not add real
`.env` files, generated outputs, or unpublished reports. `LICENSE-REVIEW.md` must be
resolved before publication.
