from __future__ import annotations

import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.open_pipeline import OpenPipeline


app = FastAPI(title="LLM-BMP-Opt Open Pipeline", version="0.1.0")
pipeline: OpenPipeline | None = None


class InteractionRequest(BaseModel):
    user_input: str = Field(min_length=1)
    session_id: str | None = None
    execution_mode: Literal[1, 2, 3] = 3


def get_pipeline() -> OpenPipeline:
    global pipeline
    if pipeline is None:
        pipeline = OpenPipeline(generated_root=os.getenv("GENERATED_CODE_DIR", "generated_code"))
    return pipeline


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "llm-bmp-opt-open"}


@app.post("/interact")
async def interact(request: InteractionRequest) -> dict:
    try:
        return await get_pipeline().interact(**request.model_dump())
    except (RuntimeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
