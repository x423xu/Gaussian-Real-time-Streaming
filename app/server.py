from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.engine import PlaybackEngine, REPO_ROOT


class StartRequest(BaseModel):
    scene: str = "5aca87f95a9412c6"
    checkpoint_path: str | None = None
    num_frames: int | None = None
    gap: int = 4
    stride: int | None = None
    max_snapshots: int | None = None


class RenderRequest(BaseModel):
    index: int = 0
    camera: dict[str, float] | None = None
    append: bool = False
    append_window: int = 5


app = FastAPI(title="RTGS Gaussian Playback")
engine = PlaybackEngine()
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/status")
def status() -> dict[str, Any]:
    return engine.manifest()


@app.post("/api/start")
def start(request: StartRequest) -> dict[str, Any]:
    return engine.start(
        scene=request.scene,
        checkpoint_path=request.checkpoint_path,
        num_frames=request.num_frames,
        gap=request.gap,
        stride=request.stride,
        max_snapshots=request.max_snapshots,
    )


@app.post("/api/render")
def render(request: RenderRequest) -> dict[str, Any]:
    try:
        if request.append:
            return engine.render_append(request.index, request.append_window, request.camera)
        return engine.render(request.index, request.camera)
    except IndexError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/output/{path:path}")
def output_file(path: str) -> FileResponse:
    output_root = (REPO_ROOT / "app" / "outputs").resolve()
    target = (output_root / path).resolve()
    if output_root not in target.parents and target != output_root:
        raise FileNotFoundError(path)
    return FileResponse(target)
