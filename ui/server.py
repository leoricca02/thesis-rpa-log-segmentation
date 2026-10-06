"""FastAPI backend for the demo UI.

Run from the repository root:

    python ui/server.py            # then open http://127.0.0.1:8000

The server only orchestrates; every phase is the unmodified code in ``src/``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

_UI_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _UI_DIR.parent
for _path in (_REPO_ROOT / "src", _UI_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dotenv import load_dotenv  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from instrumented_client import install_stdout_router  # noqa: E402
from pipeline_runner import (  # noqa: E402
    THESIS_CACHE,
    LiveRun,
    ReplayRun,
    Run,
    describe_raw_log,
    list_recordings,
    load_recording,
    read_raw_log,
    save_recording,
)

load_dotenv(_REPO_ROOT / ".env")
install_stdout_router()

DATA_DIR = _REPO_ROOT / "data"
UPLOAD_DIR = _UI_DIR / "uploads"
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

app = FastAPI(title="RPA Log Segmentation — demo UI")
app.mount("/static", StaticFiles(directory=_UI_DIR / "static"), name="static")
RUNS: dict[str, Run] = {}


# --------------------------------------------------------------------- models
class Routine(BaseModel):
    routine_name: str = Field(min_length=1, max_length=80)
    executions: int = Field(ge=1, le=50)


class RunRequest(BaseModel):
    log_id: str
    routines: list[Routine] = Field(min_length=1, max_length=8)
    model: str = DEFAULT_MODEL
    review: bool = True
    relevance_filter: bool = True
    include_thoughts: bool = True
    use_cache: bool = True


class ReviewDecision(BaseModel):
    action: str
    topology: list[dict[str, Any]] = []


class SaveRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class ReplayRequest(BaseModel):
    recording_id: str
    speed: float = 1.0


# -------------------------------------------------------------------- helpers
def _resolve_log(log_id: str) -> tuple[Path, str]:
    """Map a log id ('sample:<relpath>' or 'upload:<file>') to a safe path."""
    kind, _, ref = log_id.partition(":")
    base = {"sample": DATA_DIR, "upload": UPLOAD_DIR}.get(kind)
    if base is None or not ref:
        raise HTTPException(400, "Unknown log id.")
    path = (base / ref).resolve()
    if base.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404, "Log not found.")
    name = path.name.split("__", 1)[-1] if kind == "upload" else path.name
    return path, name


def _preview(log_id: str) -> dict[str, Any]:
    path, name = _resolve_log(log_id)
    try:
        info = describe_raw_log(read_raw_log(path))
    except (ValueError, OSError) as exc:
        raise HTTPException(422, f"Could not read the log: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - pandas parser errors vary
        raise HTTPException(422, f"Could not parse the log: {exc}") from exc
    return {"log_id": log_id, "name": name, **info}


def _get_run(run_id: str) -> Run:
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "Run not found.")
    return run


# ----------------------------------------------------------------------- API
@app.get("/")
def index() -> FileResponse:
    return FileResponse(_UI_DIR / "static" / "index.html")


@app.get("/api/status")
def status() -> dict[str, Any]:
    return {
        "api_key": bool(os.getenv("GEMINI_API_KEY")),
        "model": DEFAULT_MODEL,
        "thesis_cache": THESIS_CACHE.exists(),
    }


@app.get("/api/samples")
def samples() -> list[dict[str, Any]]:
    items = []
    for path in sorted(DATA_DIR.rglob("*.csv")):
        rel = path.relative_to(DATA_DIR).as_posix()
        items.append(
            {
                "log_id": f"sample:{rel}",
                "name": path.name,
                "group": rel.split("/", 1)[0].replace("_", " "),
            }
        )
    items.sort(key=lambda i: (i["group"] != "case study", i["name"]))
    return items


@app.get("/api/logs/preview")
def preview(log_id: str) -> dict[str, Any]:
    return _preview(log_id)


@app.post("/api/logs")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file.")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(file.filename).name)
    stored = f"{uuid.uuid4().hex[:8]}__{safe}"
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(413, "The log is larger than 20 MB.")
    (UPLOAD_DIR / stored).write_bytes(content)
    return _preview(f"upload:{stored}")


@app.post("/api/runs")
def start_run(request: RunRequest) -> dict[str, str]:
    path, name = _resolve_log(request.log_id)
    names = [r.routine_name.strip().lower() for r in request.routines]
    if len(set(names)) != len(names) or not all(names):
        raise HTTPException(400, "Routine names must be non-empty and distinct.")
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            400,
            "GEMINI_API_KEY is not set. Add it to .env for a live run, "
            "or pick a recording to replay.",
        )
    run = LiveRun(
        csv_path=str(path),
        log_name=name,
        routines=[r.model_dump() for r in request.routines],
        model=request.model.strip() or DEFAULT_MODEL,
        review=request.review,
        relevance_filter=request.relevance_filter,
        include_thoughts=request.include_thoughts,
        use_cache=request.use_cache,
    )
    RUNS[run.id] = run
    run.start()
    return {"run_id": run.id}


@app.get("/api/runs/{run_id}/stream")
async def stream(run_id: str) -> StreamingResponse:
    run = _get_run(run_id)

    async def events():
        sent = 0
        idle = 0
        while True:
            batch = run.events[sent:]
            for event in batch:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            sent += len(batch)
            if run.finished and sent >= len(run.events):
                yield "event: end\ndata: {}\n\n"
                return
            idle = 0 if batch else idle + 1
            if idle and idle % 150 == 0:
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/runs/{run_id}/review")
def review(run_id: str, decision: ReviewDecision) -> dict[str, bool]:
    run = _get_run(run_id)
    if decision.action == "abort":
        run.cancel()
    else:
        run.submit_review(decision.model_dump())
    return {"ok": True}


@app.post("/api/runs/{run_id}/cancel")
def cancel(run_id: str) -> dict[str, bool]:
    _get_run(run_id).cancel()
    return {"ok": True}


@app.post("/api/runs/{run_id}/save")
def save(run_id: str, request: SaveRequest) -> dict[str, str]:
    run = _get_run(run_id)
    if not isinstance(run, LiveRun) or not run.finished:
        raise HTTPException(400, "Only a finished live run can be saved.")
    return {"recording_id": save_recording(run, request.name)}


@app.get("/api/runs/{run_id}/files/{name}")
def download(run_id: str, name: str) -> FileResponse:
    path = _get_run(run_id).file_path(name)
    if path is None:
        raise HTTPException(404, "File not available.")
    return FileResponse(path, filename=name)


@app.get("/api/recordings")
def recordings() -> list[dict[str, Any]]:
    return list_recordings()


@app.get("/api/recordings/{rec_id}")
def recording(rec_id: str) -> dict[str, Any]:
    try:
        events = load_recording(rec_id)
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(404, "Recording not found.") from exc
    start = next((e for e in events if e["type"] == "run_start"), {})
    raw = next((e for e in events if e["type"] == "raw_log"), {})
    meta = next((m for m in list_recordings() if m["id"] == rec_id), {})
    return {
        "meta": meta,
        "run_start": start,
        "preview": {"log_id": None, "name": start.get("log_name", ""),
                    **raw.get("data", {})},
    }


@app.post("/api/replays")
def replay(request: ReplayRequest) -> dict[str, str]:
    try:
        run = ReplayRun(request.recording_id, request.speed)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Recording not found.") from exc
    RUNS[run.id] = run
    run.start()
    return {"run_id": run.id}


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Demo UI for the thesis pipeline.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"\n  RPA Log Segmentation UI  ->  http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
