"""Live and replayed pipeline runs, exposed to the UI as an event stream.

A ``LiveRun`` executes the real thesis pipeline (``src/``) on a background
thread and turns what it observes into structured events: the phase timeline
(parsed from the phase headers the pipeline prints), every LLM call (from
``InstrumentedClient``), and the data each phase produced. Every event is also
appended to ``events.jsonl`` in the run folder, which is what makes a run
replayable.

A ``ReplayRun`` re-emits a recorded ``events.jsonl`` with its original pacing
(scaled by a speed factor), so a demo never depends on network, quota or the
stochasticity of Phase 2.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from data_pipeline import (
    _ANCHOR_COLUMNS,
    _FALLBACK_IGNORE_COLUMNS,
    load_and_serialize_smartrpa,
    tag_irrelevant_nodes,
)
from phase1_boundary_reasoning import infer_sharing_topology
from phase2_execution_mapping import (
    _check_declared_counts,
    generate_segmented_log,
)

from instrumented_client import InstrumentedClient, install_stdout_router

REPO_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = Path(__file__).resolve().parent
RUNS_DIR = UI_DIR / "runs"
RECORDINGS_DIR = UI_DIR / "recordings"
CACHE_DIR = UI_DIR / ".cache"
THESIS_CACHE = REPO_ROOT / "results" / "case_study" / "gemini_cache.json"
OUTPUT_XLSX = "Final_Segmented_Master_Log.xlsx"
OUTPUT_JSON = "Final_Segmented_Master_Log_routing.json"

# Phase headers printed by the pipeline modules, in pipeline order.
_PHASE_HEADERS = (
    ("--- Phase 0A-2", "0A2"),
    ("--- Phase 0B-2", "0B2"),
    ("--- Phase 0A:", "0A"),
    ("--- Phase 0B:", "0B"),
    ("--- Phase 1:", "1"),
    ("--- Phase 2:", "2"),
)
_KEPT_COLUMNS_RE = re.compile(r"Columns kept in narrative \(\d+\): (\[.*\])")
_LABEL_COLUMNS = (
    "tag_innerText",
    "clipboard_content",
    "cell_content",
    "title",
    "tag_value",
    "tag_name",
)


# --------------------------------------------------------------------- helpers
def read_raw_log(csv_path: str | Path) -> pd.DataFrame:
    """Read a SmartRPA CSV exactly as the pipeline does (sep ';', stable sort)."""
    df = pd.read_csv(csv_path, sep=";", dtype=str, keep_default_na=True)
    if df.empty:
        raise ValueError("The log contains no data rows.")
    if len(df.columns) < 2:
        raise ValueError(
            "Expected a semicolon-delimited SmartRPA CSV, but only one column "
            "was found."
        )
    if "timestamp" in df.columns:
        stamps = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.assign(_ts=stamps).sort_values(by="_ts", kind="stable")
        df = df.drop(columns="_ts")
    return df.reset_index(drop=True)


def _cell(row: pd.Series, col: str) -> str:
    if col not in row.index:
        return ""
    val = row[col]
    return "" if pd.isna(val) else str(val).strip()


_NARRATIVE_FIELD = re.compile(r"\|\s*([A-Za-z_]+):\s*([^|]*)")


def describe_narrative(narrative: str) -> dict[str, str]:
    """Split a serialised narrative into the parts the UI displays."""
    app = re.search(r"\[APP:\s*([^\]]+)\]", narrative)
    action = re.search(r"action:\s*([^|]+)", narrative)
    fields = {k.lower(): v.strip() for k, v in _NARRATIVE_FIELD.findall(narrative)}
    text = next(
        (fields[c.lower()] for c in _LABEL_COLUMNS if fields.get(c.lower())), ""
    )
    if not text:
        for col in ("event_dest_path", "event_src_path", "workbook"):
            if fields.get(col):
                text = fields[col].replace("\\", "/").rsplit("/", 1)[-1]
                break
    url = fields.get("browser_url", "")
    return {
        "app": app.group(1).strip() if app else "",
        "action": action.group(1).strip() if action else "",
        "text": text,
        "where": url.rstrip("/").rsplit("/", 1)[-1] if url else "",
    }


def _raw_label(row: pd.Series) -> str:
    for col in _LABEL_COLUMNS:
        value = _cell(row, col)
        if value:
            return value
    for col in ("event_dest_path", "event_src_path", "workbook"):
        value = _cell(row, col)
        if value:
            return value.replace("\\", "/").rsplit("/", 1)[-1]
    url = _cell(row, "browser_url")
    if url:
        return url.split("://", 1)[-1]
    return ""


def describe_raw_log(df: pd.DataFrame) -> dict[str, Any]:
    """Summary statistics and a compact per-event view for the UI."""
    events = []
    for i, row in df.iterrows():
        ts = _cell(row, "timestamp")
        url = _cell(row, "browser_url")
        events.append(
            {
                "i": int(i),
                "time": ts[11:19] if len(ts) >= 19 else ts,
                "app": _cell(row, "application") or "?",
                "type": _cell(row, "event_type") or "?",
                "label": _raw_label(row),
                "where": url.rstrip("/").rsplit("/", 1)[-1] if url else "",
            }
        )
    populated = [c for c in df.columns if df[c].notna().any()]
    span = ""
    if "timestamp" in df.columns:
        stamps = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
        if not stamps.empty:
            span = f"{stamps.min():%H:%M:%S} – {stamps.max():%H:%M:%S}"
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "populated_columns": len(populated),
        "apps": df["application"].value_counts().to_dict()
        if "application" in df.columns
        else {},
        "event_types": df["event_type"].value_counts().to_dict()
        if "event_type" in df.columns
        else {},
        "span": span,
        "events": events,
    }


# ----------------------------------------------------------------- base class
class Run:
    """An append-only event log that SSE subscribers read from."""

    def __init__(self, run_id: str, run_dir: Path) -> None:
        self.id = run_id
        self.dir = run_dir
        self.events: list[dict[str, Any]] = []
        self.finished = False
        self.cancelled = False
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self._review_decision: dict[str, Any] | None = None
        self._review_ready = threading.Event()
        self._journal = None

    def open_journal(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._journal = open(self.dir / "events.jsonl", "w", encoding="utf-8")

    def emit(self, event: dict[str, Any], t: float | None = None) -> None:
        with self._lock:
            event = dict(event)
            event["t"] = round(time.monotonic() - self._t0 if t is None else t, 3)
            event["seq"] = len(self.events)
            self.events.append(event)
            if self._journal is not None:
                self._journal.write(json.dumps(event, ensure_ascii=False) + "\n")
                self._journal.flush()

    def finish(self) -> None:
        self.finished = True
        if self._journal is not None:
            self._journal.close()
            self._journal = None

    def submit_review(self, decision: dict[str, Any]) -> None:
        self._review_decision = decision
        self._review_ready.set()

    def cancel(self) -> None:
        self.cancelled = True
        self._review_ready.set()

    def wait_for_review(self) -> dict[str, Any] | None:
        self._review_ready.wait()
        return None if self.cancelled else self._review_decision

    def file_path(self, name: str) -> Path | None:
        if name not in (OUTPUT_XLSX, OUTPUT_JSON):
            return None
        path = self.dir / name
        return path if path.exists() else None


# ------------------------------------------------------------------ live runs
class LiveRun(Run):
    """Runs the real pipeline and narrates it as events."""

    def __init__(
        self,
        *,
        csv_path: str,
        log_name: str,
        routines: list[dict[str, Any]],
        model: str,
        review: bool = True,
        relevance_filter: bool = True,
        include_thoughts: bool = True,
        use_cache: bool = True,
        transport: Callable[..., dict[str, Any]] | None = None,
        source: str = "Live run",
        cache_dir: Path | None = None,
    ) -> None:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        super().__init__(run_id, RUNS_DIR / run_id)
        self.csv_path = csv_path
        self.log_name = log_name
        self.routines = [
            {"routine_name": str(r["routine_name"]).strip(),
             "executions": int(r["executions"])}
            for r in routines
        ]
        self.model = model
        self.review = review
        self.relevance_filter = relevance_filter
        self.include_thoughts = include_thoughts
        self.use_cache = use_cache
        self.transport = transport
        self.source = source
        self.cache_dir = cache_dir or CACHE_DIR

        self.phase = "load"
        self._phase_started: dict[str, float] = {}
        self._answers: dict[str, Any] = {}
        self._failed_calls: set[str] = set()
        self._log_lines: list[str] = []
        self._kept_columns: list[str] | None = None
        self._muted = False
        self._raw: pd.DataFrame | None = None
        self._noise_types: list[str] = []

    # -------------------------------------------------------------- plumbing
    def start(self, background: bool = True) -> None:
        self.open_journal()
        if background:
            threading.Thread(target=self._execute, daemon=True).start()
        else:
            self._execute()

    def _emit_from_client(self, event: dict[str, Any]) -> None:
        if event["type"] == "llm_result":
            self._answers[event["phase"]] = event["answer"]
        elif event["type"] == "llm_error":
            self._failed_calls.add(event["phase"])
        self.emit(event)

    def _on_line(self, line: str) -> None:
        if self._muted:
            return
        self._log_lines.append(line)
        match = _KEPT_COLUMNS_RE.search(line)
        if match:
            try:
                self._kept_columns = list(ast.literal_eval(match.group(1)))
            except (ValueError, SyntaxError):
                pass
        for prefix, phase in _PHASE_HEADERS:
            if line.startswith(prefix):
                self._enter_phase(phase)
                break
        if line.strip():
            self.emit({"type": "log", "phase": self.phase, "line": line})

    def _enter_phase(self, phase: str) -> None:
        if phase == self.phase:
            return
        self._close_phase(self.phase)
        self.phase = phase
        self._phase_started[phase] = time.monotonic()
        self.emit({"type": "phase", "phase": phase, "status": "start"})

    def _close_phase(self, phase: str, status: str = "done") -> None:
        finaliser = {"0A": self._finalise_0a, "0B": self._finalise_0b}.get(phase)
        if finaliser is not None and status == "done":
            finaliser()
        started = self._phase_started.get(phase, time.monotonic())
        self.emit(
            {
                "type": "phase",
                "phase": phase,
                "status": status,
                "seconds": round(time.monotonic() - started, 2),
            }
        )

    def _phase_data(self, phase: str, data: dict[str, Any]) -> None:
        self.emit({"type": "phase_data", "phase": phase, "data": data})

    # ------------------------------------------------------------- execution
    def _execute(self) -> None:
        router = install_stdout_router()
        router.register(self._on_line)
        success = False
        try:
            success = self._pipeline()
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            traceback.print_exc()
            self.emit({"type": "error", "phase": self.phase, "message": str(exc)})
        finally:
            router.unregister()
            self.emit({"type": "done", "success": success})
            self._write_meta(success)
            self.finish()

    def _fail(self, message: str) -> bool:
        self._close_phase(self.phase, status="error")
        recent = [l for l in self._log_lines[-40:] if "[!]" in l or "CRITICAL" in l]
        self.emit(
            {"type": "error", "phase": self.phase, "message": message,
             "details": recent[-6:]}
        )
        return False

    def _pipeline(self) -> bool:
        declared = [r["routine_name"] for r in self.routines]
        self.emit(
            {
                "type": "run_start",
                "run_id": self.id,
                "log_name": self.log_name,
                "routines": self.routines,
                "model": self.model,
                "options": {
                    "review": self.review,
                    "relevance_filter": self.relevance_filter,
                    "include_thoughts": self.include_thoughts,
                    "use_cache": self.use_cache,
                },
                "source": self.source,
            }
        )
        self._phase_started["load"] = time.monotonic()
        self.emit({"type": "phase", "phase": "load", "status": "start"})
        self._raw = read_raw_log(self.csv_path)
        self.emit({"type": "raw_log", "data": describe_raw_log(self._raw)})

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = (
            self.cache_dir / "gemini_cache.json" if self.use_cache
            else self.dir / "gemini_cache.json"
        )
        client = InstrumentedClient(
            on_event=self._emit_from_client,
            phase_of=lambda: self.phase,
            include_thoughts=self.include_thoughts,
            thoughts_file=str(self.cache_dir / "thoughts.json"),
            seed_cache_files=(str(THESIS_CACHE),) if self.use_cache else (),
            cache_file=str(cache_file),
            token_log_file=str(self.dir / "token_telemetry.csv"),
            transport=self.transport,
        )

        # ---- Phase 0A / 0B / 0B-2 (phase changes come from printed headers)
        try:
            df = load_and_serialize_smartrpa(self.csv_path, self.model, client)
        except (FileNotFoundError, ValueError) as exc:
            return self._fail(f"Could not load or serialise the log: {exc}")
        self._finalise_0b2(df)

        # ---- Serialisation (local, no LLM call)
        self._enter_phase("serialize")
        self._phase_data("serialize", self._serialisation_data(df))

        # ---- Phase 0A-2
        if self.relevance_filter:
            df = tag_irrelevant_nodes(df, declared, self.model, client)
            flagged = self._answers.get("0A2") or []
            noise = sorted(
                int(n) for n in df.loc[df["is_probable_noise"], "node_id"].tolist()
            )
            valid = set(int(n) for n in df["node_id"])
            flagged_ids = sorted({int(n) for n in flagged if _is_int(n) and int(n) in valid})
            self._phase_data(
                "0A2",
                {
                    "flagged": flagged_ids,
                    "rescued": sorted(set(flagged_ids) - set(noise)),
                    "noise": noise,
                    "failed": "0A2" in self._failed_calls,
                },
            )
        else:
            self.emit({"type": "phase", "phase": "0A2", "status": "skipped"})

        # ---- Phase 1
        topology = infer_sharing_topology(df, self.routines, self.model, client)
        if topology is None:
            return self._fail("Phase 1 could not infer the sharing topology.")
        narrative = dict(zip(df["node_id"].astype(int), df["llm_narrative"].astype(str)))
        answer1 = self._answers.get("1") or {}
        self._phase_data(
            "1",
            {
                "reasoning": str(answer1.get("reasoning", "")) if isinstance(answer1, dict) else "",
                "proposed": len(answer1.get("shared_actions", []))
                if isinstance(answer1, dict) else 0,
                "topology": self._decorate(topology, narrative),
            },
        )

        # ---- Human review (the CLI's --review, as an interactive step)
        if self.review:
            self._enter_phase("review")
            self.emit(
                {
                    "type": "review_required",
                    "topology": self._decorate(topology, narrative),
                    "routines": declared,
                }
            )
            decision = self.wait_for_review()
            if decision is None or decision.get("action") != "accept":
                self._close_phase("review", status="error")
                self.emit({"type": "error", "phase": "review",
                           "message": "Review cancelled by the operator."})
                return False
            topology, changes = self._apply_review(topology, decision, declared)
            self.emit(
                {
                    "type": "review_resolved",
                    "topology": self._decorate(topology, narrative),
                    "changes": changes,
                }
            )

        # ---- Phase 2
        ok = generate_segmented_log(
            df,
            topology,
            self.routines,
            self.model,
            output_file=str(self.dir / OUTPUT_XLSX),
            client=client,
        )
        if not ok:
            return self._fail("Phase 2 could not construct a valid execution mapping.")
        self._phase_data("2", self._result_data(df, topology))
        self._close_phase("2")
        return True

    # ------------------------------------------------------------ finalisers
    def _finalise_0a(self) -> None:
        answer = self._answers.get("0A")
        noise_types = [str(t) for t in answer] if isinstance(answer, list) else []
        self._noise_types = noise_types
        raw = self._raw
        removed = []
        if raw is not None and "event_type" in raw.columns:
            removed = [int(i) for i in raw.index[raw["event_type"].isin(noise_types)]]
        self._phase_data(
            "0A",
            {
                "vocabulary": raw["event_type"].value_counts().to_dict()
                if raw is not None and "event_type" in raw.columns else {},
                "noise_types": noise_types,
                "removed": removed,
                "failed": "0A" in self._failed_calls,
            },
        )

    def _finalise_0b(self) -> None:
        answer = self._answers.get("0B")
        columns = list(self._raw.columns) if self._raw is not None else []
        failed = "0B" in self._failed_calls or not isinstance(answer, list)
        if failed:
            proposed = [c for c in _FALLBACK_IGNORE_COLUMNS if c in columns]
        else:
            proposed = [
                str(c) for c in answer
                if str(c) in columns and str(c) not in _ANCHOR_COLUMNS
            ]
        self._phase_data(
            "0B",
            {
                "columns": columns,
                "empty": [c for c in columns if self._raw is not None
                          and not self._raw[c].notna().any()],
                "pruned": proposed,
                "anchors": list(_ANCHOR_COLUMNS),
                "failed": failed,
            },
        )

    def _finalise_0b2(self, df: pd.DataFrame) -> None:
        columns = [c for c in df.columns if c not in ("node_id", "llm_narrative")]
        kept = self._kept_columns or columns
        answer = self._answers.get("0B")
        proposed = {str(c) for c in answer} if isinstance(answer, list) else set()
        restored = [c for c in kept if c in proposed]
        self._phase_data("0B2", {"kept": kept, "restored": restored})

    def _serialisation_data(self, df: pd.DataFrame) -> dict[str, Any]:
        raw = self._raw
        kept_raw = list(range(len(raw))) if raw is not None else []
        if raw is not None and self._noise_types:
            kept_raw = [
                int(i) for i in raw.index[~raw["event_type"].isin(self._noise_types)]
            ]
        if len(kept_raw) != len(df):
            kept_raw = [None] * len(df)
        nodes = []
        for (nid, nar), raw_i in zip(
            zip(df["node_id"].astype(int), df["llm_narrative"].astype(str)), kept_raw
        ):
            nodes.append(
                {"id": int(nid), "raw": raw_i, "narrative": nar,
                 **describe_narrative(nar)}
            )
        # A representative example: the first event with a rich narrative.
        example = None
        for node in nodes:
            if node["narrative"].count("|") >= 3 and node["raw"] is not None:
                row = raw.iloc[node["raw"]]
                example = {
                    "node": node["id"],
                    "raw_fields": {c: _cell(row, c) for c in raw.columns if _cell(row, c)},
                    "narrative": node["narrative"],
                }
                break
        return {"nodes": nodes, "example": example}

    @staticmethod
    def _decorate(
        topology: list[dict[str, Any]], narrative: dict[int, str]
    ) -> list[dict[str, Any]]:
        return [
            {
                "node_id": int(a["node_id"]),
                "shared_with": list(a["shared_with"]),
                "justification": str(a.get("justification", "")),
                **describe_narrative(narrative.get(int(a["node_id"]), "")),
                "narrative": narrative.get(int(a["node_id"]), ""),
            }
            for a in topology
        ]

    @staticmethod
    def _apply_review(
        topology: list[dict[str, Any]],
        decision: dict[str, Any],
        declared: list[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Apply the operator's edits, with the CLI review's semantics."""
        edits = {}
        for item in decision.get("topology", []) or []:
            try:
                edits[int(item["node_id"])] = [
                    n for n in declared if n in set(item.get("shared_with", []))
                ]
            except (KeyError, TypeError, ValueError):
                continue
        result, changes = [], []
        for action in topology:
            nid = int(action["node_id"])
            subset = edits.get(nid, list(action["shared_with"]))
            if sorted(subset) != sorted(action["shared_with"]):
                changes.append(
                    {"node_id": nid, "before": list(action["shared_with"]),
                     "after": subset}
                )
            if subset:  # emptied actions become ordinary routable steps
                result.append({**action, "shared_with": subset})
        return result, changes

    def _result_data(
        self, df: pd.DataFrame, topology: list[dict[str, Any]]
    ) -> dict[str, Any]:
        with open(self.dir / OUTPUT_JSON, "r", encoding="utf-8") as handle:
            audit = json.load(handle)
        plan = audit.get("routing_plan", [])
        declared = {r["routine_name"].lower(): r["routine_name"] for r in self.routines}
        order = {r["routine_name"]: i for i, r in enumerate(self.routines)}
        names = [declared[str(b.get("routine_name", "")).strip().lower()] for b in plan]
        subset_by_id = {int(a["node_id"]): list(a["shared_with"]) for a in topology}

        traces, seen, collisions = [], set(), 0
        for block, name in zip(plan, names):
            exec_idx = _int_or(block.get("execution_index"), 0)
            if exec_idx <= 0:
                exec_idx = sum(1 for t in traces if t["routine"] == name) + 1
            trace_id = f"{name}_exec{exec_idx}"
            while trace_id in seen:
                trace_id += "_b"
                collisions += 1
            seen.add(trace_id)
            specific = [_int_or(i, -1) for i in block.get("assigned_node_indices", [])]
            shared = sorted(n for n, subset in subset_by_id.items() if name in subset)
            traces.append(
                {
                    "id": trace_id,
                    "routine": name,
                    "execution": exec_idx,
                    "nodes": sorted(set(shared + [i for i in specific if i >= 0])),
                    "shared": shared,
                    "reasoning": str(block.get("reasoning", "")),
                }
            )
        traces.sort(key=lambda t: (order[t["routine"]], t["execution"]))

        noise_all = [int(n) for n in audit.get("noise_node_ids", [])]
        relevance = set()
        if "is_probable_noise" in df.columns:
            relevance = {int(n) for n in df.loc[df["is_probable_noise"], "node_id"]}
        relevance -= set(subset_by_id)
        unrouted = sorted(set(noise_all) - relevance)

        s1 = _check_declared_counts(names, self.routines)
        checks = [
            {"id": "P1", "label": "Partition: every routable event in exactly one execution",
             "status": "pass",
             "detail": f"{len(unrouted)} unrouted event(s) diverted to Noise" if unrouted else ""},
            {"id": "P2", "label": "Every execution names a declared routine",
             "status": "pass", "detail": ""},
            {"id": "P3", "label": "Cover: every event in a trace or in the Noise sheet",
             "status": "pass", "detail": ""},
            {"id": "S1", "label": "Execution counts match the declared routines",
             "status": "warn" if s1 else "pass", "detail": " ".join(s1)},
            {"id": "S2", "label": "Trace identifiers are unique",
             "status": "warn" if collisions else "pass",
             "detail": f"{collisions} duplicate id(s) suffixed" if collisions else ""},
        ]
        return {
            "traces": traces,
            "noise": sorted(noise_all),
            "noise_relevance": sorted(relevance & set(noise_all)),
            "noise_unrouted": unrouted,
            "checks": checks,
            "files": [OUTPUT_XLSX, OUTPUT_JSON],
        }

    def _write_meta(self, success: bool) -> None:
        meta = {
            "id": self.id,
            "created": datetime.now().isoformat(timespec="seconds"),
            "log_name": self.log_name,
            "routines": self.routines,
            "model": self.model,
            "source": self.source,
            "success": success,
            "duration": self.events[-1]["t"] if self.events else 0,
        }
        try:
            with open(self.dir / "meta.json", "w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2, ensure_ascii=False)
        except OSError:
            pass


def _is_int(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- recordings
def save_recording(run: Run, name: str) -> str:
    """Copy a finished run folder into ``ui/recordings/<slug>``."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or run.id
    target = RECORDINGS_DIR / slug
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for item in ("events.jsonl", "meta.json", OUTPUT_XLSX, OUTPUT_JSON):
        if (run.dir / item).exists():
            shutil.copy2(run.dir / item, target / item)
    meta_path = target / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["title"] = name
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return slug


def list_recordings() -> list[dict[str, Any]]:
    items = []
    if not RECORDINGS_DIR.exists():
        return items
    for folder in sorted(RECORDINGS_DIR.iterdir()):
        meta_path = folder / "meta.json"
        if folder.is_dir() and meta_path.exists() and (folder / "events.jsonl").exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["id"] = folder.name
            items.append(meta)
    return items


def load_recording(rec_id: str) -> list[dict[str, Any]]:
    folder = _recording_dir(rec_id)
    with open(folder / "events.jsonl", "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _recording_dir(rec_id: str) -> Path:
    folder = (RECORDINGS_DIR / rec_id).resolve()
    if folder.parent != RECORDINGS_DIR.resolve() or not folder.is_dir():
        raise FileNotFoundError(rec_id)
    return folder


# ------------------------------------------------------------------- replays
class ReplayRun(Run):
    """Re-emits a recorded run with its original pacing."""

    # Minimum on-screen gaps so that cached calls and log bursts stay legible.
    _MIN_GAP = {"llm_result": 1.4, "phase": 0.35, "phase_data": 0.25}

    def __init__(self, rec_id: str, speed: float = 1.0) -> None:
        super().__init__("replay-" + uuid.uuid4().hex[:8], _recording_dir(rec_id))
        self.recorded = load_recording(rec_id)
        self.speed = max(0.25, min(float(speed), 16.0))

    def start(self) -> None:
        threading.Thread(target=self._play, daemon=True).start()

    def _play(self) -> None:
        clock = 0.0
        previous_t = None
        for event in self.recorded:
            if self.cancelled:
                break
            t = float(event.get("t", 0))
            gap = 0.0 if previous_t is None else max(0.0, t - previous_t)
            previous_t = t
            if event["type"] == "review_resolved":
                gap = 0.2  # the human's thinking time was spent live, below
            gap = gap / self.speed
            gap = min(gap, 25.0)
            gap = max(gap, self._MIN_GAP.get(event["type"], 0.03) / max(1.0, self.speed / 2))
            time.sleep(gap)
            clock += gap
            payload = {k: v for k, v in event.items() if k not in ("t", "seq")}
            if payload["type"] == "run_start":
                payload["replay"] = True
            if payload["type"] == "review_required":
                payload["replay"] = True
                self.emit(payload, t=clock)
                if self.wait_for_review() is None:
                    break
                continue
            self.emit(payload, t=clock)
        if not self.events or self.events[-1]["type"] != "done":
            self.emit({"type": "done", "success": False}, t=clock)
        self.finish()
