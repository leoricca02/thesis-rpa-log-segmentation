"""Tests for the demo UI backend (ui/) — fully offline.

The UI observes the pipeline from the outside; these tests pin down that it
reports what the pipeline did without changing what the pipeline does.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time

import pytest

_UI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui")
if _UI_DIR not in sys.path:
    sys.path.insert(0, _UI_DIR)

import build_reference_recording as reference  # noqa: E402
import pipeline_runner  # noqa: E402
from instrumented_client import InstrumentedClient  # noqa: E402

_SCHEMA = {"type": "ARRAY", "items": {"type": "STRING"}}


def _client(tmp_path, transport, events, **kwargs):
    return InstrumentedClient(
        on_event=events.append,
        phase_of=lambda: "1",
        cache_file=str(tmp_path / "cache.json"),
        token_log_file=str(tmp_path / "tokens.csv"),
        thoughts_file=str(tmp_path / "thoughts.json"),
        transport=transport,
        sleep=lambda _s: None,
        **kwargs,
    )


def _thinking_envelope(thoughts: str, answer: str) -> dict:
    return {
        "candidates": [{"content": {"parts": [
            {"text": thoughts, "thought": True},
            {"text": answer},
        ]}}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 4,
                          "thoughtsTokenCount": 7, "totalTokenCount": 21},
    }


def test_thought_summary_is_split_from_the_answer(tmp_path):
    payloads, events = [], []

    def transport(url, payload, timeout):
        payloads.append(payload)
        return _thinking_envelope("**Plan** weigh the counts", '["a"]')

    client = _client(tmp_path, transport, events, include_thoughts=True)
    assert client.generate_content("gemini-3.5-flash", "s", "u", _SCHEMA, 512) == ["a"]

    assert payloads[0]["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "low", "includeThoughts": True}
    call, result = events
    assert call["type"] == "llm_call" and call["cached"] is False
    assert result["thoughts"] == "**Plan** weigh the counts"
    assert result["tokens"] == {"input": 10, "output": 4, "thoughts": 7, "total": 21}


def test_thoughts_are_not_requested_without_a_thinking_budget(tmp_path):
    payloads = []

    def transport(url, payload, timeout):
        payloads.append(payload)
        return {"candidates": [{"content": {"parts": [{"text": "[]"}]}}]}

    client = _client(tmp_path, transport, [], include_thoughts=True)
    client.generate_content("gemini-3.5-flash", "s", "u", _SCHEMA, 0)
    assert "includeThoughts" not in payloads[0]["generationConfig"]["thinkingConfig"]


def test_cache_hit_replays_the_stored_thoughts(tmp_path):
    calls, events = [], []

    def transport(url, payload, timeout):
        calls.append(1)
        return _thinking_envelope("first pass", '["x"]')

    client = _client(tmp_path, transport, events, include_thoughts=True)
    client.generate_content("m-gemini-3", "s", "u", _SCHEMA, 1024)
    client.generate_content("m-gemini-3", "s", "u", _SCHEMA, 1024)

    assert len(calls) == 1
    hit = events[-1]
    assert hit["cached"] is True and hit["thoughts"] == "first pass"


def test_seed_cache_is_reused_without_being_overwritten(tmp_path):
    seed = tmp_path / "seed.json"
    key = InstrumentedClient._get_cache_key("m", "s", "u", _SCHEMA, 0)
    seed.write_text(json.dumps({key: ["seeded"]}), encoding="utf-8")

    def transport(url, payload, timeout):
        raise AssertionError("a seeded call must not reach the network")

    client = _client(tmp_path, transport, [], seed_cache_files=(str(seed),))
    assert client.generate_content("m", "s", "u", _SCHEMA, 0) == ["seeded"]


def test_review_edits_follow_the_cli_semantics():
    topology = [
        {"node_id": 0, "shared_with": ["A", "B"], "justification": "login"},
        {"node_id": 3, "shared_with": ["A"], "justification": "opener"},
    ]
    decision = {"topology": [
        {"node_id": 0, "shared_with": ["B", "A", "Ghost"]},  # unchanged, unknown name ignored
        {"node_id": 3, "shared_with": []},                   # emptied -> ordinary step
    ]}
    result, changes = pipeline_runner.LiveRun._apply_review(topology, decision, ["A", "B"])

    assert result == [{"node_id": 0, "shared_with": ["A", "B"], "justification": "login"}]
    assert changes == [{"node_id": 3, "before": ["A"], "after": []}]


def _reference_run(tmp_path, monkeypatch) -> pipeline_runner.LiveRun:
    monkeypatch.setattr(pipeline_runner, "RUNS_DIR", tmp_path / "runs")
    run = pipeline_runner.LiveRun(
        csv_path=str(reference.CASE_STUDY),
        log_name=reference.CASE_STUDY.name,
        routines=reference.ROUTINES,
        model="gemini-3.5-flash",
        review=True,
        include_thoughts=False,
        transport=reference._committed_artefacts_transport,
        cache_dir=tmp_path / "cache",
    )
    run.start(background=True)
    deadline = time.monotonic() + 60
    while not run.finished and time.monotonic() < deadline:
        if any(e["type"] == "review_required" for e in run.events):
            run.submit_review({"action": "accept", "topology": []})
        time.sleep(0.05)
    assert run.finished
    return run


def test_reference_run_reproduces_the_thesis_segmentation(tmp_path, monkeypatch):
    run = _reference_run(tmp_path, monkeypatch)
    events = run.events
    assert events[-1] == {**events[-1], "type": "done", "success": True}

    started = [e["phase"] for e in events if e["type"] == "phase" and e["status"] == "start"]
    assert started == ["load", "0A", "0B", "0B2", "serialize", "0A2", "1", "review", "2"]

    data = {e["phase"]: e["data"] for e in events if e["type"] == "phase_data"}
    assert len(data["1"]["topology"]) == 8
    assert data["0A2"]["noise"] == [18, 30, 42, 54, 66, 78, 90, 102]
    result = data["2"]
    assert len(result["traces"]) == 8
    assert all(c["status"] == "pass" for c in result["checks"])
    routed = {n for t in result["traces"] for n in t["nodes"]} | set(result["noise"])
    assert routed == set(range(114))  # the cover: nothing is lost
    assert (run.dir / pipeline_runner.OUTPUT_XLSX).exists()


def test_replay_reemits_a_recorded_run(tmp_path, monkeypatch):
    run = _reference_run(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline_runner, "RECORDINGS_DIR", tmp_path / "recordings")
    rec_id = pipeline_runner.save_recording(run, "Test recording")
    assert [r["id"] for r in pipeline_runner.list_recordings()] == [rec_id]

    replay = pipeline_runner.ReplayRun(rec_id, speed=16)
    replay.start()
    deadline = time.monotonic() + 60
    while not replay.finished and time.monotonic() < deadline:
        if any(e["type"] == "review_required" for e in replay.events):
            replay.submit_review({"action": "accept"})
        time.sleep(0.05)

    assert [e["type"] for e in replay.events] == [e["type"] for e in run.events]
    assert replay.file_path(pipeline_runner.OUTPUT_XLSX) is not None
    shutil.rmtree(tmp_path / "recordings")
