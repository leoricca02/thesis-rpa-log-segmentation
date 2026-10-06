"""Build a replayable recording from the committed thesis artefacts.

No API key and no network are needed. Phases 0A, 0B, 0A-2 and 1 replay the
responses committed in ``results/case_study/gemini_cache.json``; Phase 2,
which the cache deliberately does not contain, is served the routing plan
committed in ``results/case_study/Final_Segmented_Master_Log_routing.json``.
Every other call fails loudly, so a recording is produced only if the prompts
still match the committed cache exactly.

The result is the thesis's reference run as seen through the UI. It is not a
live inference: it carries no Gemini thought summaries, and the recorded
timings are those of a cache replay. For the demo video, record a live run
from the UI instead ("Save as recording" on the result screen).

    python ui/build_reference_recording.py
"""

from __future__ import annotations

import itertools
import json
import re
import sys
import tempfile
import time
from pathlib import Path

_UI_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _UI_DIR.parent
for _path in (_REPO_ROOT / "src", _UI_DIR):
    sys.path.insert(0, str(_path))

from pipeline_runner import THESIS_CACHE, LiveRun, save_recording  # noqa: E402
from smart_llm_client import SmartLLMClient  # noqa: E402

CASE_STUDY = _REPO_ROOT / "data" / "case_study" / "caso_studio_trasferte_2exec.csv"
ROUTING = _REPO_ROOT / "results" / "case_study" / "Final_Segmented_Master_Log_routing.json"
ROUTINES = [
    {"routine_name": "Travel Authorization", "executions": 2},
    {"routine_name": "Expense Reimbursement", "executions": 2},
    {"routine_name": "Purchase Order Approval", "executions": 2},
    {"routine_name": "Student Grant Disbursement", "executions": 2},
]


_COUNT_LINE = re.compile(r"^- '.*': (\d+) occurrences$")


def _phase0a_tie_variant(payload: dict) -> object | None:
    """Find the committed Phase 0A answer despite tie ordering.

    Phase 0A lists event types via ``value_counts()``, whose order among
    equal counts differs across pandas versions (the committed entry was
    recorded with pandas 2.2); that alone changes the cache key. Re-ordering
    only types that share a count recovers the committed entry without
    altering what the model was shown.
    """
    user = payload["contents"][0]["parts"][0]["text"]
    head, _, body = user.partition("\n\n")
    lines = body.split("\n")
    if not lines or not all(_COUNT_LINE.match(l) for l in lines):
        return None
    groups = [list(g) for _, g in itertools.groupby(
        lines, key=lambda l: _COUNT_LINE.match(l).group(1))]
    cache = json.loads(THESIS_CACHE.read_text(encoding="utf-8"))
    system = payload["systemInstruction"]["parts"][0]["text"]
    schema = payload["generationConfig"]["responseSchema"]
    for combo in itertools.product(*(itertools.permutations(g) for g in groups)):
        prompt = head + "\n\n" + "\n".join(l for g in combo for l in g)
        key = SmartLLMClient._get_cache_key(
            "gemini-3.5-flash", system, prompt, schema, 0)
        if key in cache:
            return cache[key]
    return None


def _envelope(answer: object) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(answer)}]}}]}


def _committed_artefacts_transport(url: str, payload: dict, timeout: float) -> dict:
    system = payload["systemInstruction"]["parts"][0]["text"]
    if "untangles interleaved UI" in system:
        plan = json.loads(ROUTING.read_text(encoding="utf-8"))["routing_plan"]
        return _envelope(plan)
    answer = _phase0a_tie_variant(payload)
    if answer is not None:
        return _envelope(answer)
    raise RuntimeError(
        "A Phase 0/1 prompt missed the committed cache; the recording would "
        "not reflect the thesis run."
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as cache_dir:
        run = LiveRun(
            csv_path=str(CASE_STUDY),
            log_name=CASE_STUDY.name,
            routines=ROUTINES,
            model="gemini-3.5-flash",
            review=True,
            include_thoughts=False,
            use_cache=True,
            transport=_committed_artefacts_transport,
            cache_dir=Path(cache_dir),
            source=(
                "Reference run rebuilt from the committed thesis artefacts "
                "(cached Phase 0–1 responses, committed Phase 2 routing). "
                "Not a live inference: no thought summaries."
            ),
        )
        run.start(background=True)
        while not run.finished:
            if any(e["type"] == "review_required" for e in run.events) and not any(
                e["type"] == "review_resolved" for e in run.events
            ):
                time.sleep(1.0)  # a short, visible pause for the review step
                run.submit_review({"action": "accept", "topology": []})
            time.sleep(0.1)

    if not run.events or not run.events[-1].get("success"):
        print("\n[!] The reference run failed; no recording written.")
        return 1
    rec_id = save_recording(run, "Case study — thesis reference run")
    print(f"\n[SUCCESS] Recording written to ui/recordings/{rec_id}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
