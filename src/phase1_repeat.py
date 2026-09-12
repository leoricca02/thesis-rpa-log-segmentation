"""Phase-1 repeatability harness (does NOT modify the pipeline).

Runs ``infer_sharing_topology`` N times on one log, each time with a FRESH
throwaway cache so every call is a real API inference (the production cache
would otherwise return one fixed answer). Tabulates, per shared action, how
often each routine appears in its inferred subset, so the run-to-run variance
of the topology can be measured rather than assumed.

Usage:
    python src/phase1_repeat.py <log.csv> <n_runs> "Name1:execs,Name2:execs,..."

Example (case study, 4 routines x2 executions, 5 repeats):
    python src/phase1_repeat.py data/case_study/caso_studio_trasferte_2exec.csv 5 \
      "Travel Authorization:2,Expense Reimbursement:2,Purchase Order Approval:2,Student Grant Disbursement:2"
"""
from __future__ import annotations

import os
import sys
import tempfile
from collections import defaultdict

from data_pipeline import load_and_serialize_smartrpa
from phase1_boundary_reasoning import infer_sharing_topology
from smart_llm_client import SmartLLMClient

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")


def _parse_constraints(spec: str) -> list[dict]:
    out = []
    for part in spec.split(","):
        name, _, execs = part.rpartition(":")
        out.append({"routine_name": name.strip(), "executions": int(execs)})
    return out


def main() -> None:
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    log_path, n_runs, spec = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    constraints = _parse_constraints(spec)
    routine_names = [c["routine_name"] for c in constraints]

    # Serialise ONCE with a normal client (Phase 0 is deterministic enough and
    # we want the same node set every run). Cache Phase 0 to save tokens.
    base_client = SmartLLMClient()
    df = load_and_serialize_smartrpa(log_path, MODEL, base_client)

    # node_id -> short label, for a readable table
    label = {}
    for nid, nar in zip(df["node_id"], df["llm_narrative"].astype(str)):
        txt = nar
        lab = ""
        if "tag_innerText:" in txt:
            lab = txt.split("tag_innerText:")[-1].strip()
        elif "browser_url:" in txt:
            lab = txt.split("browser_url:")[-1].strip().rsplit("/", 1)[-1]
        label[int(nid)] = lab[:28]

    # node_id -> routine -> count of runs in which it was in the subset
    tally: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seen_counts: dict[int, int] = defaultdict(int)  # runs where node was shared at all
    n_shared_per_run = []

    for i in range(n_runs):
        # Fresh throwaway cache => guaranteed real inference each time.
        with tempfile.TemporaryDirectory() as tmp:
            client = SmartLLMClient(
                cache_file=os.path.join(tmp, "c.json"),
                token_log_file=os.path.join(tmp, "t.csv"),
            )
            topo = infer_sharing_topology(df, constraints, MODEL, client)
        if topo is None:
            print(f"[run {i+1}] API failure; skipping")
            continue
        n_shared_per_run.append(len(topo))
        for action in topo:
            nid = int(action["node_id"])
            seen_counts[nid] += 1
            for r in action["shared_with"]:
                tally[nid][r] += 1

    # ---- report ----
    valid_runs = len(n_shared_per_run)
    print("\n\n==================== REPEATABILITY REPORT ====================")
    print(f"Log: {log_path}")
    print(f"Runs: {valid_runs} (of {n_runs} requested)")
    print(f"Shared-action count per run: {n_shared_per_run}")
    print("\nPer shared node: how many runs put each routine in its subset")
    print("(node only listed if it was shared in >=1 run)\n")
    for nid in sorted(seen_counts):
        print(f"Node {nid}  \"{label.get(nid,'')}\"  "
              f"(shared in {seen_counts[nid]}/{valid_runs} runs)")
        for r in routine_names:
            c = tally[nid].get(r, 0)
            bar = "#" * c + "." * (valid_runs - c)
            flag = ""
            if 0 < c < valid_runs:
                flag = "  <-- VARIES"
            print(f"    {r:<22} {c}/{valid_runs}  [{bar}]{flag}")
        print()
    print("==============================================================")
    print("Stable subset membership = c equals 0 or equals #runs.")
    print("Any '<-- VARIES' line is a boundary the model flips on across runs.")


if __name__ == "__main__":
    main()
