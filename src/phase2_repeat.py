"""Phase-2 repeatability harness (does NOT modify the pipeline).

Companion to ``phase1_repeat.py``. Where that harness measures how stable the
inferred sharing TOPOLOGY is across runs, this one measures how stable the
ROUTING is: given a fixed topology, does Phase 2 place the same event in the
same execution every time?

Experimental design
-------------------
Phase 0 and Phase 1 are run ONCE and their results held fixed. Every repeat then
calls Phase 2 alone, each time with a FRESH throwaway cache so the call is a real
inference rather than a cached reply. Holding the topology fixed is deliberate:
it isolates Phase 2's own variance instead of letting Phase 1's variance leak in
and confound the measurement.

Each repeat calls the production ``generate_segmented_log`` — the same code path
a real run takes, including the P1/P2/P3 validation invariants — writing into a
throwaway directory, and reads back the routing it recorded in the audit JSON.
Runs that hard-abort are counted rather than hidden, since an abort rate is
itself a reliability figure.

Two things are reported. First, STABILITY: for each event, how many runs placed
it in each (routine, execution), and which events vary. Second, and only if a
reference routing is supplied via --truth, ACCURACY per run against it.

The report additionally cross-references unstable events against events whose
serialised narrative is IDENTICAL to that of another event. Under H3 an event
with no distinguishing payload has nothing for the model to reason over, so any
instability should concentrate exactly there. Whether it does is the question
this harness exists to answer.

Usage:
    python phase2_repeat.py <log.csv> <n_runs> "Name1:execs,Name2:execs,..." \
        [--truth <reference_routing.json>]

Example:
    python src\\phase2_repeat.py .\\data\\case_study\\caso_studio_trasferte_2exec.csv 5 ^
      "Travel Authorization:2,Expense Reimbursement:2,Purchase Order Approval:2,Student Grant Disbursement:2" ^
      --truth .\\results\\case_study\\Final_Segmented_Master_Log_routing.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict

from data_pipeline import load_and_serialize_smartrpa, tag_irrelevant_nodes
from phase1_boundary_reasoning import infer_sharing_topology
from phase2_execution_mapping import generate_segmented_log
from smart_llm_client import SmartLLMClient

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")


def _parse_constraints(spec: str) -> list[dict]:
    out = []
    for part in spec.split(","):
        name, _, execs = part.rpartition(":")
        out.append({"routine_name": name.strip(), "executions": int(execs)})
    return out


def _load_reference(path: str) -> dict[int, tuple[str, int]]:
    """Read a routing JSON into {node_id: (routine_name, execution_index)}."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    ref: dict[int, tuple[str, int]] = {}
    for block in data.get("routing_plan", []):
        key = (str(block["routine_name"]), int(block["execution_index"]))
        for nid in block.get("assigned_node_indices", []):
            ref[int(nid)] = key
    return ref


def _duplicate_narrative_nodes(df) -> set[int]:
    """Node ids whose serialised narrative is identical to another node's.

    These are the events H3 says cannot be separated on content, because their
    recorded text carries nothing that distinguishes them.
    """
    buckets: dict[str, list[int]] = defaultdict(list)
    for nid, text in zip(df["node_id"].astype(int), df["llm_narrative"].astype(str)):
        buckets[text].append(int(nid))
    return {n for nodes in buckets.values() if len(nodes) > 1 for n in nodes}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the run-to-run stability of Phase 2 routing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("log_path", help="Path to the input UI log (CSV).")
    parser.add_argument("n_runs", type=int, help="How many times to repeat Phase 2.")
    parser.add_argument(
        "constraints",
        help='Oracle constraints, e.g. "Task A:2,Task B:3".',
    )
    parser.add_argument(
        "--truth",
        default=None,
        metavar="ROUTING_JSON",
        help="Optional reference routing JSON to score each run against.",
    )
    return parser.parse_args()


def main() -> None:
    ns = _parse_args()
    log_path, n_runs, spec = ns.log_path, ns.n_runs, ns.constraints
    truth_path = ns.truth
    if n_runs < 1:
        print("[!] n_runs must be at least 1.")
        sys.exit(1)

    constraints = _parse_constraints(spec)
    routine_names = [c["routine_name"] for c in constraints]

    # ---- Phase 0 + Phase 1: run once, hold fixed -------------------------
    base_client = SmartLLMClient()
    df = load_and_serialize_smartrpa(log_path, MODEL, base_client)
    df = tag_irrelevant_nodes(df, routine_names, MODEL, base_client)
    topology = infer_sharing_topology(df, constraints, MODEL, base_client)
    if topology is None:
        print("[!] Phase 1 failed; cannot measure Phase 2.")
        sys.exit(1)
    print(f"\n[*] Topology fixed for all {n_runs} repeats "
          f"({len(topology)} shared action(s)). Only Phase 2 will vary.\n")

    reference = _load_reference(truth_path) if truth_path else None
    ambiguous = _duplicate_narrative_nodes(df)

    # ---- repeat Phase 2 --------------------------------------------------
    tally: dict[int, dict[tuple[str, int], int]] = defaultdict(lambda: defaultdict(int))
    per_run_accuracy: list[float] = []
    aborts = 0
    valid_runs = 0

    for i in range(n_runs):
        with tempfile.TemporaryDirectory() as tmp:
            client = SmartLLMClient(
                cache_file=os.path.join(tmp, "c.json"),
                token_log_file=os.path.join(tmp, "t.csv"),
            )
            out_xlsx = os.path.join(tmp, "run.xlsx")
            ok = generate_segmented_log(
                df, topology, constraints, MODEL,
                output_file=out_xlsx, client=client,
            )
            if not ok:
                print(f"[run {i + 1}] Phase 2 hard-aborted; counted as a failure.")
                aborts += 1
                continue
            plan = _load_reference(out_xlsx.rsplit(".", 1)[0] + "_routing.json")

        valid_runs += 1
        for nid, key in plan.items():
            tally[nid][key] += 1
        if reference:
            hits = sum(1 for n, k in reference.items() if plan.get(n) == k)
            per_run_accuracy.append(hits / len(reference))

    # ---- report ----------------------------------------------------------
    print("\n\n================= PHASE-2 REPEATABILITY REPORT =================")
    print(f"Log      : {log_path}")
    print(f"Model    : {MODEL}")
    print(f"Runs     : {valid_runs} completed, {aborts} hard-aborted "
          f"(of {n_runs} requested)")
    if not valid_runs:
        print("No completed runs; nothing to report.")
        return

    if reference:
        print(f"Accuracy : " + ", ".join(f"{a * 100:.1f}%" for a in per_run_accuracy))
        perfect = sum(1 for a in per_run_accuracy if a == 1.0)
        print(f"           {perfect}/{valid_runs} run(s) reproduced the reference "
              f"routing exactly")

    unstable = {n: d for n, d in tally.items() if len(d) > 1}
    routed = len(tally)
    print(f"\nRouted events           : {routed}")
    print(f"Perfectly stable        : {routed - len(unstable)} "
          f"({(routed - len(unstable)) * 100 / routed:.1f}%)")
    print(f"Varying across runs     : {len(unstable)}")

    if unstable:
        print("\nEvents the model does not place consistently:\n")
        for nid in sorted(unstable):
            flag = "  [no distinguishing payload]" if nid in ambiguous else ""
            print(f"  Node {nid}{flag}")
            for key, c in sorted(tally[nid].items(), key=lambda x: -x[1]):
                bar = "#" * c + "." * (valid_runs - c)
                mark = ""
                if reference and reference.get(nid) == key:
                    mark = "  <- reference"
                print(f"      {key[0]} exec{key[1]:<3} {c}/{valid_runs} [{bar}]{mark}")
            print()

    # ---- the H3 cross-check ---------------------------------------------
    amb_routed = ambiguous & set(tally)
    dist_routed = set(tally) - ambiguous
    amb_unstable = len(ambiguous & set(unstable))
    dist_unstable = len(dist_routed & set(unstable))
    print("--------------- stability vs. lexical distinguishability ---------------")
    print(f"  events with a distinctive narrative : "
          f"{len(dist_routed) - dist_unstable}/{len(dist_routed)} stable")
    if amb_routed:
        print(f"  events identical to another event  : "
              f"{len(amb_routed) - amb_unstable}/{len(amb_routed)} stable")
    print("\nUnder H3, an event carrying no distinguishing payload gives the model")
    print("nothing to reason over, so instability is expected to concentrate there.")
    print("========================================================================")


if __name__ == "__main__":
    main()
