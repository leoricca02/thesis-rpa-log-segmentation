"""Orchestrator (Seventh Approach): partial-sharing segmentation pipeline.

Pipeline: Human-in-the-Loop wizard (routine names + execution counts)
-> Phase 0 (clean/serialise) -> Phase 1 (infer SHARING TOPOLOGY: which shared
action belongs to which routines) -> Phase 2 (route, validate the cover,
reassemble, export). Fail-fast checkpoints between phases; one shared
SmartLLMClient across all phases.

Hypothesis change vs the Sixth Approach: H1 (Global Sharing) is relaxed to
H1' (Subset Sharing). A shared action may serve all routines, a subset, or a
single routine (shared across its executions). Full sharing is the special
case subset == all declared routines.

Supervised fallback (option "c"): pass --review to inspect the inferred
topology and approve/abort before Phase 2 spends tokens on it.
"""

from __future__ import annotations

import argparse
import re
import os
from pathlib import Path

from dotenv import load_dotenv

from data_pipeline import load_and_serialize_smartrpa, tag_irrelevant_nodes
from phase1_boundary_reasoning import infer_sharing_topology
from phase2_execution_mapping import generate_segmented_log
from smart_llm_client import SmartLLMClient

load_dotenv()

# Model is env-driven so swapping (e.g. to gemini-2.5-flash-lite for a
# high-RPD stress run) needs no code change.
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
# Default sample log, resolved against the repository root (this file lives
# in src/) so it works regardless of the launch directory.
_DEFAULT_LOG_RELPATH = "data/case_study/caso_studio_trasferte_2exec.csv"
DEFAULT_LOG_FILE = str(Path(__file__).resolve().parent.parent / _DEFAULT_LOG_RELPATH)


def _prompt_positive_int(message: str) -> int:
    """Read a positive integer from stdin, re-prompting until valid."""
    while True:
        raw = input(message).strip()
        try:
            value = int(raw)
        except ValueError:
            print("    Invalid input. Please enter a whole number.")
            continue
        if value <= 0:
            print("    Please enter a positive number.")
            continue
        return value


def run_human_in_the_loop_wizard() -> list[dict[str, object]]:
    """Collect Oracle constraints (routine names + execution counts).

    Returns:
        A list of dicts, each with 'routine_name' and 'executions'.
    """
    print("\n=================================================")
    print("--- HUMAN-IN-THE-LOOP CONFIGURATION WIZARD ---")
    print("=================================================")

    num_routines = _prompt_positive_int(
        "How many distinct business routines are in this log? "
    )
    constraints: list[dict[str, object]] = []
    seen: set[str] = set()
    for i in range(num_routines):
        name = ""
        while not name:
            name = input(
                f" -> Short name for Routine {i + 1} (e.g. 'Process Refund'): "
            ).strip()
            if not name:
                print("    Name cannot be empty.")
            elif name.lower() in seen:
                print("    That name is already used; names must be distinct.")
                name = ""
        seen.add(name.lower())
        execs = _prompt_positive_int(f" -> How many times was '{name}' executed? ")
        constraints.append({"routine_name": name, "executions": execs})
    return constraints


def _describe_action(narrative: str) -> str:
    """Turn a serialised node narrative into a short, human-readable label.

    Example input:
      "[APP: Chrome] action: click | browser_url: https://erp/finance-hub |
       tag_type: button | tag_innerText: Open Finance Workspace"
    Example output:
      "click \"Open Finance Workspace\"  (Chrome, finance-hub)"

    Falls back gracefully if the narrative does not contain the usual fields,
    so it never crashes on an unexpected schema.
    """
    fields: dict[str, str] = {}
    app = ""
    app_match = re.search(r"\[APP:\s*([^\]]+)\]", narrative)
    if app_match:
        app = app_match.group(1).strip()
    for part in narrative.split("|"):
        if ":" in part:
            key, _, val = part.partition(":")
            fields[key.strip().lower()] = val.strip()

    action = fields.get("action", "")
    label = fields.get("tag_innertext", "") or fields.get("clipboard_content", "")
    url = fields.get("browser_url", "")
    # Use the last path segment of the URL as a friendly "where".
    where = ""
    if url:
        where = url.rstrip("/").split("/")[-1] or url

    pieces: list[str] = []
    if action:
        pieces.append(action)
    if label:
        pieces.append(f'"{label}"')
    desc = " ".join(pieces) if pieces else (narrative[:60] + "...")

    context_bits = [b for b in (app, where) if b]
    context = f"  ({', '.join(context_bits)})" if context_bits else ""
    return f"{desc}{context}"


def _print_action_card(
    n: int, action: dict, narrative_by_id: dict[int, str], n_routines: int
) -> None:
    """Print one shared action in a friendly, numbered card."""
    nid = int(action["node_id"])
    desc = _describe_action(narrative_by_id.get(nid, ""))
    subset = list(action["shared_with"])
    if len(subset) == n_routines:
        scope = "ALL routines"
    else:
        scope = f"{len(subset)} of {n_routines} routines"
    print(f"\n  [{n}] {desc}")
    print(f"      currently shared with {scope}:")
    for name in subset:
        print(f"          - {name}")
    if action.get("justification"):
        print(f"      why the system thinks so: {action['justification']}")


def _edit_one_action(
    action: dict, all_routines: list[str], narrative_by_id: dict[int, str]
) -> bool:
    """Interactively edit a single action's routine subset.

    Returns:
        True if the action should remain shared, False if the operator chose to
        reclassify it as a normal (non-shared) business step.
    """
    while True:
        nid = int(action["node_id"])
        print(f"\n  Editing: {_describe_action(narrative_by_id.get(nid, ''))}")
        print("  This action is currently shared with:")
        for name in all_routines:
            mark = "[x]" if name in action["shared_with"] else "[ ]"
            print(f"      {mark} {name}")
        print(
            "\n  Type the NAME of a routine to toggle it on/off,\n"
            "  'none' to mark this action as NOT shared (make it a normal step),\n"
            "  or 'done' to finish editing this action."
        )
        choice = input("  > ").strip()
        low = choice.lower()
        if low == "done":
            if not action["shared_with"]:
                print(
                    "  This action now has no routines. It will be treated as a "
                    "normal (non-shared) step."
                )
                return False
            return True
        if low == "none":
            action["shared_with"] = []
            print("  Marked as NOT shared.")
            return False
        # Toggle a routine by (case-insensitive) name match.
        match = next((r for r in all_routines if r.lower() == low), None)
        if match is None:
            print(
                f"  '{choice}' is not one of the declared routines. "
                "Please type a routine name exactly as listed above."
            )
            continue
        if match in action["shared_with"]:
            action["shared_with"].remove(match)
            print(f"  Removed '{match}'.")
        else:
            action["shared_with"].append(match)
            print(f"  Added '{match}'.")


def _review_topology(
    topology: list[dict], all_routines: list[str], narrative_by_id: dict[int, str]
) -> list[dict] | None:
    """Supervised fallback: let the operator inspect AND edit the topology.

    Shows each inferred shared action in plain language (what it does, where,
    which routines depend on it), then lets the operator approve as-is, edit
    any action's routine subset, reclassify an action as non-shared, or abort.

    Args:
        topology: The Phase-1 inferred topology (mutated in place if edited).
        all_routines: Declared routine names, in order.
        narrative_by_id: node_id -> serialised narrative, for friendly labels.

    Returns:
        The (possibly edited) topology to use for Phase 2, or None if the
        operator chose to abort the run.
    """
    print("\n=================================================")
    print("--- REVIEW: SHARED ACTIONS FOUND IN THE LOG ---")
    print("=================================================")
    print(
        "A 'shared action' is a step performed once that several executions\n"
        "(or several routines) rely on — for example logging in, opening a\n"
        "workspace, or logging out. The system has guessed which routines\n"
        "depend on each shared action. Please review the guesses below."
    )

    while True:
        n_routines = len(all_routines)
        if not topology:
            print("\n  The system found NO shared actions in this log.")
        for i, action in enumerate(topology, 1):
            _print_action_card(i, action, narrative_by_id, n_routines)

        print("\n-------------------------------------------------")
        print("  Options:")
        print("    [number]  edit that action (change which routines use it)")
        print("    accept    use these shared actions as shown and continue")
        print("    abort     stop without producing a result")
        choice = input("  > ").strip().lower()

        if choice == "accept":
            # Drop any actions the operator emptied out (now non-shared).
            kept = [a for a in topology if a["shared_with"]]
            removed = len(topology) - len(kept)
            if removed:
                print(
                    f"\n  {removed} action(s) you marked as NOT shared will be "
                    "routed as normal steps."
                )
            return kept
        if choice == "abort":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(topology):
            _edit_one_action(
                topology[int(choice) - 1], all_routines, narrative_by_id
            )
        else:
            print(
                "  Please type a listed number, 'accept', or 'abort'."
            )


def run_segmentation_pipeline(
    csv_path: str, review: bool = False, relevance_filter: bool = True
) -> bool:
    """Execute the full pipeline against one log file.

    Args:
        csv_path: Path to the input CSV.
        review: If True, pause after Phase 1 for human approval of the
            inferred sharing topology (supervised fallback).
        relevance_filter: If True (default), run Phase 0A-2 relevance tagging,
            diverting events unrelated to the declared routines to a Noise
            sheet. Disable to reproduce the pre-upgrade behaviour.

    Returns:
        True if the pipeline completed and wrote output, else False.
    """
    print("\n=== STARTING PARTIAL-SHARING SEGMENTATION PIPELINE (7th) ===")
    client = SmartLLMClient()

    routine_constraints = run_human_in_the_loop_wizard()

    try:
        df = load_and_serialize_smartrpa(csv_path, MODEL_NAME, client)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[!] ABORTING: could not load/serialise log: {exc}")
        return False

    if relevance_filter:
        routine_names = [str(r["routine_name"]) for r in routine_constraints]
        df = tag_irrelevant_nodes(df, routine_names, MODEL_NAME, client)

    topology = infer_sharing_topology(df, routine_constraints, MODEL_NAME, client)
    if topology is None:
        print("\n[!] ABORTING: Phase 1 could not infer the sharing topology.")
        return False

    if review:
        narrative_by_id = dict(
            zip(df["node_id"].astype(int), df["llm_narrative"].astype(str))
        )
        all_routines = [str(r["routine_name"]) for r in routine_constraints]
        topology = _review_topology(topology, all_routines, narrative_by_id)
        if topology is None:
            print("\n[!] ABORTING: review cancelled by the operator.")
            return False

    success = generate_segmented_log(
        df, topology, routine_constraints, MODEL_NAME, client=client
    )
    if not success:
        print("\n[!] ABORTING: Phase 2 could not construct the execution mapping.")
        return False

    print("\n=== PIPELINE COMPLETE ===")
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RPA UI-log segmentation pipeline (partial sharing, 7th)."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=DEFAULT_LOG_FILE,
        help=f"Path to the input CSV (default: {_DEFAULT_LOG_RELPATH}).",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Pause after Phase 1 to approve the inferred sharing topology.",
    )
    parser.add_argument(
        "--no-relevance-filter",
        action="store_true",
        help="Disable Phase 0A-2 relevance tagging (reproduces old behaviour).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_segmentation_pipeline(
        args.csv_path,
        review=args.review,
        relevance_filter=not args.no_relevance_filter,
    )
