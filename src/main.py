"""Orchestrator for the 3-stage semi-supervised LLM segmentation pipeline.

Runs the Human-in-the-Loop Oracle wizard, then chains Phase 0 (clean/serialise)
-> Phase 1 (shared boundaries) -> Phase 2 (execution routing) with fail-fast
checkpoints. A single SmartLLMClient is shared across all phases so the cache
and telemetry log are read/written once.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from data_pipeline import load_and_serialize_smartrpa, tag_irrelevant_nodes
from phase1_boundary_reasoning import identify_shared_boundaries
from phase2_execution_mapping import generate_segmented_log
from smart_llm_client import SmartLLMClient

load_dotenv()

# Model is env-driven so swapping (e.g. to gemini-3.5-flash-lite for a high-RPD
# stress run) needs no code change.
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
# Default sample log, resolved against the repository root (this file lives in
# src/) so it works regardless of the directory the script is launched from.
_DEFAULT_LOG_RELPATH = "data/04_noise/test_noise_case.csv"
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
    for i in range(num_routines):
        name = ""
        while not name:
            name = input(
                f" -> Short name for Routine {i + 1} (e.g. 'Process Refund'): "
            ).strip()
            if not name:
                print("    Name cannot be empty.")
        execs = _prompt_positive_int(f" -> How many times was '{name}' executed? ")
        constraints.append({"routine_name": name, "executions": execs})
    return constraints


def run_segmentation_pipeline(csv_path: str, relevance_filter: bool = True) -> bool:
    """Execute the full pipeline against one log file.

    Args:
        csv_path: Path to the input CSV.
        relevance_filter: If True (default), run Phase 0A-2 relevance tagging,
            diverting events unrelated to the declared routines to a Noise
            sheet. Disable to reproduce the pre-upgrade behaviour exactly.

    Returns:
        True if the pipeline completed and wrote output, else False.
    """
    print("\n=== STARTING DUAL-PHASE SEGMENTATION PIPELINE ===")
    client = SmartLLMClient()

    routine_constraints = run_human_in_the_loop_wizard()

    try:
        df = load_and_serialize_smartrpa(csv_path, MODEL_NAME, client)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[!] ABORTING: could not load/serialise log: {exc}")
        return False

    # Phase 0A-2: tag events unrelated to the declared routines so Phase 2 can
    # divert them to a Noise sheet instead of mis-routing them. Skipped if
    # disabled; on a clean log it tags nothing and changes nothing.
    if relevance_filter:
        routine_names = [str(r["routine_name"]) for r in routine_constraints]
        df = tag_irrelevant_nodes(df, routine_names, MODEL_NAME, client)

    shared_indices = identify_shared_boundaries(df, MODEL_NAME, client)
    if shared_indices is None:
        print("\n[!] ABORTING: Phase 1 could not resolve shared boundaries.")
        return False

    success = generate_segmented_log(
        df, shared_indices, routine_constraints, MODEL_NAME, client=client
    )
    if not success:
        print("\n[!] ABORTING: Phase 2 could not construct the execution mapping.")
        return False

    print("\n=== PIPELINE COMPLETE ===")
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RPA UI-log segmentation pipeline.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=DEFAULT_LOG_FILE,
        help=f"Path to the input CSV (default: {_DEFAULT_LOG_RELPATH}).",
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
        args.csv_path, relevance_filter=not args.no_relevance_filter
    )
