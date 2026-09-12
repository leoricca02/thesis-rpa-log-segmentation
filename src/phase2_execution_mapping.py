"""Phase 2 (Seventh Approach): route, validate as a COVER, reassemble, export.

The Sixth Approach validated a PARTITION: every node assigned to exactly one
execution, then one global boundary set was prepended/appended to every trace.

Under H1' (Subset Sharing) the unit of validation becomes a COVER:
  HARD INVARIANTS (violation -> abort, no output):
    P1. Every NON-shared node is assigned to exactly one execution block
        (still a partition over the routable nodes).
    P2. Every block's routine name is one of the declared routine names
        (an invented name would silently detach traces from the topology).
    P3. After reassembly, every node of the log appears in at least one
        trace, and a shared node appears in exactly the traces whose routine
        is in its subset. Nothing is ever dropped or smuggled.
  SOFT CHECKS (warning, output still produced):
    S1. The number of executions per routine matches the Oracle.
    S2. Every routine receives at least one shared action (a routine with no
        setup at all is suspicious and worth operator review).
    S3. Duplicate (routine, execution_index) pairs are de-collided with a
        suffix and reported.
"""

from __future__ import annotations

import colorsys
import json
from typing import Any

import pandas as pd

from smart_llm_client import SmartLLMClient

_THINKING_BUDGET = 512

# Distinct base hues per routine family; executions vary lightness within hue.
_BASE_COLORS = (
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#17BECF",
)


def adjust_color_lightness(hex_color: str, amount: float = 1.0) -> str:
    """Return a lighter/darker shade of ``hex_color``.

    Args:
        hex_color: A '#RRGGBB' string.
        amount: Multiplier on the lightness channel; clamped so output stays in
            a legible mid band (0.35-0.85) rather than washing out to white.

    Returns:
        A '#RRGGBB' string (upper-case).
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    new_l = min(0.85, max(0.35, l * amount))
    nr, ng, nb = colorsys.hls_to_rgb(h, new_l, s)
    return f"#{int(nr * 255):02X}{int(ng * 255):02X}{int(nb * 255):02X}"


def _trace_color(routine_index: int, exec_index: int, exec_total: int) -> str:
    """Deterministically map (routine, execution) to a stable shade."""
    base = _BASE_COLORS[routine_index % len(_BASE_COLORS)]
    if exec_total <= 1:
        return adjust_color_lightness(base, 1.0)
    span = 0.7 + 0.55 * (exec_index / max(1, exec_total - 1))
    return adjust_color_lightness(base, span)


def _build_routing_prompt(routine_constraints: list[dict[str, Any]]) -> str:
    """Compose the system prompt, embedding the Oracle execution buckets."""
    blocks = ["You must distribute the nodes into EXACTLY these execution blocks:"]
    for r in routine_constraints:
        blocks.append(
            f"- Routine '{r['routine_name']}': {int(r['executions'])} distinct "
            f"execution(s)."
        )
    blocks_instruction = "\n".join(blocks)

    return f"""\
You are an expert RPA process-mining engine that untangles interleaved UI
events into distinct robot executions. The shared actions (authentication,
module openers, teardown, etc.) have already been removed; you only route the
remaining routine-specific nodes.

{blocks_instruction}

CRITICAL RULES
1. Every listed node id MUST be assigned to exactly ONE execution block.
2. Use ONLY the node ids provided. Never invent ids.
3. Produce exactly the number of executions requested per routine — no more,
   no fewer.
4. Use the routine NAME as a semantic anchor: decide which routine a node
   belongs to from its meaning (the area/section it touches, button text,
   typed/pasted payload), then which execution it belongs to from payload
   continuity and chronological flow.
5. For payload-less actions, attach them to the execution whose preceding
   payload they logically complete.

OUTPUT
For each execution return an object with: routine_name (exactly one of the
names above), execution_index (1-based within that routine), a short reasoning
string, and assigned_node_indices (array of integer node ids).
"""


_SCHEMA: dict[str, Any] = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "routine_name": {"type": "STRING"},
            "execution_index": {"type": "INTEGER"},
            "reasoning": {"type": "STRING"},
            "assigned_node_indices": {
                "type": "ARRAY",
                "items": {"type": "INTEGER"},
            },
        },
        "required": [
            "routine_name",
            "execution_index",
            "assigned_node_indices",
        ],
    },
}


# ----------------------------------------------------------------- validation
def _validate_partition(
    routing_plan: list[dict[str, Any]], expected_ids: set[int]
) -> tuple[bool, set[int], set[int]]:
    """P1: every routable node used exactly once. Returns (ok, missing, dupes)."""
    seen: set[int] = set()
    duplicates: set[int] = set()
    for block in routing_plan:
        for idx in block.get("assigned_node_indices", []):
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                continue
            if idx in seen:
                duplicates.add(idx)
            seen.add(idx)
    missing = expected_ids - seen
    return (not missing and not duplicates), missing, duplicates


def _canonical_block_names(
    routing_plan: list[dict[str, Any]], declared: dict[str, str]
) -> list[str] | None:
    """P2: map each block's routine name onto a declared name (or fail).

    Returns the list of canonical names (parallel to routing_plan), or None if
    any block names a routine the Oracle never declared.
    """
    canonical: list[str] = []
    for block in routing_plan:
        key = str(block.get("routine_name", "")).strip().lower()
        if key not in declared:
            print(
                f"[!] CRITICAL: Phase 2 produced an undeclared routine name: "
                f"{block.get('routine_name')!r}. Declared: "
                f"{sorted(declared.values())}."
            )
            return None
        canonical.append(declared[key])
    return canonical


def _check_oracle_counts(
    block_names: list[str], routine_constraints: list[dict[str, Any]]
) -> list[str]:
    """S1: did the LLM produce the requested executions per routine?"""
    produced: dict[str, int] = {}
    for name in block_names:
        produced[name] = produced.get(name, 0) + 1
    warnings: list[str] = []
    for r in routine_constraints:
        name = str(r["routine_name"]).strip()
        want = int(r["executions"])
        got = produced.get(name, 0)
        if got != want:
            warnings.append(
                f"Routine '{name}': expected {want} execution(s), got {got}."
            )
    return warnings


def _check_topology_reach(
    sharing_topology: list[dict[str, Any]],
    routine_constraints: list[dict[str, Any]],
) -> list[str]:
    """S2: every declared routine should receive at least one shared action."""
    covered: set[str] = set()
    for action in sharing_topology:
        for name in action.get("shared_with", []):
            covered.add(str(name).strip())
    warnings: list[str] = []
    for r in routine_constraints:
        name = str(r["routine_name"]).strip()
        if name not in covered:
            warnings.append(
                f"Routine '{name}' received NO shared actions; verify the "
                "inferred topology (it may be correct, but it is unusual)."
            )
    return warnings


# ------------------------------------------------------------------- pipeline
def generate_segmented_log(
    df: pd.DataFrame,
    sharing_topology: list[dict[str, Any]],
    routine_constraints: list[dict[str, Any]],
    model_name: str,
    output_file: str = "Final_Segmented_Master_Log.xlsx",
    client: SmartLLMClient | None = None,
) -> bool:
    """Route unshared nodes, validate the cover, reassemble, and export.

    Args:
        df: DataFrame with 'node_id' and 'llm_narrative' columns.
        sharing_topology: Phase 1 output — list of dicts with 'node_id',
            'shared_with' (list of declared routine names), 'justification'.
        routine_constraints: Oracle constraints, each a dict with
            'routine_name' and 'executions'.
        model_name: Gemini model identifier.
        output_file: Destination .xlsx path.
        client: Optional shared SmartLLMClient.

    Returns:
        True on success (workbook written). False on a hard failure
        (cover violation, undeclared routine name, API/parse/export error).
        Soft issues print warnings but still produce output.
    """
    print("\n--- Phase 2: LLM Execution Mapping (subset-aware) ---")
    client = client or SmartLLMClient()

    df = df.set_index("node_id", drop=False)
    all_ids = set(int(i) for i in df["node_id"].tolist())
    shared_ids = {int(a["node_id"]) for a in sharing_topology}
    subset_by_id: dict[int, list[str]] = {
        int(a["node_id"]): [str(n).strip() for n in a["shared_with"]]
        for a in sharing_topology
    }
    declared = {
        str(r["routine_name"]).strip().lower(): str(r["routine_name"]).strip()
        for r in routine_constraints
    }

    # Relevance-aware noise bucket (Phase 0A-2). Nodes tagged as unrelated to
    # the declared routines are withheld from routing and the cover check, and
    # diverted to a reviewable 'Noise' sheet. No-op on clean logs.
    if "is_probable_noise" in df.columns:
        noise_mask = df["is_probable_noise"].fillna(False).astype(bool)
    else:
        noise_mask = pd.Series(False, index=df.index)
    noise_ids = set(int(i) for i in df.loc[noise_mask, "node_id"].tolist())
    noise_ids -= shared_ids  # a shared boundary node is never noise
    if noise_ids:
        print(
            f"[*] {len(noise_ids)} pre-tagged noise node(s) diverted to the "
            "Noise sheet (not routed)."
        )

    df_routable = df[~df["node_id"].isin(shared_ids) & ~df["node_id"].isin(noise_ids)]
    expected_ids = set(int(i) for i in df_routable["node_id"].tolist())
    print(
        f"[*] {len(shared_ids)} shared node(s) withheld; "
        f"{len(expected_ids)} routable node(s) sent to the LLM."
    )

    log_sequence = "".join(
        f"Node {int(nid)}: {nar}\n"
        for nid, nar in zip(
            df_routable["node_id"], df_routable["llm_narrative"].astype(str)
        )
    )
    system_prompt = _build_routing_prompt(routine_constraints)
    user_prompt = (
        "Map the following nodes into their respective executions. Return the "
        f"JSON array:\n\n{log_sequence}"
    )

    try:
        routing_plan = client.generate_content(
            model_name,
            system_prompt,
            user_prompt,
            _SCHEMA,
            thinking_budget=_THINKING_BUDGET,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[!] CRITICAL: Phase 2 API call failed.\n    Details: {exc}")
        return False

    if not isinstance(routing_plan, list) or not routing_plan:
        print("[!] CRITICAL: Phase 2 returned no execution blocks.")
        return False
    print(f"[*] LLM produced {len(routing_plan)} execution block(s).")

    # ---- P1: partition over routable nodes ----
    ok, missing, duplicates = _validate_partition(routing_plan, expected_ids)
    # Duplicates are always fatal: a node in two executions is real corruption.
    if duplicates:
        print(
            f"[!] CRITICAL: {len(duplicates)} node(s) assigned to multiple "
            f"blocks: {duplicates}"
        )
        print("[!] Halting Phase 2 to prevent silent data corruption.")
        return False
    # Missing nodes are diverted to the Noise sheet (soft) rather than aborting.
    # Nothing is lost: every missing node is preserved for human review.
    if missing:
        print(
            f"[SOFT WARNING] {len(missing)} node(s) were not routed by the LLM "
            f"and are diverted to the Noise sheet for review: {sorted(missing)}"
        )
        noise_ids |= set(int(i) for i in missing)

    # ---- P2: block names must be declared ----
    block_names = _canonical_block_names(routing_plan, declared)
    if block_names is None:
        print("[!] Halting Phase 2: traces cannot be tied to the topology.")
        return False

    # ---- S1 / S2 soft checks ----
    for warning in _check_oracle_counts(block_names, routine_constraints):
        print(f"[SOFT WARNING] {warning}")
    for warning in _check_topology_reach(sharing_topology, routine_constraints):
        print(f"[SOFT WARNING] {warning}")

    final_master_df = _reassemble_traces(
        df, routing_plan, block_names, subset_by_id
    )

    # ---- P3: cover check on the reassembled output (noise excluded) ----
    covered = set(int(i) for i in final_master_df["node_id"].tolist())
    uncovered = all_ids - covered - noise_ids
    if uncovered:
        print(
            f"[!] CRITICAL: {len(uncovered)} node(s) appear in NO trace after "
            f"reassembly: {sorted(uncovered)}. A shared node's subset may "
            "reference a routine that produced no executions."
        )
        print("[!] Halting Phase 2 to prevent silent data loss.")
        return False

    noise_df = df[df["node_id"].isin(noise_ids)].copy() if noise_ids else None
    _export_workbook(final_master_df, sharing_topology, df, output_file, noise_df)
    _write_audit_trail(sharing_topology, routing_plan, output_file, sorted(noise_ids))
    print(f"[SUCCESS] Segmented log written to: {output_file}")
    if noise_ids:
        print(f"[*] {len(noise_ids)} node(s) placed in the Noise sheet for review.")
    return True


def _reassemble_traces(
    df: pd.DataFrame,
    routing_plan: list[dict[str, Any]],
    block_names: list[str],
    subset_by_id: dict[int, list[str]],
) -> pd.DataFrame:
    """Attach each shared node to exactly the traces in its subset."""
    routine_index: dict[str, int] = {}
    exec_totals: dict[str, int] = {}
    for name in block_names:
        exec_totals[name] = exec_totals.get(name, 0) + 1

    seen_trace_ids: set[str] = set()
    records: list[pd.DataFrame] = []
    for block, name in zip(routing_plan, block_names):
        if name not in routine_index:
            routine_index[name] = len(routine_index)
        try:
            exec_idx = int(block.get("execution_index", 0))
        except (TypeError, ValueError):
            exec_idx = 0
        if exec_idx <= 0:
            exec_idx = sum(1 for n in block_names[: len(records)] if n == name) + 1

        trace_id = f"{name}_exec{exec_idx}"
        while trace_id in seen_trace_ids:  # S3: de-collide duplicates
            print(f"[SOFT WARNING] Duplicate trace id '{trace_id}'; suffixing.")
            trace_id += "_b"
        seen_trace_ids.add(trace_id)

        specific = []
        for i in block.get("assigned_node_indices", []):
            try:
                specific.append(int(i))
            except (TypeError, ValueError):
                continue
        shared_for_trace = [
            nid for nid, subset in subset_by_id.items() if name in subset
        ]
        ordered = sorted(dict.fromkeys(shared_for_trace + specific))

        trace_df = df.loc[ordered].copy()
        trace_df.insert(0, "trace_id", trace_id)
        trace_df.insert(0, "routine_name", name)
        trace_df["trace_color"] = _trace_color(
            routine_index[name], exec_idx - 1, exec_totals[name]
        )
        records.append(trace_df)

    return pd.concat(records, ignore_index=True)


def _export_workbook(
    master: pd.DataFrame,
    sharing_topology: list[dict[str, Any]],
    df_indexed: pd.DataFrame,
    output_file: str,
    noise_df: pd.DataFrame | None = None,
) -> None:
    """Write the styled workbook: segmented log + sharing-topology (+ noise)."""
    print(f"[*] Exporting formatted workbook to {output_file} ...")
    data = master.drop(
        columns=["trace_color", "llm_narrative", "node_id", "is_probable_noise"],
        errors="ignore",
    )

    topo_rows = []
    for action in sharing_topology:
        nid = int(action["node_id"])
        topo_rows.append(
            {
                "node_id": nid,
                "shared_with": ", ".join(action["shared_with"]),
                "n_routines": len(action["shared_with"]),
                "justification": str(action.get("justification", "")),
                "narrative": str(df_indexed.loc[nid, "llm_narrative"]),
            }
        )
    topo_df = pd.DataFrame(
        topo_rows,
        columns=["node_id", "shared_with", "n_routines", "justification",
                 "narrative"],
    )

    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        data.to_excel(writer, index=False, sheet_name="Segmented_Logs")
        topo_df.to_excel(writer, index=False, sheet_name="Sharing_Topology")
        workbook = writer.book

        header_fmt = workbook.add_format(
            {
                "bold": True,
                "font_name": "Arial",
                "font_color": "#FFFFFF",
                "bg_color": "#404040",
                "border": 1,
                "align": "center",
                "valign": "vcenter",
            }
        )

        ws_main = writer.sheets["Segmented_Logs"]
        for col_num, name in enumerate(data.columns):
            ws_main.write(0, col_num, name, header_fmt)
            max_len = data[name].astype(str).str.len().max()
            if pd.isna(max_len):
                max_len = 12
            ws_main.set_column(col_num, col_num, min(48, max(12, int(max_len))))
        for row_num, color_hex in enumerate(master["trace_color"]):
            row_fmt = workbook.add_format(
                {"bg_color": color_hex, "font_name": "Arial", "border": 1}
            )
            ws_main.set_row(row_num + 1, cell_format=row_fmt)
        ws_main.freeze_panes(1, 0)
        ws_main.autofilter(0, 0, len(data), len(data.columns) - 1)

        ws_topo = writer.sheets["Sharing_Topology"]
        for col_num, name in enumerate(topo_df.columns):
            ws_topo.write(0, col_num, name, header_fmt)
            max_len = topo_df[name].astype(str).str.len().max()
            if pd.isna(max_len):
                max_len = 12
            ws_topo.set_column(col_num, col_num, min(70, max(12, int(max_len))))
        ws_topo.freeze_panes(1, 0)

        if noise_df is not None and not noise_df.empty:
            noise_out = noise_df.drop(
                columns=["trace_color", "llm_narrative", "node_id",
                         "is_probable_noise"],
                errors="ignore",
            )
            noise_out.to_excel(writer, index=False, sheet_name="Noise")
            ws_noise = writer.sheets["Noise"]
            for col_num, name in enumerate(noise_out.columns):
                ws_noise.write(0, col_num, name, header_fmt)
                max_len = noise_out[name].astype(str).str.len().max()
                if pd.isna(max_len):
                    max_len = 12
                ws_noise.set_column(col_num, col_num, min(48, max(12, int(max_len))))
            ws_noise.freeze_panes(1, 0)


def _write_audit_trail(
    sharing_topology: list[dict[str, Any]],
    routing_plan: list[dict[str, Any]],
    output_file: str,
    noise_ids: list[int] | None = None,
) -> None:
    """Persist the topology and routing plan (with reasoning) as JSON."""
    audit_path = output_file.rsplit(".", 1)[0] + "_routing.json"
    payload: dict[str, Any] = {
        "sharing_topology": sharing_topology,
        "routing_plan": routing_plan,
    }
    if noise_ids:
        payload["noise_node_ids"] = noise_ids
    try:
        with open(audit_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"[*] Audit trail (topology + routing) saved to {audit_path}")
    except OSError as exc:
        print(f"[!] Warning: could not write audit trail: {exc}")
