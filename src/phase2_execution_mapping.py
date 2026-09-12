"""Phase 2: route unshared nodes into executions and reassemble traces.

Takes the nodes left after Phase 1 removed the globally shared boundaries and
asks the LLM to distribute them into the exact execution buckets the human
Oracle declared. The routing is then validated (hard coverage check + soft
Oracle/semantic checks), the shared boundaries are re-attached to every trace,
and a colour-coded Excel workbook plus a JSON audit trail are written.
"""

from __future__ import annotations

import colorsys
import json
from typing import Any

import pandas as pd

from smart_llm_client import SmartLLMClient

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
            a legible mid band (0.35–0.85) rather than washing out to white.

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
    """Deterministically map (routine, execution) to a stable shade.

    Same input always yields the same colour, so figures are reproducible.
    """
    base = _BASE_COLORS[routine_index % len(_BASE_COLORS)]
    if exec_total <= 1:
        return adjust_color_lightness(base, 1.0)
    # Spread executions across a 0.7–1.25 lightness band.
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
events into distinct robot executions. The globally shared actions (login,
logout, etc.) have already been removed; you only route the remaining
routine-specific nodes.

{blocks_instruction}

CRITICAL RULES
1. Every listed node id MUST be assigned to exactly ONE execution block.
2. Use ONLY the node ids provided. Never invent ids.
3. Produce exactly the number of executions requested per routine — no more,
   no fewer.
4. Use the routine NAME as a semantic anchor: decide which routine a node
   belongs to from its meaning (URL section, button text, typed/pasted
   payload), then which execution it belongs to from payload continuity and
   chronological flow.
5. For payload-less actions (e.g. a generic 'Submit' click), attach them to
   the execution whose preceding payload they logically complete.

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


def _validate_coverage(
    routing_plan: list[dict[str, Any]], expected_ids: set[int]
) -> tuple[bool, set[int], set[int]]:
    """Hard check: every expected node used exactly once.

    Returns:
        (ok, missing, duplicates).
    """
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


def _check_oracle_counts(
    routing_plan: list[dict[str, Any]], routine_constraints: list[dict[str, Any]]
) -> list[str]:
    """Soft check: did the LLM produce the requested executions per routine?"""
    produced: dict[str, int] = {}
    for block in routing_plan:
        name = str(block.get("routine_name", "")).strip()
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
    extra = set(produced) - {str(r["routine_name"]).strip() for r in routine_constraints}
    for name in sorted(extra):
        warnings.append(f"Unexpected routine name from LLM: '{name}'.")
    return warnings


def generate_segmented_log(
    df: pd.DataFrame,
    shared_indices: list[int],
    routine_constraints: list[dict[str, Any]],
    model_name: str,
    output_file: str = "Final_Segmented_Master_Log.xlsx",
    client: SmartLLMClient | None = None,
) -> bool:
    """Route unshared nodes, validate, reassemble, and export.

    Args:
        df: DataFrame with 'node_id' and 'llm_narrative' columns.
        shared_indices: Globally shared node ids from Phase 1.
        routine_constraints: Oracle constraints, each a dict with
            'routine_name' and 'executions'.
        model_name: Gemini model identifier.
        output_file: Destination .xlsx path.
        client: Optional shared SmartLLMClient.

    Returns:
        True on success (workbook written). False on a hard failure
        (coverage violation or API/parse/export error). Soft issues
        (Oracle-count mismatch) print warnings but still produce output.
    """
    print("\n--- Phase 2: LLM Execution Mapping ---")
    client = client or SmartLLMClient()

    df = df.set_index("node_id", drop=False)
    shared_set = set(int(i) for i in shared_indices)

    # Relevance-aware noise bucket (Phase 0A-2). Nodes tagged as unrelated to
    # the declared routines are NEVER sent to the router and are NOT expected in
    # coverage; they are diverted to a reviewable 'Noise' sheet. If the column
    # is absent (older callers) or all-False (clean log), this is a no-op and
    # behaviour is identical to before.
    if "is_probable_noise" in df.columns:
        noise_mask = df["is_probable_noise"].fillna(False).astype(bool)
    else:
        noise_mask = pd.Series(False, index=df.index)
    noise_ids = set(int(i) for i in df.loc[noise_mask, "node_id"].tolist())
    # A shared boundary node is never noise, even if mis-tagged.
    noise_ids -= shared_set
    if noise_ids:
        print(
            f"[*] {len(noise_ids)} pre-tagged noise node(s) diverted to the "
            "Noise sheet (not routed)."
        )

    df_unshared = df[
        ~df["node_id"].isin(shared_set) & ~df["node_id"].isin(noise_ids)
    ]
    expected_ids = set(int(i) for i in df_unshared["node_id"].tolist())

    log_sequence = "".join(
        f"Node {int(nid)}: {nar}\n"
        for nid, nar in zip(
            df_unshared["node_id"], df_unshared["llm_narrative"].astype(str)
        )
    )
    system_prompt = _build_routing_prompt(routine_constraints)
    user_prompt = (
        "Map the following nodes into their respective executions. Return the "
        f"JSON array:\n\n{log_sequence}"
    )

    try:
        routing_plan = client.generate_content(
            model_name, system_prompt, user_prompt, _SCHEMA, thinking_budget=512
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[!] CRITICAL: Phase 2 API call failed.\n    Details: {exc}")
        return False

    if not isinstance(routing_plan, list) or not routing_plan:
        print("[!] CRITICAL: Phase 2 returned no execution blocks.")
        return False
    print(f"[*] LLM produced {len(routing_plan)} execution block(s).")

    ok, missing, duplicates = _validate_coverage(routing_plan, expected_ids)
    # Duplicates are always fatal: a node in two executions is real corruption.
    if duplicates:
        print(
            f"[!] CRITICAL: {len(duplicates)} node(s) assigned to multiple "
            f"blocks: {duplicates}"
        )
        print("[!] Halting Phase 2 to prevent silent data corruption.")
        return False

    # Missing nodes are nodes the router declined to place. Rather than abort,
    # divert them to the reviewable Noise sheet (alongside any pre-tagged noise)
    # and warn loudly. Nothing is lost: every missing node is preserved in the
    # Noise sheet for human inspection. This is the noise-bucket relaxation —
    # on a clean log 'missing' is empty and behaviour is unchanged.
    if missing:
        print(
            f"[SOFT WARNING] {len(missing)} node(s) were not routed by the LLM "
            f"and are diverted to the Noise sheet for review: {sorted(missing)}"
        )
        noise_ids |= set(int(i) for i in missing)

    count_warnings = _check_oracle_counts(routing_plan, routine_constraints)
    for warning in count_warnings:
        print(f"[SOFT WARNING] {warning}")

    final_master_df, color_map = _reassemble_traces(
        df, routing_plan, sorted(shared_set)
    )

    noise_df = df[df["node_id"].isin(noise_ids)].copy() if noise_ids else None
    _export_workbook(final_master_df, output_file, noise_df)
    _write_audit_trail(routing_plan, output_file, sorted(noise_ids))
    print(f"[SUCCESS] Segmented log written to: {output_file}")
    if noise_ids:
        print(
            f"[*] {len(noise_ids)} node(s) placed in the Noise sheet for review."
        )
    return True


def _reassemble_traces(
    df: pd.DataFrame,
    routing_plan: list[dict[str, Any]],
    shared_sorted: list[int],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Attach shared boundaries to each routed trace and stack into one frame."""
    routine_index: dict[str, int] = {}
    exec_totals: dict[str, int] = {}
    for block in routing_plan:
        name = str(block.get("routine_name", "Routine")).strip() or "Routine"
        exec_totals[name] = exec_totals.get(name, 0) + 1

    records: list[pd.DataFrame] = []
    color_map: dict[str, str] = {}
    for block in routing_plan:
        name = str(block.get("routine_name", "Routine")).strip() or "Routine"
        if name not in routine_index:
            routine_index[name] = len(routine_index)
        exec_idx = int(block.get("execution_index", len(color_map) + 1))
        trace_id = f"{name}_exec{exec_idx}"

        specific = [int(i) for i in block.get("assigned_node_indices", [])]
        ordered = sorted(dict.fromkeys(shared_sorted + specific))
        trace_df = df.loc[ordered].copy()
        trace_df.insert(0, "trace_id", trace_id)
        trace_df.insert(0, "routine_name", name)

        color = _trace_color(routine_index[name], exec_idx - 1, exec_totals[name])
        color_map[trace_id] = color
        trace_df["trace_color"] = color
        records.append(trace_df)

    master = pd.concat(records, ignore_index=True)
    master.drop(
        columns=["llm_narrative", "node_id", "is_probable_noise"],
        inplace=True,
        errors="ignore",
    )
    return master, color_map


def _export_workbook(
    master: pd.DataFrame,
    output_file: str,
    noise_df: pd.DataFrame | None = None,
) -> None:
    """Write the styled, colour-coded workbook with a frozen header row.

    If ``noise_df`` is provided and non-empty, a second 'Noise' sheet lists the
    nodes that were diverted from routing (pre-tagged as unrelated, or declined
    by the router) so a human can review them. None is ever silently lost.
    """
    print(f"[*] Exporting formatted workbook to {output_file} ...")
    data = master.drop(columns=["trace_color"])
    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        data.to_excel(writer, index=False, sheet_name="Segmented_Logs")
        workbook = writer.book
        worksheet = writer.sheets["Segmented_Logs"]

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
        for col_num, name in enumerate(data.columns):
            worksheet.write(0, col_num, name, header_fmt)
            max_len = data[name].astype(str).str.len().max()
            if pd.isna(max_len):
                max_len = 12
            width = min(48, max(12, int(max_len)))
            worksheet.set_column(col_num, col_num, width)

        for row_num, color_hex in enumerate(master["trace_color"]):
            row_fmt = workbook.add_format(
                {"bg_color": color_hex, "font_name": "Arial", "border": 1}
            )
            worksheet.set_row(row_num + 1, cell_format=row_fmt)

        worksheet.freeze_panes(1, 0)
        worksheet.autofilter(0, 0, len(data), len(data.columns) - 1)

        if noise_df is not None and not noise_df.empty:
            noise_out = noise_df.drop(
                columns=["trace_color", "is_probable_noise"], errors="ignore"
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
    routing_plan: list[dict[str, Any]],
    output_file: str,
    noise_ids: list[int] | None = None,
) -> None:
    """Persist the routing plan (with reasoning) and noise ids next to the xlsx."""
    audit_path = output_file.rsplit(".", 1)[0] + "_routing.json"
    payload: dict[str, Any] | list[Any]
    if noise_ids:
        payload = {"routing_plan": routing_plan, "noise_node_ids": noise_ids}
    else:
        payload = routing_plan  # unchanged shape when there is no noise
    try:
        with open(audit_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"[*] Routing audit trail saved to {audit_path}")
    except OSError as exc:
        print(f"[!] Warning: could not write audit trail: {exc}")
