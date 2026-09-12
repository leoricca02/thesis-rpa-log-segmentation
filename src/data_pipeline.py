"""Data ingestion and Phase 0 (self-cleaning) layer.

Transforms a raw, wide, sparse UI-log CSV into a chronological natural-language
narrative, pruning useless rows (Phase 0A: noise event types) and columns
(Phase 0B: metadata) dynamically via the LLM so the pipeline stays vendor
agnostic. A stable integer ``node_id`` column is attached so that downstream
phases never depend on pandas' positional index.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from smart_llm_client import SmartLLMClient

# Columns the serializer always uses as the sentence anchor; never pruned.
_ANCHOR_COLUMNS = ("application", "event_type")

# Conservative fallback list used only if Phase 0B cannot reach the LLM. These
# are the SmartRPA columns that are metadata by definition across all logs.
_FALLBACK_IGNORE_COLUMNS = (
    "timestamp",
    "user",
    "event_relevance",
    "id",
    "screenshot",
    "window_size",
)

_NOISE_SCHEMA: dict[str, Any] = {"type": "ARRAY", "items": {"type": "STRING"}}
_COLUMN_SCHEMA: dict[str, Any] = {"type": "ARRAY", "items": {"type": "STRING"}}
_RELEVANCE_SCHEMA: dict[str, Any] = {"type": "ARRAY", "items": {"type": "INTEGER"}}

_MAX_SAMPLES_PER_COLUMN = 3
_SAMPLE_TRUNCATE = 60


def _nonempty_str_series(series: pd.Series) -> pd.Series:
    """Return the series as stripped strings with empty/NaN entries dropped."""
    as_str = series.dropna().astype(str).str.strip()
    return as_str[as_str != ""]


def filter_noise_with_llm(
    df: pd.DataFrame, model_name: str, client: SmartLLMClient
) -> pd.DataFrame:
    """Phase 0A: drop event types the LLM classifies as semantic noise.

    Args:
        df: The raw ingested log. Must contain an 'event_type' column.
        model_name: Gemini model identifier.
        client: A shared SmartLLMClient instance.

    Returns:
        A new DataFrame with noise rows removed and a reset index. On any API
        failure the original DataFrame is returned unchanged (fail-open here:
        keeping too many rows is safe; dropping good rows is not).
    """
    print("\n--- Phase 0A: Dynamic Noise Filtration ---")

    if "event_type" not in df.columns:
        print("[!] No 'event_type' column found; skipping noise filtration.")
        return df

    unique_events = df["event_type"].value_counts().to_dict()
    event_summary = "\n".join(
        f"- '{etype}': {count} occurrences" for etype, count in unique_events.items()
    )

    system_prompt = (
        "You are an RPA log analysis engine.\n"
        "You will receive a list of UI event types found in a log, with their "
        "occurrence counts.\n"
        "Classify which event types are semantic NOISE — events that carry no "
        "intentional business meaning and only bloat the log (e.g. mouse "
        "movements, hover events, window focus changes, scroll events, resize "
        "actions).\n"
        "Return ONLY a JSON array of the event-type names to filter out. "
        "If none should be filtered, return an empty array: []."
    )
    user_prompt = (
        "Classify the following event types as noise or meaningful:\n\n"
        f"{event_summary}"
    )

    try:
        noise_events = client.generate_content(
            model_name, system_prompt, user_prompt, _NOISE_SCHEMA
        )
        if not isinstance(noise_events, list):
            noise_events = []
        noise_events = [str(e) for e in noise_events]
    except Exception as exc:  # noqa: BLE001 - fail-open is intentional here
        print(f"[!] Phase 0A failed ({exc}); keeping all rows (zero filtration).")
        return df

    print(f"[*] LLM identified noise types: {noise_events}")
    filtered = df[~df["event_type"].isin(noise_events)].copy()
    filtered = filtered.reset_index(drop=True)
    removed = len(df) - len(filtered)
    print(f"[*] Remaining actionable events: {len(filtered)} (removed {removed}).")
    return filtered


def identify_metadata_columns_with_llm(
    df: pd.DataFrame, model_name: str, client: SmartLLMClient
) -> list[str]:
    """Phase 0B: ask the LLM which columns are pure system metadata.

    Improvement over the original: shows the LLM up to three *distinct,
    non-empty* sample values per column (instead of just the first cell), which
    makes the keep/prune decision far more reliable on sparse logs where the
    first row of a column is often empty.

    Args:
        df: The (noise-filtered) log.
        model_name: Gemini model identifier.
        client: A shared SmartLLMClient instance.

    Returns:
        A list of column names to exclude from the narrative. Falls back to a
        conservative hard-coded list on API failure.
    """
    print("\n--- Phase 0B: Dynamic Feature Selection ---")

    col_summary: list[str] = []
    for col in df.columns:
        values = _nonempty_str_series(df[col])
        if values.empty:
            col_summary.append(f"- '{col}' (samples: <ALWAYS EMPTY>)")
            continue
        distinct = list(dict.fromkeys(values.tolist()))[:_MAX_SAMPLES_PER_COLUMN]
        rendered = ", ".join(f'"{v[:_SAMPLE_TRUNCATE]}"' for v in distinct)
        col_summary.append(f"- '{col}' (samples: {rendered})")

    system_prompt = (
        "You are an expert RPA process-mining data architect.\n"
        "Analyse the schema of an RPA log and identify which columns contain "
        "purely SYSTEM METADATA or NOISE.\n\n"
        "IGNORE (metadata): timestamps/dates, database row IDs or GUIDs, user "
        "IDs or employee names, window dimensions (e.g. 1920x1080), file paths "
        "or screenshot names, relevancy scores.\n\n"
        "KEEP (business context): URLs, clicked button names, typed text, "
        "clipboard content, window titles, XPath selectors, application names.\n\n"
        "Return ONLY a JSON array of the exact column names to ignore."
    )
    user_prompt = (
        "Analyse these columns and return the array of metadata columns to "
        f"ignore:\n\n" + "\n".join(col_summary)
    )

    try:
        ignore_cols = client.generate_content(
            model_name, system_prompt, user_prompt, _COLUMN_SCHEMA
        )
        if not isinstance(ignore_cols, list):
            ignore_cols = []
        # Keep only names that actually exist, and never prune the anchors.
        ignore_cols = [
            str(c)
            for c in ignore_cols
            if str(c) in df.columns and str(c) not in _ANCHOR_COLUMNS
        ]
    except Exception as exc:  # noqa: BLE001 - fall back to a safe baseline
        print(f"[!] Phase 0B failed ({exc}); using fallback metadata list.")
        return [c for c in _FALLBACK_IGNORE_COLUMNS if c in df.columns]

    print(f"[*] LLM flagged {len(ignore_cols)} metadata columns: {ignore_cols}")
    return ignore_cols


def _serialize_row(
    row: pd.Series, columns: list[str], ignore_cols: set[str]
) -> str:
    """Render a single row as an anchored natural-language sentence."""
    app_name = str(row.get("application", "Unknown_App"))
    event_type = str(row.get("event_type", "Unknown_Event"))
    parts = [f"[APP: {app_name}] action: {event_type}"]
    for col in columns:
        if col in ignore_cols or col in _ANCHOR_COLUMNS:
            continue
        val = row[col]
        if pd.notna(val) and str(val).strip() != "":
            parts.append(f"{col}: {str(val).strip()}")
    return " | ".join(parts)



def _is_row_identifier(
    df: pd.DataFrame, col: str, coverage: float = 0.9, distinctness: float = 0.9
) -> bool:
    """True if ``col`` behaves as a per-row identifier.

    A column populated on nearly every row AND holding a near-unique value on
    each (timestamps, GUIDs, screenshot paths, sequence numbers) separates every
    pair of events trivially. Restoring one would defeat Phase 0B entirely,
    inflate every prompt, and add no semantic signal — so such columns are never
    restored, and are also excluded from the reference the guard compares against.
    """
    values = _nonempty_str_series(df[col])
    if len(values) < coverage * len(df):
        return False
    return values.nunique() >= distinctness * len(values)


def guard_against_collapse(df: pd.DataFrame, ignore_cols: list[str]) -> list[str]:
    """Phase 0B-2: undo any part of Phase 0B's pruning that destroys evidence.

    Phase 0B decides which columns are metadata but never checks what the
    decision COSTS. Dropping a column can make two events that were previously
    distinguishable serialise to identical text; when that happens the pipeline
    has destroyed exactly the evidence H3 says Phase 2 depends on, and Phase 2
    is left with no basis for separating those events beyond guessing.

    This guard measures that collapse and restores the minimum set of columns
    needed to undo it. It is schema-agnostic by construction: it never names a
    column, an application, or a routine, and decides purely on whether pruning
    destroys discriminability in the log actually being processed. It makes no
    API call.

    Args:
        df: The noise-filtered log.
        ignore_cols: The columns Phase 0B proposed to prune.

    Returns:
        A possibly shorter ignore-list.
    """
    print("\n--- Phase 0B-2: Collapse Guard ---")
    columns = list(df.columns)
    ignore = set(ignore_cols)

    identifiers = {c for c in columns if _is_row_identifier(df, c)}
    if identifiers:
        print(f"[*] Per-row identifier column(s) {sorted(identifiers)} will never "
              f"be restored (they separate every event trivially).")

    def narratives(ign: set[str]) -> list[str]:
        return [_serialize_row(row, columns, ign) for _, row in df.iterrows()]

    # Reference: what the log CAN distinguish, ignoring trivial identifiers.
    reference = narratives(identifiers)

    def collapsed(ign: set[str]) -> list[list[int]]:
        """Groups of events identical after pruning but distinct in the source."""
        buckets: dict[str, list[int]] = {}
        for idx, text in enumerate(narratives(ign)):
            buckets.setdefault(text, []).append(idx)
        return [
            nodes for nodes in buckets.values()
            if len(nodes) > 1 and len({reference[i] for i in nodes}) > 1
        ]

    groups = collapsed(ignore)
    if not groups:
        print("[*] Pruning collapses no distinguishable events; keeping the "
              "Phase 0B selection unchanged.")
        return sorted(ignore)

    total = sum(len(g) for g in groups)
    print(f"[!] Pruning collapsed {total} distinguishable event(s) into "
          f"{len(groups)} indistinguishable group(s). Restoring evidence.")

    def payload_likeness(col: str) -> tuple[float, int]:
        """Rank a candidate by how much it looks like a business payload.

        A real payload (a record id, a filename) is SPARSE and near-unique where
        present: it appears only on the events that carry it. Incidental metadata
        that merely happens to vary (a window size, a zoom factor) is DENSE and
        low-cardinality. Preferring the former keeps the restored narrative
        meaningful rather than merely distinguishable.
        """
        values = _nonempty_str_series(df[col])
        if values.empty:
            return (0.0, 0)
        return (values.nunique() / len(values), -len(values))

    restored: list[str] = []
    while groups:
        best, best_key = None, None
        for col in sorted(ignore):
            if col in identifiers:
                continue
            gain = len(groups) - len(collapsed(ignore - {col}))
            if gain <= 0:
                continue
            key = (gain,) + payload_likeness(col)
            if best_key is None or key > best_key:
                best, best_key = col, key
        if best is None:
            print(f"    [!] {len(groups)} group(s) cannot be resolved by any "
                  f"prunable column. Those events are indistinguishable in the "
                  f"source log itself — a genuine H3 limit, not a pruning error.")
            for nodes in groups[:5]:
                print(f"        nodes {nodes}")
            break
        ignore.discard(best)
        restored.append(best)
        groups = collapsed(ignore)
        print(f"    -> restored '{best}' (resolved {best_key[0]} group(s); "
              f"{len(groups)} remaining)")

    if restored:
        print(f"[*] Restored {len(restored)} column(s) to preserve H3 evidence: "
              f"{restored}")
    return sorted(ignore)


def tag_irrelevant_nodes(
    df: pd.DataFrame,
    routine_names: list[str],
    model_name: str,
    client: SmartLLMClient,
) -> pd.DataFrame:
    """Phase 0A-2: tag nodes semantically unrelated to the declared routines.

    Relevance-aware complement to ``filter_noise_with_llm`` (which only removes
    noise *event types*). Some noise survives type-based filtering because it
    shares an event type with real actions (a junk clipboard copy/paste, a
    click in an unrelated app, a dead click). Given the human-declared routine
    names as the relevance anchor, the LLM marks which serialised events belong
    to NONE of those routines. No application or keyword is hardcoded: relevance
    is judged relative to the declared routines.

    Tagged nodes are not dropped — they are marked ``is_probable_noise=True`` so
    Phase 2 can divert them into a reviewable noise bucket instead of being
    forced into a business routine. Fail-open: on API failure nothing is tagged.
    """
    print("\n--- Phase 0A-2: Relevance-Aware Noise Tagging ---")
    df = df.copy()
    df["is_probable_noise"] = False

    if not routine_names:
        print("[*] No declared routines provided; skipping relevance tagging.")
        return df

    log_sequence = "".join(
        f"Node {int(nid)}: {nar}\n"
        for nid, nar in zip(df["node_id"], df["llm_narrative"].astype(str))
    )
    routines_block = "\n".join(f"- {name}" for name in routine_names)
    system_prompt = (
        "You are an RPA log-analysis engine. A human operator is segmenting a "
        "UI log into executions of these declared business routines:\n"
        f"{routines_block}\n\n"
        "Some events in the log do NOT belong to ANY of these routines. They "
        "are contextual noise that survived basic filtering, such as:\n"
        "- actions performed in an application unrelated to the routines;\n"
        "- dead interactions that accomplish nothing;\n"
        "- copying or pasting content unrelated to the routines.\n\n"
        "Judge relevance ONLY relative to the declared routines above — do not "
        "assume any application is inherently noise. Return ONLY a JSON array "
        "of the integer node ids that belong to NONE of the declared routines. "
        "If every event is plausibly part of some routine, return []."
    )
    user_prompt = (
        "Identify the node ids that are unrelated to the declared routines:\n\n"
        f"{log_sequence}"
    )

    try:
        flagged = client.generate_content(
            model_name, system_prompt, user_prompt, _RELEVANCE_SCHEMA
        )
        if not isinstance(flagged, list):
            flagged = []
    except Exception as exc:  # noqa: BLE001 - fail-open reproduces old behaviour
        print(f"[!] Phase 0A-2 failed ({exc}); tagging nothing as noise.")
        return df

    valid_ids = set(int(n) for n in df["node_id"].tolist())
    flagged_ids: set[int] = set()
    for value in flagged:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        if idx in valid_ids:
            flagged_ids.add(idx)

    # Payload guard: never divert a node whose narrative carries a DISTINCTIVE
    # value (appears on <=2 nodes). Real payloads (record ids, pasted values)
    # are rare-by-definition; generic repeated labels are not. Protects signal
    # from false-positive noise tagging, schema-agnostically.
    protected = _distinctive_value_nodes(df)
    rescued = flagged_ids & protected
    if rescued:
        print(f"[*] Payload guard rescued {len(rescued)} node(s) from noise: "
              f"{sorted(rescued)}")
    flagged_ids -= rescued

    df.loc[df["node_id"].isin(flagged_ids), "is_probable_noise"] = True
    print(
        f"[*] Tagged {len(flagged_ids)} node(s) as probable noise "
        f"(unrelated to the {len(routine_names)} declared routines)."
    )
    if flagged_ids:
        print("    These will be diverted to a 'Noise' sheet, not routed.")
    return df


def _distinctive_value_nodes(df: pd.DataFrame, rarity: int = 2) -> set[int]:
    """Return node_ids whose narrative contains a token appearing on <= rarity
    nodes (a likely payload). Schema-agnostic: tokenises the narrative itself.
    """
    import re as _re
    token_nodes: dict[str, set[int]] = {}
    narr = dict(zip(df["node_id"].astype(int), df["llm_narrative"].astype(str)))
    for nid, text in narr.items():
        for tok in _re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-./@]{2,}", text):
            if any(ch.isdigit() for ch in tok):
                token_nodes.setdefault(tok, set()).add(nid)
    protected: set[int] = set()
    for tok, nodes in token_nodes.items():
        if len(nodes) <= rarity:
            protected |= nodes
    return protected


def load_and_serialize_smartrpa(
    file_path: str, model_name: str, client: SmartLLMClient | None = None
) -> pd.DataFrame:
    """Ingest, clean (Phase 0A/0B), and serialise a UI log.

    Args:
        file_path: Path to the semicolon-delimited SmartRPA CSV.
        model_name: Gemini model identifier.
        client: Optional shared SmartLLMClient. A new one is created if omitted.

    Returns:
        A DataFrame with a contiguous RangeIndex, a stable integer ``node_id``
        column, and an ``llm_narrative`` column. The DataFrame index equals
        ``node_id`` for every row, which is the contract downstream phases rely
        on.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        ValueError: If the CSV has no usable rows.
    """
    print(f"[*] Loading UI log from: {file_path}")
    client = client or SmartLLMClient()

    df = pd.read_csv(file_path, sep=";", dtype=str, keep_default_na=True)
    if df.empty:
        raise ValueError(f"Log '{file_path}' contains no data rows.")

    # Chronological ordering only if a timestamp column exists; otherwise keep
    # file order (the pipeline must not crash on non-SmartRPA schemas).
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(by="timestamp", kind="stable").reset_index(drop=True)
    else:
        print("[!] No 'timestamp' column; preserving original row order.")
        df = df.reset_index(drop=True)

    df = filter_noise_with_llm(df, model_name, client)
    if df.empty:
        raise ValueError("All rows were filtered as noise; nothing to segment.")

    ignore_cols = identify_metadata_columns_with_llm(df, model_name, client)
    ignore_cols = set(guard_against_collapse(df, ignore_cols))

    columns = list(df.columns)
    kept = [c for c in columns if c not in ignore_cols]
    print(f"[*] Columns kept in narrative ({len(kept)}): {kept}")

    narratives = [
        _serialize_row(row, columns, ignore_cols) for _, row in df.iterrows()
    ]

    # Payload-less warning: if many rows serialise to the same anchor-only text,
    # the routines may be indistinguishable (the H3 limitation). Flag it so the
    # operator knows Phase 2 is at risk, rather than discovering it silently.
    distinct_narratives = len(set(narratives))
    if narratives and distinct_narratives < max(2, len(narratives) // 2):
        print(
            f"[!] Warning: only {distinct_narratives} distinct narratives across "
            f"{len(narratives)} events — routines may be hard to distinguish "
            f"(payload-less interleaving risk)."
        )

    df = df.reset_index(drop=True)
    df["node_id"] = df.index
    df["llm_narrative"] = narratives
    print(f"[*] Successfully serialised {len(df)} actionable events.")
    return df
