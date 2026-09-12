"""Phase 1: top-down LLM reasoning to find globally shared boundaries.

Scans the serialised narrative and returns the node ids of actions that are
shared prerequisites/post-requisites for the *entire* session (e.g. a login or
logout). Isolating this from the routing step keeps the LLM's cognitive load
low and lets Phase 2 untangle the remainder without a shared "gravity well".
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from smart_llm_client import SmartLLMClient

# Chain-of-thought is enforced by ordering 'reasoning' before the indices in the
# schema, so the model commits to an argument before committing to numbers.
_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "reasoning": {
            "type": "STRING",
            "description": (
                "Step-by-step justification for why each chosen node is a "
                "GLOBALLY shared boundary rather than a routine-specific action."
            ),
        },
        "shared_indices": {"type": "ARRAY", "items": {"type": "INTEGER"}},
    },
    "required": ["reasoning", "shared_indices"],
}

_SYSTEM_PROMPT = """\
You are an expert Robotic Process Automation (RPA) log-analysis engine.

You are given a chronological sequence of UI events, one per line, each tagged
"Node <id>:". The log contains MULTIPLE interleaved business routines and may
contain multiple interleaved executions of the same routine.

TASK
Identify only the actions that are GLOBALLY SHARED boundaries of the whole
session: setup steps that every routine depends on (e.g. opening the app,
authentication/login) and teardown steps that close the whole session (e.g.
saving global preferences, clearing/ending the session, logout, closing the
browser). Session-cleanup actions performed once at the very end (e.g. "Save
Preferences", "Clear Session") ARE global teardown boundaries, even if they are
not the final logout. There may be several at the start and several at
the end (multiple prerequisites and/or post-requisites).

DECISION RULES
- A node is GLOBALLY SHARED only if it logically belongs to EVERY routine and
  EVERY execution, not just one of them.
- A first action that already carries routine-specific payload (a specific URL
  section, a specific clicked record, specific typed/pasted text) is NOT a
  shared boundary — it is the first step of one routine.
- Authentication/login pages, generic landing/home navigation, and
  logout/sign-out are the classic shared boundaries.
- If the log has no globally shared action, return an empty array. Do not
  invent boundaries.

OUTPUT
First write concise reasoning naming each candidate and why it is (or is not)
global. Then return the JSON array of integer node ids in "shared_indices".

WORKED EXAMPLE
Node 0: [APP: Chrome] action: navigateTo | browser_url: https://crm/auth
Node 1: [APP: Chrome] action: click | browser_url: https://crm/billing | tag_innerText: Refund Client
Node 2: [APP: Chrome] action: click | browser_url: https://crm/support | tag_innerText: TKT-901
Node 3: [APP: Chrome] action: navigateTo | browser_url: https://crm/logout
Reasoning: Node 0 is the auth page shared by all routines (a global prerequisite).
Node 3 is the logout that ends the whole session (a global post-requisite).
Nodes 1-2 carry routine-specific payload (billing refund; a specific ticket),
so they belong to individual routines, not the shared boundary.
shared_indices = [0, 3]
"""


def identify_shared_boundaries(
    df: pd.DataFrame, model_name: str, client: SmartLLMClient | None = None
) -> list[int] | None:
    """Identify globally shared boundary node ids.

    Args:
        df: DataFrame with 'node_id' and 'llm_narrative' columns (as produced by
            ``load_and_serialize_smartrpa``).
        model_name: Gemini model identifier.
        client: Optional shared SmartLLMClient.

    Returns:
        A sorted, de-duplicated list of valid integer node ids, or None on a
        genuine API/parse failure (signals the orchestrator to fail-fast).
    """
    print("\n--- Phase 1: LLM Semantic Boundary Reasoning ---")
    client = client or SmartLLMClient()

    valid_ids = set(int(n) for n in df["node_id"].tolist())
    narrative_by_id = dict(
        zip(df["node_id"].astype(int), df["llm_narrative"].astype(str))
    )

    log_sequence = "".join(
        f"Node {node_id}: {narrative}\n"
        for node_id, narrative in narrative_by_id.items()
    )
    user_prompt = (
        "Analyse the following UI log sequence and return the required JSON "
        f"object:\n\n{log_sequence}"
    )

    try:
        response = client.generate_content(
            model_name, _SYSTEM_PROMPT, user_prompt, _SCHEMA, thinking_budget=512
        )
    except Exception as exc:  # noqa: BLE001 - genuine API failure -> fail-fast
        print(f"[!] CRITICAL: Phase 1 API call failed.\n    Details: {exc}")
        print("[!] Halting Phase 1 to protect data integrity.")
        return None

    reasoning = ""
    raw_indices: list[Any] = []
    if isinstance(response, dict):
        reasoning = str(response.get("reasoning", "No reasoning provided."))
        candidate = response.get("shared_indices", [])
        if isinstance(candidate, list):
            raw_indices = candidate

    # Coerce to ints and validate against real node ids. Out-of-range ids are a
    # model hallucination, NOT an API failure — drop them with a clear warning.
    shared_indices: list[int] = []
    invalid: list[Any] = []
    for value in raw_indices:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            invalid.append(value)
            continue
        if idx in valid_ids:
            shared_indices.append(idx)
        else:
            invalid.append(value)

    shared_indices = sorted(set(shared_indices))

    print("[*] Phase 1 analysis complete.")
    print("\n--- LLM Chain-of-Thought ---")
    print(reasoning)
    print("----------------------------\n")
    if invalid:
        print(f"[!] Ignored {len(invalid)} invalid/out-of-range indices: {invalid}")
    print(f"[*] Shared boundary node ids: {shared_indices}")
    for idx in shared_indices:
        print(f"    -> Node {idx}: {narrative_by_id[idx]}")

    return shared_indices
