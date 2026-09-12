"""Phase 1 (Seventh Approach): LLM inference of the SHARING TOPOLOGY.

The Sixth Approach operated under H1 (Global Sharing): every shared action
belongs to ALL routines, so Phase 1 returned a flat list of boundary node ids.

The Seventh Approach relaxes H1 to H1' (Subset Sharing): a shared action may
be a prerequisite/post-requisite for only a SUBSET of the declared routines
(e.g. an "Open Finance Workspace" click needed by 2 of 4 routines). Phase 1
therefore returns a *topology*: a mapping from each shared node to the set of
routine names it is shared with. Full sharing is the special case where the
subset equals all declared routines.

Design choices (documented for the thesis):
- The topology is INFERRED by the LLM (option "b"): the human Oracle still
  declares only routine names and execution counts; the model decides which
  shared action belongs to which routines, justifying each assignment via
  chain-of-thought. A supervised fallback ("c") is available in the
  orchestrator via the --review flag, which lets the operator inspect and
  veto the inferred topology before Phase 2 spends tokens on it.
- Model output is SANITISED, not blindly trusted: invalid node ids and
  unknown routine names are dropped with loud warnings; a node whose subset
  becomes empty is reclassified as routable (i.e. NOT shared). If more than
  half of the proposed shared actions required correction, the function
  recommends switching to the supervised fallback.
- Classification bias is deliberately conservative: the prompt instructs the
  model that, when in doubt, an action should be treated as NOT shared. A
  wrongly-unshared action merely gets routed into one execution (a local
  error); a wrongly-shared action is duplicated into many traces (a global
  corruption).
"""

from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd

from smart_llm_client import SmartLLMClient

# Topology inference is a harder reasoning task than the Sixth Approach's flat
# boundary detection, so it gets a larger (but still bounded) thinking budget.
_THINKING_BUDGET = 1024


class SharedAction(TypedDict):
    """One entry of the inferred sharing topology."""

    node_id: int
    shared_with: list[str]
    justification: str


# Chain-of-thought is enforced by ordering 'reasoning' before the topology in
# the schema, so the model commits to an argument before committing to ids.
_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "reasoning": {
            "type": "STRING",
            "description": (
                "Step-by-step analysis: which actions are shared rather than "
                "execution-specific, and which routines each one serves."
            ),
        },
        "shared_actions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "node_id": {"type": "INTEGER"},
                    "shared_with_routines": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "description": (
                            "Exact routine names (copied verbatim from the "
                            "declared list) that depend on this action."
                        ),
                    },
                    "justification": {"type": "STRING"},
                },
                "required": ["node_id", "shared_with_routines"],
            },
        },
    },
    "required": ["reasoning", "shared_actions"],
}

_SYSTEM_PROMPT = """\
You are an expert Robotic Process Automation (RPA) log-analysis engine.

You receive a chronological sequence of UI events, one per line, each tagged
"Node <id>:". The log contains MULTIPLE interleaved business routines, and may
contain multiple interleaved executions of the same routine. You also receive
the list of routine names present in the log and how many times each was
executed, declared by a human operator.

TASK
Identify the SHARED actions and, for each one, decide WHICH routines it is
shared with. A shared action is a setup, navigation, authentication, teardown,
or other supporting step that one or more routines depend on, but which is not
itself a business step of any single execution. Sharing is now PARTIAL: an
action may serve ALL routines (e.g. logging in), or only a SUBSET of them
(e.g. opening a module that only some routines use), or even a single routine
(a preparation step performed once that all executions of that routine rely
on).

DECISION RULES
1. FREQUENCY SIGNAL: compare how often an action occurs with the declared
   execution counts. An action that recurs roughly once PER EXECUTION of a
   routine is an execution-specific business step, NOT shared. An action that
   occurs ONCE (or far fewer times than the executions that need it) is a
   sharing candidate.
2. CONTENT SIGNAL: actions carrying an execution-specific payload (a unique
   record id, a unique typed/pasted value) belong to that single execution and
   are NOT shared. Shared actions are generic: authentication, opening a
   workspace/module/section, saving global preferences, logging out.
3. SUBSET ASSIGNMENT: assign a shared action to a routine only if that routine
   plausibly DEPENDS on it. Use the action's context (which area/section it
   touches, what it enables) versus where each routine's business steps take
   place. Authentication and session teardown typically serve ALL routines.
   A section/module opener typically serves only the routines whose steps
   happen in that section.
4. WHEN IN DOUBT, DO NOT SHARE: if you cannot justify why an action is a
   dependency of a routine, leave that routine out of the subset. If you
   cannot justify sharing at all, omit the action entirely — it will be
   routed as a normal execution step. Never invent sharing.
5. Copy routine names VERBATIM from the declared list. Never invent names.
6. If the log has no shared actions at all, return an empty array.

OUTPUT
First write concise reasoning. Then return "shared_actions": for each shared
node, its integer node_id, the exact routine names it serves, and a one-line
justification.

WORKED EXAMPLE (abstract)
Declared routines: "Task A" (2 executions), "Task B" (2 executions).
Node 0: action: navigate | url: https://app/login
Node 1: action: click | url: https://app/section-x | text: Open Section X
Node 2: action: click | url: https://app/section-x | text: Item A-1
Node 3: action: click | url: https://app/section-y | text: Item B-1
Node 4: action: click | url: https://app/section-x | text: Item A-2
Node 5: action: click | url: https://app/section-y | text: Item B-2
Node 6: action: navigate | url: https://app/logout
Reasoning: Node 0 (login) and Node 6 (logout) occur once and gate the whole
session: shared with both routines. Node 1 opens Section X, occurs once, and
only Task A's steps (nodes 2, 4) happen in Section X, so it is shared with
Task A only — both executions of Task A depend on it, but Task B never enters
Section X. Nodes 2-5 carry item-specific payloads recurring once per
execution: not shared.
shared_actions = [
  {node_id 0, shared_with ["Task A", "Task B"]},
  {node_id 1, shared_with ["Task A"]},
  {node_id 6, shared_with ["Task A", "Task B"]}
]
"""


def _canonicalize(
    raw_names: list[Any], declared: dict[str, str]
) -> tuple[list[str], list[str]]:
    """Map model-emitted routine names onto declared names.

    Matching is case-insensitive after stripping. Returns (matched canonical
    names, unmatched raw names).
    """
    matched: list[str] = []
    unmatched: list[str] = []
    for raw in raw_names:
        key = str(raw).strip().lower()
        if key in declared:
            canonical = declared[key]
            if canonical not in matched:
                matched.append(canonical)
        else:
            unmatched.append(str(raw))
    return matched, unmatched


def infer_sharing_topology(
    df: pd.DataFrame,
    routine_constraints: list[dict[str, Any]],
    model_name: str,
    client: SmartLLMClient | None = None,
) -> list[SharedAction] | None:
    """Infer which nodes are shared and with which routines.

    Args:
        df: DataFrame with 'node_id' and 'llm_narrative' columns (as produced
            by ``load_and_serialize_smartrpa``).
        routine_constraints: Oracle constraints, each a dict with
            'routine_name' and 'executions'.
        model_name: Gemini model identifier.
        client: Optional shared SmartLLMClient.

    Returns:
        The sanitised sharing topology (possibly empty), or None on a genuine
        API/parse failure (signals the orchestrator to fail-fast). Each entry
        has a valid node_id, a non-empty list of declared routine names, and a
        justification string.
    """
    print("\n--- Phase 1: LLM Sharing-Topology Inference (H1 relaxed) ---")
    client = client or SmartLLMClient()

    valid_ids = set(int(n) for n in df["node_id"].tolist())
    narrative_by_id = dict(
        zip(df["node_id"].astype(int), df["llm_narrative"].astype(str))
    )
    declared = {
        str(r["routine_name"]).strip().lower(): str(r["routine_name"]).strip()
        for r in routine_constraints
    }

    constraint_lines = "\n".join(
        f'- "{str(r["routine_name"]).strip()}": {int(r["executions"])} execution(s)'
        for r in routine_constraints
    )
    log_sequence = "".join(
        f"Node {node_id}: {narrative}\n"
        for node_id, narrative in narrative_by_id.items()
    )
    user_prompt = (
        "Declared routines and execution counts:\n"
        f"{constraint_lines}\n\n"
        "Analyse the following UI log sequence and return the required JSON "
        f"object:\n\n{log_sequence}"
    )

    try:
        response = client.generate_content(
            model_name,
            _SYSTEM_PROMPT,
            user_prompt,
            _SCHEMA,
            thinking_budget=_THINKING_BUDGET,
        )
    except Exception as exc:  # noqa: BLE001 - genuine API failure -> fail-fast
        print(f"[!] CRITICAL: Phase 1 API call failed.\n    Details: {exc}")
        print("[!] Halting Phase 1 to protect data integrity.")
        return None

    reasoning = ""
    raw_actions: list[Any] = []
    if isinstance(response, dict):
        reasoning = str(response.get("reasoning", "No reasoning provided."))
        candidate = response.get("shared_actions", [])
        if isinstance(candidate, list):
            raw_actions = candidate

    print("[*] Phase 1 analysis complete.")
    print("\n--- LLM Chain-of-Thought ---")
    print(reasoning)
    print("----------------------------\n")

    # ---------------- sanitisation (trust, but verify) ----------------
    topology: dict[int, SharedAction] = {}
    corrections = 0
    proposals = 0

    for entry in raw_actions:
        if not isinstance(entry, dict):
            corrections += 1
            continue
        proposals += 1

        try:
            node_id = int(entry.get("node_id"))
        except (TypeError, ValueError):
            print(f"[!] Dropped topology entry with invalid node_id: {entry!r}")
            corrections += 1
            continue
        if node_id not in valid_ids:
            print(f"[!] Dropped out-of-range node_id from topology: {node_id}")
            corrections += 1
            continue

        raw_subset = entry.get("shared_with_routines", [])
        if not isinstance(raw_subset, list):
            raw_subset = []
        matched, unmatched = _canonicalize(raw_subset, declared)
        if unmatched:
            print(
                f"[!] Node {node_id}: ignored unknown routine name(s) "
                f"{unmatched} (not in the declared list)."
            )
            corrections += 1
        if not matched:
            print(
                f"[!] Node {node_id}: subset empty after sanitisation -> "
                "reclassified as a normal routable action (not shared)."
            )
            corrections += 1
            continue

        justification = str(entry.get("justification", "")).strip()
        if node_id in topology:
            # Duplicate node id: merge subsets, keep first justification.
            merged = topology[node_id]["shared_with"]
            for name in matched:
                if name not in merged:
                    merged.append(name)
            print(f"[!] Node {node_id}: duplicate topology entries merged.")
            corrections += 1
        else:
            topology[node_id] = SharedAction(
                node_id=node_id,
                shared_with=matched,
                justification=justification,
            )

    result = [topology[k] for k in sorted(topology)]

    n_all = len(declared)
    print(f"[*] Inferred sharing topology: {len(result)} shared action(s).")
    for action in result:
        scope = (
            "ALL routines"
            if len(action["shared_with"]) == n_all
            else f"{len(action['shared_with'])}/{n_all} routines "
            f"({', '.join(action['shared_with'])})"
        )
        print(f"    -> Node {action['node_id']} shared with {scope}")
        print(f"       {narrative_by_id[action['node_id']][:90]}")

    if proposals and corrections > proposals / 2:
        print(
            "[!] More than half of the proposed shared actions required "
            "correction. The inferred topology may be unreliable; consider "
            "re-running with --review (supervised fallback) or a stronger "
            "model."
        )

    return result
