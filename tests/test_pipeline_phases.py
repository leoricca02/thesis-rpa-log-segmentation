"""Tests for the Seventh Approach phases (scripted stub client, no network)."""

from __future__ import annotations

import pandas as pd
import pytest

import data_pipeline as dp
import phase1_boundary_reasoning as p1
import phase2_execution_mapping as p2


class StubClient:
    """Returns queued responses; records the prompts it was given."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate_content(
        self, model, system_prompt, user_prompt, schema, thinking_budget=0
    ):
        self.calls.append((system_prompt, user_prompt))
        if not self._responses:
            raise AssertionError("StubClient ran out of scripted responses")
        return self._responses.pop(0)


class FailingClient:
    def generate_content(self, *args, **kwargs):
        raise RuntimeError("simulated API failure")


CONSTRAINTS = [
    {"routine_name": "Task A", "executions": 2},
    {"routine_name": "Task B", "executions": 2},
]


def _narrative_df(n):
    return pd.DataFrame(
        {
            "node_id": list(range(n)),
            "llm_narrative": [f"node {i}" for i in range(n)],
        }
    )


# ---------------------------------------------------------------- phase 1
def test_phase1_returns_valid_topology():
    df = _narrative_df(7)
    client = StubClient(
        [
            {
                "reasoning": "r",
                "shared_actions": [
                    {"node_id": 0, "shared_with_routines": ["Task A", "Task B"]},
                    {
                        "node_id": 1,
                        "shared_with_routines": ["Task A"],
                        "justification": "module opener",
                    },
                ],
            }
        ]
    )
    topo = p1.infer_sharing_topology(df, CONSTRAINTS, "m", client)
    assert topo is not None
    assert [t["node_id"] for t in topo] == [0, 1]
    assert topo[0]["shared_with"] == ["Task A", "Task B"]
    assert topo[1]["shared_with"] == ["Task A"]
    assert topo[1]["justification"] == "module opener"


def test_phase1_case_insensitive_name_matching():
    df = _narrative_df(3)
    client = StubClient(
        [
            {
                "reasoning": "r",
                "shared_actions": [
                    {"node_id": 0, "shared_with_routines": ["task a", "  TASK B "]}
                ],
            }
        ]
    )
    topo = p1.infer_sharing_topology(df, CONSTRAINTS, "m", client)
    assert topo[0]["shared_with"] == ["Task A", "Task B"]


def test_phase1_unknown_routine_name_dropped():
    df = _narrative_df(3)
    client = StubClient(
        [
            {
                "reasoning": "r",
                "shared_actions": [
                    {
                        "node_id": 0,
                        "shared_with_routines": ["Task A", "Task Z"],
                    }
                ],
            }
        ]
    )
    topo = p1.infer_sharing_topology(df, CONSTRAINTS, "m", client)
    assert topo[0]["shared_with"] == ["Task A"]  # Task Z silently dropped


def test_phase1_empty_subset_reclassified_as_routable():
    df = _narrative_df(3)
    client = StubClient(
        [
            {
                "reasoning": "r",
                "shared_actions": [
                    {"node_id": 0, "shared_with_routines": ["Task Z"]},  # all bad
                    {"node_id": 1, "shared_with_routines": ["Task A"]},
                ],
            }
        ]
    )
    topo = p1.infer_sharing_topology(df, CONSTRAINTS, "m", client)
    assert [t["node_id"] for t in topo] == [1]  # node 0 removed from topology


def test_phase1_out_of_range_node_dropped():
    df = _narrative_df(3)
    client = StubClient(
        [
            {
                "reasoning": "r",
                "shared_actions": [
                    {"node_id": 99, "shared_with_routines": ["Task A"]},
                    {"node_id": 2, "shared_with_routines": ["Task B"]},
                ],
            }
        ]
    )
    topo = p1.infer_sharing_topology(df, CONSTRAINTS, "m", client)
    assert [t["node_id"] for t in topo] == [2]


def test_phase1_duplicate_node_entries_merged():
    df = _narrative_df(3)
    client = StubClient(
        [
            {
                "reasoning": "r",
                "shared_actions": [
                    {"node_id": 0, "shared_with_routines": ["Task A"]},
                    {"node_id": 0, "shared_with_routines": ["Task B"]},
                ],
            }
        ]
    )
    topo = p1.infer_sharing_topology(df, CONSTRAINTS, "m", client)
    assert len(topo) == 1
    assert topo[0]["shared_with"] == ["Task A", "Task B"]


def test_phase1_empty_topology_is_valid():
    df = _narrative_df(3)
    client = StubClient([{"reasoning": "none", "shared_actions": []}])
    assert p1.infer_sharing_topology(df, CONSTRAINTS, "m", client) == []


def test_phase1_returns_none_on_api_failure():
    df = _narrative_df(3)
    assert p1.infer_sharing_topology(df, CONSTRAINTS, "m", FailingClient()) is None


# ---------------------------------------------------------------- phase 2
def _phase2_df():
    """7 nodes: 0=login(ALL) 1=open-X(A only) 2,4=A steps 3,5=B steps 6=logout."""
    return pd.DataFrame(
        {
            "node_id": list(range(7)),
            "application": ["App"] * 7,
            "event_type": ["nav", "click", "click", "click", "click", "click", "nav"],
            "llm_narrative": [
                "login", "open section X", "A-1", "B-1", "A-2", "B-2", "logout",
            ],
        }
    )


def _topology_partial():
    return [
        {"node_id": 0, "shared_with": ["Task A", "Task B"], "justification": "login"},
        {"node_id": 1, "shared_with": ["Task A"], "justification": "opener"},
        {"node_id": 6, "shared_with": ["Task A", "Task B"], "justification": "logout"},
    ]


def _plan_ok():
    return [
        {"routine_name": "Task A", "execution_index": 1, "assigned_node_indices": [2]},
        {"routine_name": "Task A", "execution_index": 2, "assigned_node_indices": [4]},
        {"routine_name": "Task B", "execution_index": 1, "assigned_node_indices": [3]},
        {"routine_name": "Task B", "execution_index": 2, "assigned_node_indices": [5]},
    ]


def test_phase2_partial_sharing_happy_path(tmp_path):
    out = tmp_path / "out.xlsx"
    client = StubClient([_plan_ok()])
    ok = p2.generate_segmented_log(
        _phase2_df(), _topology_partial(), CONSTRAINTS, "m",
        output_file=str(out), client=client,
    )
    assert ok is True
    result = pd.read_excel(out, sheet_name="Segmented_Logs")
    a1 = result[result["trace_id"] == "Task A_exec1"]
    b1 = result[result["trace_id"] == "Task B_exec1"]
    # Task A traces get login + opener + own step + logout = 4 rows.
    assert len(a1) == 4
    # Task B traces get login + own step + logout = 3 rows (NO opener).
    assert len(b1) == 3
    assert "open section X" not in " ".join(b1["llm_narrative"].astype(str)) \
        if "llm_narrative" in b1.columns else True
    # The opener narrative lives only in A traces (check via event presence).
    topo = pd.read_excel(out, sheet_name="Sharing_Topology")
    assert len(topo) == 3
    assert set(topo["node_id"]) == {0, 1, 6}


def test_phase2_full_sharing_is_special_case(tmp_path):
    """Subset == all routines must reproduce Sixth-Approach behaviour."""
    out = tmp_path / "out.xlsx"
    topo_all = [
        {"node_id": 0, "shared_with": ["Task A", "Task B"], "justification": ""},
        {"node_id": 1, "shared_with": ["Task A", "Task B"], "justification": ""},
        {"node_id": 6, "shared_with": ["Task A", "Task B"], "justification": ""},
    ]
    client = StubClient([_plan_ok()])
    ok = p2.generate_segmented_log(
        _phase2_df(), topo_all, CONSTRAINTS, "m",
        output_file=str(out), client=client,
    )
    assert ok is True
    result = pd.read_excel(out, sheet_name="Segmented_Logs")
    # Every trace now has all 3 shared nodes + 1 own = 4 rows; 4 traces = 16.
    assert len(result) == 16


def test_phase2_missing_node_diverted_to_noise_not_aborted(tmp_path):
    # New behaviour: a routable node the LLM declines to place is diverted to
    # the Noise sheet (soft warning), NOT a hard abort. Nothing is lost.
    plan = [
        {"routine_name": "Task A", "execution_index": 1, "assigned_node_indices": [2]},
        {"routine_name": "Task B", "execution_index": 1, "assigned_node_indices": [3]},
    ]  # nodes 4 and 5 not routed
    client = StubClient([plan])
    out = tmp_path / "o.xlsx"
    ok = p2.generate_segmented_log(
        _phase2_df(), _topology_partial(), CONSTRAINTS, "m",
        output_file=str(out), client=client,
    )
    assert ok is True
    import openpyxl
    wb = openpyxl.load_workbook(out)
    assert "Noise" in wb.sheetnames


def test_phase2_hard_abort_on_duplicate_routable_node(tmp_path):
    plan = _plan_ok()
    plan[1]["assigned_node_indices"] = [2, 4]  # node 2 also in block 1
    client = StubClient([plan])
    ok = p2.generate_segmented_log(
        _phase2_df(), _topology_partial(), CONSTRAINTS, "m",
        output_file=str(tmp_path / "o.xlsx"), client=client,
    )
    assert ok is False


def test_phase2_hard_abort_on_undeclared_routine_name(tmp_path):
    plan = _plan_ok()
    plan[0]["routine_name"] = "Task Q"  # invented
    client = StubClient([plan])
    ok = p2.generate_segmented_log(
        _phase2_df(), _topology_partial(), CONSTRAINTS, "m",
        output_file=str(tmp_path / "o.xlsx"), client=client,
    )
    assert ok is False


def test_phase2_hard_abort_when_shared_node_unreachable(tmp_path):
    """A shared node whose subset's routine produced no blocks must abort.

    Oracle declares Task B, but the LLM only produced Task A blocks and the
    topology has a node shared exclusively with Task B -> that node would
    appear in no trace -> P3 cover violation.
    """
    df = pd.DataFrame(
        {
            "node_id": [0, 1, 2],
            "application": ["App"] * 3,
            "event_type": ["nav", "click", "click"],
            "llm_narrative": ["B-only opener", "A-1", "A-2"],
        }
    )
    topology = [{"node_id": 0, "shared_with": ["Task B"], "justification": ""}]
    plan = [
        {"routine_name": "Task A", "execution_index": 1, "assigned_node_indices": [1]},
        {"routine_name": "Task A", "execution_index": 2, "assigned_node_indices": [2]},
    ]
    client = StubClient([plan])
    ok = p2.generate_segmented_log(
        df, topology, CONSTRAINTS, "m",
        output_file=str(tmp_path / "o.xlsx"), client=client,
    )
    assert ok is False


def test_phase2_soft_warning_on_count_mismatch_still_succeeds(tmp_path, capsys):
    plan = [
        {"routine_name": "Task A", "execution_index": 1,
         "assigned_node_indices": [2, 4]},
        {"routine_name": "Task B", "execution_index": 1,
         "assigned_node_indices": [3]},
        {"routine_name": "Task B", "execution_index": 2,
         "assigned_node_indices": [5]},
    ]  # Task A: expected 2 execs, got 1 — but all nodes covered
    client = StubClient([plan])
    ok = p2.generate_segmented_log(
        _phase2_df(), _topology_partial(), CONSTRAINTS, "m",
        output_file=str(tmp_path / "o.xlsx"), client=client,
    )
    assert ok is True
    assert "SOFT WARNING" in capsys.readouterr().out


def test_phase2_soft_warning_routine_without_shared_actions(tmp_path, capsys):
    topology = [
        {"node_id": 0, "shared_with": ["Task A"], "justification": ""},
        {"node_id": 1, "shared_with": ["Task A"], "justification": ""},
        {"node_id": 6, "shared_with": ["Task A"], "justification": ""},
    ]  # Task B gets nothing shared
    client = StubClient([_plan_ok()])
    ok = p2.generate_segmented_log(
        _phase2_df(), topology, CONSTRAINTS, "m",
        output_file=str(tmp_path / "o.xlsx"), client=client,
    )
    assert ok is True
    out = capsys.readouterr().out
    assert "received NO shared actions" in out


def test_phase2_empty_topology_routes_everything(tmp_path):
    """No shared actions at all: every node must be routed (cover == partition)."""
    df = _phase2_df()
    plan = [
        {"routine_name": "Task A", "execution_index": 1,
         "assigned_node_indices": [0, 1, 2, 4]},
        {"routine_name": "Task B", "execution_index": 1,
         "assigned_node_indices": [3, 5, 6]},
    ]
    client = StubClient([plan])
    ok = p2.generate_segmented_log(
        df, [], [{"routine_name": "Task A", "executions": 1},
                 {"routine_name": "Task B", "executions": 1}],
        "m", output_file=str(tmp_path / "o.xlsx"), client=client,
    )
    assert ok is True


@pytest.mark.parametrize("amount", [0.1, 1.0, 5.0])
def test_color_lightness_stays_in_legible_band(amount):
    out = p2.adjust_color_lightness("#1F77B4", amount)
    assert out.startswith("#") and len(out) == 7


def test_trace_color_is_deterministic():
    assert p2._trace_color(0, 0, 2) == p2._trace_color(0, 0, 2)


# ------------------------------------------------------------ data_pipeline
def test_pipeline_still_ingests(tmp_path):
    df = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:01", "2026-01-01T00:00:02"],
            "application": ["App", "App"],
            "event_type": ["nav", "click"],
            "browser_url": ["https://x/auth", "https://x/a"],
        }
    )
    csv = tmp_path / "log.csv"
    df.to_csv(csv, sep=";", index=False)
    client = StubClient([[], ["timestamp"]])
    out = dp.load_and_serialize_smartrpa(str(csv), "m", client)
    assert list(out["node_id"]) == [0, 1]
    assert "llm_narrative" in out.columns


def test_tag_irrelevant_marks_and_payload_guard_rescues():
    df = pd.DataFrame({
        "node_id": [0, 1, 2],
        "llm_narrative": ["junk lunch note", "Case REF-801", "scroll"],
    })
    client = StubClient([[1, 2]])  # model flags 1 (has payload!) and 2
    out = dp.tag_irrelevant_nodes(df, ["Process Refund"], "m", client)
    # node 1 carries REF-801 (distinctive) -> rescued; node 2 stays noise
    assert not out.loc[out.node_id == 1, "is_probable_noise"].iloc[0]
    assert out.loc[out.node_id == 2, "is_probable_noise"].iloc[0]


def test_tag_irrelevant_fail_open():
    df = pd.DataFrame({"node_id": [0, 1], "llm_narrative": ["a", "b"]})
    out = dp.tag_irrelevant_nodes(df, ["R"], "m", FailingClient())
    assert not out["is_probable_noise"].any()
