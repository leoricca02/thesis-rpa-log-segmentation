"""Tests for the pipeline phases using a scripted stub client (no network)."""

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

    def generate_content(self, model, system_prompt, user_prompt, schema, thinking_budget=0):
        self.calls.append((system_prompt, user_prompt))
        if not self._responses:
            raise AssertionError("StubClient ran out of scripted responses")
        return self._responses.pop(0)


class FailingClient:
    def generate_content(self, *args, **kwargs):
        raise RuntimeError("simulated API failure")


# --------------------------------------------------------------- data_pipeline
def _toy_df():
    return pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:01", "2026-01-01T00:00:02"],
            "application": ["Chrome", "Chrome"],
            "event_type": ["navigateTo", "mouseMove"],
            "browser_url": ["https://x/auth", ""],
            "id": ["1", "2"],
        }
    )


def test_filter_noise_removes_flagged_events():
    df = _toy_df()
    client = StubClient([["mouseMove"]])
    out = dp.filter_noise_with_llm(df, "m", client)
    assert list(out["event_type"]) == ["navigateTo"]
    assert out.index.tolist() == [0]


def test_filter_noise_fail_open_keeps_all_rows():
    df = _toy_df()
    out = dp.filter_noise_with_llm(df, "m", FailingClient())
    assert len(out) == len(df)


def test_metadata_columns_drops_invalid_and_anchors():
    df = _toy_df()
    # LLM tries to drop a real metadata col, a nonexistent col, and an anchor.
    client = StubClient([["id", "does_not_exist", "event_type"]])
    out = dp.identify_metadata_columns_with_llm(df, "m", client)
    assert "id" in out
    assert "does_not_exist" not in out
    assert "event_type" not in out  # anchors are protected


def test_metadata_columns_fallback_on_failure():
    df = _toy_df()
    out = dp.identify_metadata_columns_with_llm(df, "m", FailingClient())
    assert "id" in out and "timestamp" in out


def test_tag_irrelevant_marks_flagged_nodes():
    df = pd.DataFrame(
        {"node_id": [0, 1, 2], "llm_narrative": ["a", "b", "c"]}
    )
    client = StubClient([[1]])  # node 1 unrelated
    out = dp.tag_irrelevant_nodes(df, ["Routine X"], "m", client)
    assert out.loc[out.node_id == 1, "is_probable_noise"].iloc[0]
    assert not out.loc[out.node_id == 0, "is_probable_noise"].iloc[0]


def test_tag_irrelevant_fail_open_tags_nothing():
    df = pd.DataFrame({"node_id": [0, 1], "llm_narrative": ["a", "b"]})
    out = dp.tag_irrelevant_nodes(df, ["Routine X"], "m", FailingClient())
    assert not out["is_probable_noise"].any()


def test_tag_irrelevant_ignores_out_of_range():
    df = pd.DataFrame({"node_id": [0, 1], "llm_narrative": ["a", "b"]})
    client = StubClient([[99]])  # invalid id
    out = dp.tag_irrelevant_nodes(df, ["Routine X"], "m", client)
    assert not out["is_probable_noise"].any()


def test_phase2_pretagged_noise_diverted_and_clean_routing(tmp_path):
    df = _phase2_df()
    df["is_probable_noise"] = [False, False, True, False]  # node 2 tagged noise
    out = tmp_path / "o.xlsx"
    # Only node 1 is routable now (0,3 shared; 2 noise).
    plan = [
        {"routine_name": "R", "execution_index": 1, "assigned_node_indices": [1]},
    ]
    client = StubClient([plan])
    ok = p2.generate_segmented_log(
        df, [0, 3], [{"routine_name": "R", "executions": 1}], "m",
        output_file=str(out), client=client,
    )
    assert ok is True
    import openpyxl
    wb = openpyxl.load_workbook(out)
    assert "Noise" in wb.sheetnames


def test_phase2_no_noise_means_no_noise_sheet(tmp_path):
    # Regression: clean log (no tag column, full coverage) -> no Noise sheet,
    # identical to pre-upgrade behaviour.
    out = tmp_path / "o.xlsx"
    plan = [
        {"routine_name": "R", "execution_index": 1, "assigned_node_indices": [1]},
        {"routine_name": "R", "execution_index": 2, "assigned_node_indices": [2]},
    ]
    client = StubClient([plan])
    ok = p2.generate_segmented_log(
        _phase2_df(), [0, 3],
        [{"routine_name": "R", "executions": 2}], "m",
        output_file=str(out), client=client,
    )
    assert ok is True
    import openpyxl
    wb = openpyxl.load_workbook(out)
    assert wb.sheetnames == ["Segmented_Logs"]


def test_serialize_attaches_node_id_and_narrative(tmp_path):
    csv = tmp_path / "log.csv"
    _toy_df().to_csv(csv, sep=";", index=False)
    client = StubClient([["mouseMove"], ["id", "timestamp"]])
    out = dp.load_and_serialize_smartrpa(str(csv), "m", client)
    assert list(out["node_id"]) == list(range(len(out)))
    assert out.index.tolist() == list(out["node_id"])
    assert "auth" in out.iloc[0]["llm_narrative"]
    assert "id:" not in out.iloc[0]["llm_narrative"]  # pruned


def test_serialize_without_timestamp_column(tmp_path):
    df = _toy_df().drop(columns=["timestamp"])
    csv = tmp_path / "log.csv"
    df.to_csv(csv, sep=";", index=False)
    client = StubClient([[], []])
    out = dp.load_and_serialize_smartrpa(str(csv), "m", client)
    assert len(out) == 2


# ------------------------------------------------------------------- phase 1
def _narrative_df(n):
    return pd.DataFrame(
        {
            "node_id": list(range(n)),
            "llm_narrative": [f"node {i}" for i in range(n)],
        }
    )


def test_phase1_returns_sorted_valid_indices():
    df = _narrative_df(5)
    client = StubClient([{"reasoning": "r", "shared_indices": [4, 0]}])
    assert p1.identify_shared_boundaries(df, "m", client) == [0, 4]


def test_phase1_drops_out_of_range_indices():
    df = _narrative_df(3)
    client = StubClient([{"reasoning": "r", "shared_indices": [0, 99]}])
    assert p1.identify_shared_boundaries(df, "m", client) == [0]


def test_phase1_empty_when_no_boundaries():
    df = _narrative_df(3)
    client = StubClient([{"reasoning": "none", "shared_indices": []}])
    assert p1.identify_shared_boundaries(df, "m", client) == []


def test_phase1_returns_none_on_api_failure():
    df = _narrative_df(3)
    assert p1.identify_shared_boundaries(df, "m", FailingClient()) is None


# ------------------------------------------------------------------- phase 2
def _phase2_df():
    return pd.DataFrame(
        {
            "node_id": [0, 1, 2, 3],
            "application": ["Chrome"] * 4,
            "event_type": ["navigateTo", "click", "click", "navigateTo"],
            "llm_narrative": ["auth", "refund", "ticket", "logout"],
        }
    )


def test_phase2_happy_path_writes_workbook(tmp_path):
    df = _phase2_df()
    out = tmp_path / "out.xlsx"
    plan = [
        {"routine_name": "Refund", "execution_index": 1, "assigned_node_indices": [1]},
        {"routine_name": "Ticket", "execution_index": 1, "assigned_node_indices": [2]},
    ]
    client = StubClient([plan])
    ok = p2.generate_segmented_log(
        df,
        shared_indices=[0, 3],
        routine_constraints=[
            {"routine_name": "Refund", "executions": 1},
            {"routine_name": "Ticket", "executions": 1},
        ],
        model_name="m",
        output_file=str(out),
        client=client,
    )
    assert ok is True
    assert out.exists()
    result = pd.read_excel(out)
    # Each trace gets both shared boundaries (0 and 3) plus its own node.
    refund = result[result["trace_id"] == "Refund_exec1"]
    assert set(refund["node_id"] if "node_id" in refund else refund["event_type"])
    assert len(refund) == 3  # auth + refund + logout
    assert (tmp_path / "out_routing.json").exists()


def test_phase2_missing_node_diverted_to_noise_not_aborted(tmp_path):
    # New behaviour: a node the LLM declines to route is diverted to the Noise
    # sheet (soft warning), NOT a hard abort. Nothing is lost.
    df = _phase2_df()
    out = tmp_path / "o.xlsx"
    plan = [
        {"routine_name": "R", "execution_index": 1, "assigned_node_indices": [1]},
    ]  # node 2 not routed
    client = StubClient([plan])
    ok = p2.generate_segmented_log(
        df, [0, 3], [{"routine_name": "R", "executions": 1}], "m",
        output_file=str(out), client=client,
    )
    assert ok is True
    # The unrouted node must be preserved in the Noise sheet.
    import openpyxl
    wb = openpyxl.load_workbook(out)
    assert "Noise" in wb.sheetnames
    noise_rows = list(wb["Noise"].iter_rows(values_only=True))
    assert len(noise_rows) >= 2  # header + at least the dropped node


def test_phase2_hard_abort_on_duplicate_node(tmp_path):
    df = _phase2_df()
    plan = [
        {"routine_name": "A", "execution_index": 1, "assigned_node_indices": [1, 2]},
        {"routine_name": "B", "execution_index": 1, "assigned_node_indices": [2]},
    ]
    client = StubClient([plan])
    ok = p2.generate_segmented_log(
        df, [0, 3],
        [{"routine_name": "A", "executions": 1}, {"routine_name": "B", "executions": 1}],
        "m", output_file=str(tmp_path / "o.xlsx"), client=client,
    )
    assert ok is False


def test_phase2_soft_warning_count_mismatch_still_succeeds(tmp_path, capsys):
    df = _phase2_df()
    # Coverage is fine (1 and 2 both used) but Oracle wanted 2 Refund execs.
    plan = [
        {"routine_name": "Refund", "execution_index": 1, "assigned_node_indices": [1, 2]},
    ]
    client = StubClient([plan])
    ok = p2.generate_segmented_log(
        df, [0, 3], [{"routine_name": "Refund", "executions": 2}], "m",
        output_file=str(tmp_path / "o.xlsx"), client=client,
    )
    assert ok is True
    assert "SOFT WARNING" in capsys.readouterr().out


def test_phase2_export_handles_fully_empty_column(tmp_path):
    # A kept column that is empty across every row must not crash the xlsx
    # width calculation (regression: int(NaN) at scale).
    df = pd.DataFrame(
        {
            "node_id": [0, 1, 2, 3],
            "application": ["Chrome"] * 4,
            "event_type": ["navigateTo", "click", "click", "navigateTo"],
            "always_empty": ["", "", "", ""],
            "llm_narrative": ["auth", "refund", "ticket", "logout"],
        }
    )
    out = tmp_path / "out.xlsx"
    plan = [
        {"routine_name": "R", "execution_index": 1, "assigned_node_indices": [1, 2]},
    ]
    client = StubClient([plan])
    ok = p2.generate_segmented_log(
        df, [0, 3], [{"routine_name": "R", "executions": 1}], "m",
        output_file=str(out), client=client,
    )
    assert ok is True
    assert out.exists()


@pytest.mark.parametrize("amount", [0.1, 1.0, 5.0])
def test_color_lightness_stays_in_legible_band(amount):
    out = p2.adjust_color_lightness("#1F77B4", amount)
    assert out.startswith("#") and len(out) == 7


def test_trace_color_is_deterministic():
    a = p2._trace_color(0, 0, 2)
    b = p2._trace_color(0, 0, 2)
    assert a == b
