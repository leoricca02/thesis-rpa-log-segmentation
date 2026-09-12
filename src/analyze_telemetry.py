"""Observability/economics layer: summarise token telemetry and cost.

Reads the CSV written by ``SmartLLMClient._log_tokens`` (which now has a proper
header row) and reports call counts, token volumes, and an estimated USD cost.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

# Default per-1M-token prices (USD). Override via CLI for other models.
# These are gemini-2.5-flash list prices as of mid-2026, which match the
# telemetry committed under results/. Pass --input-price/--output-price when
# reporting on a run made with a different model.
_DEFAULT_INPUT_PRICE = 0.30
_DEFAULT_OUTPUT_PRICE = 2.50


def analyze_token_telemetry(
    csv_file: str = "token_telemetry.csv",
    input_price: float = _DEFAULT_INPUT_PRICE,
    output_price: float = _DEFAULT_OUTPUT_PRICE,
) -> None:
    """Print a telemetry/cost report for the given telemetry CSV.

    Args:
        csv_file: Path to the telemetry CSV.
        input_price: USD per 1M input tokens.
        output_price: USD per 1M output tokens.
    """
    print("\n=================================================")
    print(" LLM TELEMETRY & OBSERVABILITY REPORT")
    print("=================================================")

    if not os.path.exists(csv_file):
        print(f"[!] No telemetry data at '{csv_file}'. Run the pipeline first.")
        return

    try:
        # The writer now emits a real header row, so let pandas use it.
        df = pd.read_csv(csv_file)
    except (OSError, pd.errors.ParserError) as exc:
        print(f"[!] Could not read telemetry: {exc}")
        return

    required = {"input_tokens", "output_tokens", "total_tokens"}
    if not required.issubset(df.columns):
        print(f"[!] Telemetry file missing expected columns: {required}")
        return

    for col in ("input_tokens", "output_tokens", "total_tokens"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    total_calls = len(df)
    if total_calls == 0:
        print("[!] Telemetry file has a header but no data rows.")
        return

    total_in = int(df["input_tokens"].sum())
    total_out = int(df["output_tokens"].sum())
    total_all = int(df["total_tokens"].sum())

    est_cost = (total_in / 1_000_000) * input_price + (
        total_out / 1_000_000
    ) * output_price

    print(f"Total API Calls Made:   {total_calls}")
    print(f"Total Input Tokens:     {total_in:,}")
    print(f"Total Output Tokens:    {total_out:,}")
    print(f"Total Overall Tokens:   {total_all:,}")
    print("-" * 49)
    print(f"Avg Input per Call:     {total_in / total_calls:,.0f} tokens")
    print(f"Avg Output per Call:    {total_out / total_calls:,.0f} tokens")
    print("-" * 49)
    print(
        f"Estimated Cost:         ${est_cost:.5f} USD "
        f"(@ ${input_price}/M in, ${output_price}/M out)"
    )
    print("=================================================\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise LLM token telemetry.")
    parser.add_argument("csv_file", nargs="?", default="token_telemetry.csv")
    parser.add_argument("--input-price", type=float, default=_DEFAULT_INPUT_PRICE)
    parser.add_argument("--output-price", type=float, default=_DEFAULT_OUTPUT_PRICE)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    analyze_token_telemetry(args.csv_file, args.input_price, args.output_price)
