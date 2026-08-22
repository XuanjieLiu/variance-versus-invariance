#!/usr/bin/env python3
"""Find the first stable semantic-phase window in codebook epoch history."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--required", type=int, default=8)
    parser.add_argument("--macro", type=float, default=0.70)
    parser.add_argument("--active", type=int, default=26)
    parser.add_argument("--perplexity", type=float, default=20.0)
    parser.add_argument("--coverage", type=int, default=24)
    args = parser.parse_args()

    with args.history.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["partition"] == "val"]
    rows.sort(key=lambda row: int(row["epoch"]))

    def qualifies(row):
        return (
            float(row["macro_atom_purity"]) >= args.macro
            and int(row["active_codes"]) == args.active
            and float(row["usage_perplexity"]) >= args.perplexity
            and int(row["dominant_label_coverage"]) >= args.coverage
        )

    winner = None
    for start in range(max(0, len(rows) - args.window + 1)):
        window = rows[start : start + args.window]
        epochs = [int(row["epoch"]) for row in window]
        if epochs[-1] - epochs[0] != args.window - 1:
            continue
        passing_epochs = [int(row["epoch"]) for row in window if qualifies(row)]
        if len(passing_epochs) >= args.required:
            winner = {
                "onset_epoch": passing_epochs[0],
                "stable_window_start": epochs[0],
                "stable_window_end": epochs[-1],
                "qualifying_epochs": passing_epochs,
                "qualifying_count": len(passing_epochs),
            }
            break

    payload = {
        "history": str(args.history.resolve()),
        "validation_epoch_count": len(rows),
        "thresholds": {
            "window": args.window,
            "required": args.required,
            "macro_atom_purity": args.macro,
            "active_codes": args.active,
            "usage_perplexity": args.perplexity,
            "dominant_label_coverage": args.coverage,
        },
        "stable_phase_detected": winner is not None,
        **(winner or {}),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
