from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STEPS = [
    "01_validate_inputs",
    "02_feature_finding",
    "03_align_features",
    "04_blank_subtract",
    "05_adduct_filter",
    "06_extract_ms2",
    "07_link_ms2_features",
    "08_build_library_index",
    "09_library_search",
    "10_report_tables",
    "11_predict_structures",
]


def normalize_step(value: str) -> str:
    value = value.strip()
    if value.isdigit():
        value = value.zfill(2)
    if value[:2].isdigit() and len(value) == 2:
        for step in STEPS:
            if step.startswith(f"{value}_"):
                return step
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pipeline steps with start/end bounds.")
    parser.add_argument("--start", default=STEPS[0], help="First step to run.")
    parser.add_argument("--end", default=STEPS[-1], help="Last step to run.")
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated list of step names to skip (e.g., 05_adduct_filter,08_build_library_index).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.start = normalize_step(args.start)
    args.end = normalize_step(args.end)
    if args.start not in STEPS:
        raise SystemExit(f"Unknown start step: {args.start}")
    if args.end not in STEPS:
        raise SystemExit(f"Unknown end step: {args.end}")

    start_idx = STEPS.index(args.start)
    end_idx = STEPS.index(args.end)
    if start_idx > end_idx:
        raise SystemExit("Start step must come before end step.")

    skip_set = {normalize_step(s) for s in args.skip.split(",") if s.strip()}
    unknown_skips = [s for s in skip_set if s not in STEPS]
    if unknown_skips:
        raise SystemExit(f"Unknown steps in --skip: {', '.join(unknown_skips)}")

    scripts_dir = Path(__file__).resolve().parent
    steps_to_run = [s for s in STEPS[start_idx : end_idx + 1] if s not in skip_set]

    if not steps_to_run:
        print("No steps to run.")
        return

    for step in steps_to_run:
        script_path = scripts_dir / f"{step}.py"
        print(f"Running {step}...")
        subprocess.run([sys.executable, str(script_path)], check=True)


if __name__ == "__main__":
    main()
