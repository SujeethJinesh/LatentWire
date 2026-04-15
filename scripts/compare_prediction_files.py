"""Compare two RotAlign prediction JSONL files on paired examples."""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from latent_bridge.prediction_compare import (
    compare_prediction_files,
    compare_prediction_records,
    load_prediction_records,
    write_jsonl,
    write_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, help="Prediction JSONL for the candidate condition.")
    parser.add_argument("--baseline", required=True, help="Prediction JSONL for the baseline condition.")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-method", help="Single candidate method to compare.")
    parser.add_argument("--baseline-method", help="Single baseline method to compare.")
    parser.add_argument("--methods", nargs="+", help="Exact method names to compare.")
    parser.add_argument("--method-prefix", help="Compare all common methods with this prefix.")
    parser.add_argument("--include-baseline-methods", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--output-jsonl")
    parser.add_argument("--output-md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_method:
        rows = [
            compare_prediction_records(
                load_prediction_records(args.candidate),
                load_prediction_records(args.baseline),
                method=args.candidate_method,
                baseline_method=args.baseline_method,
                candidate_label=args.candidate_label,
                baseline_label=args.baseline_label,
                n_bootstrap=args.n_bootstrap,
            )
        ]
    else:
        rows = compare_prediction_files(
            args.candidate,
            args.baseline,
            methods=args.methods,
            method_prefix=args.method_prefix,
            include_baseline_methods=args.include_baseline_methods,
            candidate_label=args.candidate_label,
            baseline_label=args.baseline_label,
            n_bootstrap=args.n_bootstrap,
        )
    if args.output_jsonl:
        write_jsonl(rows, args.output_jsonl)
    if args.output_md:
        write_markdown(rows, args.output_md)
    if not args.output_jsonl and not args.output_md:
        for row in rows:
            print(row)


if __name__ == "__main__":
    main()
