from __future__ import annotations

"""Build the ARC-Challenge train-only packet/receiver replication gate.

This reuses the OpenBookQA receiver implementation on the frozen ARC-Challenge
train/validation/test splits and the promoted 12B hashed common-basis source
packet.  The selector is chosen on validation only and evaluated once on test.
"""

import argparse
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_openbookqa_receiver_headroom_gate as receiver_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_arc_challenge_receiver_headroom_gate_20260502")
DEFAULT_TRAIN = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_train.jsonl"
)
DEFAULT_VALIDATION = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_validation.jsonl"
)
DEFAULT_TEST = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_test.jsonl"
)
DEFAULT_VALIDATION_SOURCE_CACHE = pathlib.Path(
    "results/source_private_arc_challenge_seed_stability_20260501_qwen05_hashed_validation/"
    "source_prediction_cache.jsonl"
)
DEFAULT_TEST_SOURCE_CACHE = pathlib.Path(
    "results/source_private_arc_challenge_seed_stability_20260501_qwen05_hashed_test/"
    "source_prediction_cache.jsonl"
)


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation-path", type=pathlib.Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test-path", type=pathlib.Path, default=DEFAULT_TEST)
    parser.add_argument("--validation-source-cache", type=pathlib.Path, default=DEFAULT_VALIDATION_SOURCE_CACHE)
    parser.add_argument("--test-source-cache", type=pathlib.Path, default=DEFAULT_TEST_SOURCE_CACHE)
    parser.add_argument("--seeds", type=receiver_gate._parse_int_list, default="47,53,59,61,67")
    parser.add_argument("--budget-bytes", type=int, default=12)
    parser.add_argument("--packet-feature-dim", type=int, default=384)
    parser.add_argument("--code-dim", type=int, default=96)
    parser.add_argument("--target-feature-dim", type=int, default=1536)
    parser.add_argument("--target-ridge", type=float, default=1.0)
    parser.add_argument("--selector-ridges", type=receiver_gate._parse_float_list, default="0.01,0.1,1,10,100")
    parser.add_argument(
        "--threshold-percentiles",
        type=receiver_gate._parse_int_list,
        default="0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--min-receiver-lift", type=float, default=0.005)
    parser.add_argument("--min-control-gap", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    receiver_gate.build_receiver_gate(
        output_dir=_resolve(args.output_dir),
        train_path=_resolve(args.train_path),
        validation_path=_resolve(args.validation_path),
        test_path=_resolve(args.test_path),
        validation_source_cache=_resolve(args.validation_source_cache),
        test_source_cache=_resolve(args.test_source_cache),
        seeds=args.seeds,
        budget_bytes=args.budget_bytes,
        packet_feature_dim=args.packet_feature_dim,
        code_dim=args.code_dim,
        target_feature_dim=args.target_feature_dim,
        target_ridge=args.target_ridge,
        selector_ridges=args.selector_ridges,
        threshold_percentiles=args.threshold_percentiles,
        bootstrap_samples=args.bootstrap_samples,
        min_receiver_lift=args.min_receiver_lift,
        min_control_gap=args.min_control_gap,
        benchmark_name="ARC-Challenge",
        gate_name="source_private_arc_challenge_receiver_headroom_gate",
        output_stem="arc_challenge_receiver_headroom_gate",
        source_packet_origin=(
            "answer-key-forbidden Qwen2.5-0.5B source-choice cache from the promoted 12B "
            "ARC-Challenge hashed common-basis packet gate"
        ),
        receiver_training=(
            "target public scorer trained on ARC-Challenge train; selector selected on validation only; "
            "test labels held out until final evaluation"
        ),
        claim_boundary=(
            "This is an ARC replication of the packet/target evidence-fusion receiver. It is not a native "
            "GPU systems result and does not claim source-label-copy separation because the promoted packet "
            "is still a compact source-selected-candidate sketch."
        ),
        interpretation=(
            "ARC-Challenge tests whether the OpenBookQA receiver-fusion method generalizes back to the "
            "primary benchmark surface. A pass would upgrade the method from an OpenBookQA-only positive row "
            "to a cross-science-QA evidence-fusion branch; a fail would demote receiver-fusion as benchmark-"
            "specific calibration and make common-basis or hidden-innovation compression the next live branch."
        ),
    )


if __name__ == "__main__":
    main()
