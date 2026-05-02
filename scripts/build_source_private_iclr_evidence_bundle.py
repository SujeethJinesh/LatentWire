from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
import stat
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]


REPRODUCTION_COMMANDS = [
    "./venv_arm64/bin/python scripts/build_source_private_candidate_local_competitor_basis_table.py --output-dir results/source_private_candidate_local_competitor_basis_table_20260430",
    "./venv_arm64/bin/python scripts/build_source_private_candidate_local_cross_family_gate.py --output-dir results/source_private_candidate_local_cross_family_gate_20260430",
    "./venv_arm64/bin/python scripts/build_source_private_candidate_local_systems_boundary_trace.py --output-dir results/source_private_candidate_local_systems_boundary_trace_20260430",
    "./venv_arm64/bin/python scripts/build_source_private_candidate_local_threshold_frontier.py --output-dir results/source_private_candidate_local_threshold_frontier_20260501",
    "./venv_arm64/bin/python scripts/build_source_private_candidate_local_margin_atlas.py --output-dir results/source_private_candidate_local_margin_atlas_20260501",
    "./venv_arm64/bin/python scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py --output-dir results/source_private_candidate_conditioned_packet_builder_smoke_20260501 --budgets 8 --train-examples 1024 --eval-examples 512 --seed 47 --candidate-atom-view heldout_synonym --calibration-atom-view synonym_stress --candidate-calibration all_public_eval_disjoint --packet-builder-calibration all_public_eval_disjoint --calibration-examples 1024 --packet-builder-examples 1024 --feature-dim 384 --ridge 0.05 --packet-builder-ridge 0.1 --top-k 8 --min-score 0.0 --packet-min-score 0.0 --text-feature-mode hf_last_mean --adapter-target-mode semantic_anchor_teacher --decoder-score-mode candidate_local_residual_norm --min-decision-score 0.48",
    "./venv_arm64/bin/python scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py --output-dir results/source_private_candidate_conditioned_packet_builder_smoke_20260501_seed53 --budgets 8 --train-examples 1024 --eval-examples 512 --seed 53 --candidate-atom-view heldout_synonym --calibration-atom-view synonym_stress --candidate-calibration all_public_eval_disjoint --packet-builder-calibration all_public_eval_disjoint --calibration-examples 1024 --packet-builder-examples 1024 --feature-dim 384 --ridge 0.05 --packet-builder-ridge 0.1 --top-k 8 --min-score 0.0 --packet-min-score 0.0 --text-feature-mode hf_last_mean --adapter-target-mode semantic_anchor_teacher --decoder-score-mode candidate_local_residual_norm --min-decision-score 0.48",
    "./venv_arm64/bin/python scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py --output-dir results/source_private_candidate_conditioned_packet_builder_smoke_20260501_seed59 --budgets 8 --train-examples 1024 --eval-examples 512 --seed 59 --candidate-atom-view heldout_synonym --calibration-atom-view synonym_stress --candidate-calibration all_public_eval_disjoint --packet-builder-calibration all_public_eval_disjoint --calibration-examples 1024 --packet-builder-examples 1024 --feature-dim 384 --ridge 0.05 --packet-builder-ridge 0.1 --top-k 8 --min-score 0.0 --packet-min-score 0.0 --text-feature-mode hf_last_mean --adapter-target-mode semantic_anchor_teacher --decoder-score-mode candidate_local_residual_norm --min-decision-score 0.48",
    "./venv_arm64/bin/python scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py --output-dir results/source_private_candidate_conditioned_packet_builder_loo_hybrid_w075_20260501 --budgets 12 --seed 47 --packet-builder-calibration leave_one_family_out_public --packet-builder-composition add_source --source-identity-weight 0.75 --bootstrap-samples 500",
    "./venv_arm64/bin/python scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py --output-dir results/source_private_candidate_conditioned_packet_builder_loo_hybrid_w075_20260501_seed53 --budgets 12 --seed 53 --packet-builder-calibration leave_one_family_out_public --packet-builder-composition add_source --source-identity-weight 0.75 --bootstrap-samples 500",
    "./venv_arm64/bin/python scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py --output-dir results/source_private_candidate_conditioned_packet_builder_loo_hybrid_w075_20260501_seed59_serial --budgets 12 --seed 59 --packet-builder-calibration leave_one_family_out_public --packet-builder-composition add_source --source-identity-weight 0.75 --bootstrap-samples 500",
    "./venv_arm64/bin/python scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py --output-dir results/source_private_candidate_conditioned_packet_builder_train_builder_hybrid_w075_20260501 --budgets 12 --seed 47 --candidate-calibration all_public_eval_disjoint --packet-builder-calibration train_only --packet-builder-composition add_source --source-identity-weight 0.75 --bootstrap-samples 500",
    "./venv_arm64/bin/python scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py --output-dir results/source_private_candidate_conditioned_packet_builder_train_builder_hybrid_w075_20260501_seed53 --budgets 12 --seed 53 --candidate-calibration all_public_eval_disjoint --packet-builder-calibration train_only --packet-builder-composition add_source --source-identity-weight 0.75 --bootstrap-samples 500",
    "./venv_arm64/bin/python scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py --output-dir results/source_private_candidate_conditioned_packet_builder_train_builder_hybrid_w075_20260501_seed59 --budgets 12 --seed 59 --candidate-calibration all_public_eval_disjoint --packet-builder-calibration train_only --packet-builder-composition add_source --source-identity-weight 0.75 --bootstrap-samples 500",
    "./venv_arm64/bin/python scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py --output-dir results/source_private_candidate_conditioned_packet_builder_train_builder_hybrid_w075_rate_20260501 --budgets 8 10 12 16 --seed 47 --candidate-calibration all_public_eval_disjoint --packet-builder-calibration train_only --packet-builder-composition add_source --source-identity-weight 0.75 --bootstrap-samples 500",
    "./venv_arm64/bin/python scripts/run_source_private_learned_synonym_dictionary_packet_gate.py --output-dir results/source_private_train_only_receiver_permuted_null_gap_20260501_seed47_n512 --budgets 12 --train-examples 768 --eval-examples 512 --seed 47 --candidate-atom-view heldout_synonym --calibration-atom-view synonym_stress --candidate-calibration train_only --calibration-examples 768 --feature-dim 384 --ridge 0.05 --top-k 8 --min-score 0.0 --text-feature-mode semantic_anchor --adapter-target-mode semantic_anchor_teacher --decoder-score-mode candidate_local_permuted_null_gap_residual_norm --permuted-null-weight 0.75 --min-decision-score 0.30",
    "./venv_arm64/bin/python scripts/run_source_private_learned_synonym_dictionary_packet_gate.py --output-dir results/source_private_train_only_receiver_permuted_null_gap_20260501_seed53_n512 --budgets 12 --train-examples 768 --eval-examples 512 --seed 53 --candidate-atom-view heldout_synonym --calibration-atom-view synonym_stress --candidate-calibration train_only --calibration-examples 768 --feature-dim 384 --ridge 0.05 --top-k 8 --min-score 0.0 --text-feature-mode semantic_anchor --adapter-target-mode semantic_anchor_teacher --decoder-score-mode candidate_local_permuted_null_gap_residual_norm --permuted-null-weight 0.75 --min-decision-score 0.30",
    "./venv_arm64/bin/python scripts/run_source_private_learned_synonym_dictionary_packet_gate.py --output-dir results/source_private_train_only_receiver_permuted_null_gap_20260501_seed59_n512 --budgets 12 --train-examples 768 --eval-examples 512 --seed 59 --candidate-atom-view heldout_synonym --calibration-atom-view synonym_stress --candidate-calibration train_only --calibration-examples 768 --feature-dim 384 --ridge 0.05 --top-k 8 --min-score 0.0 --text-feature-mode semantic_anchor --adapter-target-mode semantic_anchor_teacher --decoder-score-mode candidate_local_permuted_null_gap_residual_norm --permuted-null-weight 0.75 --min-decision-score 0.30",
    "./venv_arm64/bin/python scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py --output-dir results/source_private_train_donor_antishuffle_seed47_n128_budget14 --budgets 14 --train-examples 256 --eval-examples 128 --seed 47 --candidate-atom-view heldout_synonym --calibration-atom-view synonym_stress --candidate-calibration train_only --packet-builder-calibration train_only --calibration-examples 256 --packet-builder-examples 256 --feature-dim 384 --ridge 0.05 --packet-builder-ridge 0.1 --top-k 8 --min-score 0.0 --packet-min-score 0.0 --text-feature-mode semantic_anchor --adapter-target-mode semantic_anchor_teacher --decoder-score-mode candidate_local_permuted_null_gap_residual_norm --permuted-null-weight 0.75 --packet-builder-target-mode answer_minus_candidate_mean --packet-builder-composition train_donor_antishuffle_innovation --source-identity-weight 0.75 --antishuffle-train-donors 12 --antishuffle-donor-weight 0.50 --antishuffle-null-weight 0.75 --antishuffle-generic-weight 0.10 --antishuffle-carrier-mode sum --min-decision-score 0.30",
    "./venv_arm64/bin/python scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py --output-dir results/source_private_train_donor_antishuffle_seed53_n128_budget12 --budgets 12 --train-examples 256 --eval-examples 128 --seed 53 --candidate-atom-view heldout_synonym --calibration-atom-view synonym_stress --candidate-calibration train_only --packet-builder-calibration train_only --calibration-examples 256 --packet-builder-examples 256 --feature-dim 384 --ridge 0.05 --packet-builder-ridge 0.1 --top-k 8 --min-score 0.0 --packet-min-score 0.0 --text-feature-mode semantic_anchor --adapter-target-mode semantic_anchor_teacher --decoder-score-mode candidate_local_permuted_null_gap_residual_norm --permuted-null-weight 0.75 --packet-builder-target-mode answer_minus_candidate_mean --packet-builder-composition train_donor_antishuffle_innovation --source-identity-weight 0.75 --antishuffle-train-donors 12 --antishuffle-donor-weight 0.50 --antishuffle-null-weight 0.75 --antishuffle-generic-weight 0.10 --antishuffle-carrier-mode sum --min-decision-score 0.30",
    "./venv_arm64/bin/python scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py --output-dir results/source_private_train_donor_antishuffle_seed59_n128_budget12 --budgets 12 --train-examples 256 --eval-examples 128 --seed 59 --candidate-atom-view heldout_synonym --calibration-atom-view synonym_stress --candidate-calibration train_only --packet-builder-calibration train_only --calibration-examples 256 --packet-builder-examples 256 --feature-dim 384 --ridge 0.05 --packet-builder-ridge 0.1 --top-k 8 --min-score 0.0 --packet-min-score 0.0 --text-feature-mode semantic_anchor --adapter-target-mode semantic_anchor_teacher --decoder-score-mode candidate_local_permuted_null_gap_residual_norm --permuted-null-weight 0.75 --packet-builder-target-mode answer_minus_candidate_mean --packet-builder-composition train_donor_antishuffle_innovation --source-identity-weight 0.75 --antishuffle-train-donors 12 --antishuffle-donor-weight 0.50 --antishuffle-null-weight 0.75 --antishuffle-generic-weight 0.10 --antishuffle-carrier-mode sum --min-decision-score 0.30",
    "./venv_arm64/bin/python scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py --output-dir results/source_private_train_donor_antishuffle_seed47_n512_budget14 --budgets 14 --train-examples 1024 --eval-examples 512 --seed 47 --candidate-atom-view heldout_synonym --calibration-atom-view synonym_stress --candidate-calibration train_only --packet-builder-calibration train_only --calibration-examples 1024 --packet-builder-examples 1024 --feature-dim 384 --ridge 0.05 --packet-builder-ridge 0.1 --top-k 8 --min-score 0.0 --packet-min-score 0.0 --text-feature-mode semantic_anchor --adapter-target-mode semantic_anchor_teacher --decoder-score-mode candidate_local_permuted_null_gap_residual_norm --permuted-null-weight 0.75 --packet-builder-target-mode answer_minus_candidate_mean --packet-builder-composition train_donor_antishuffle_innovation --source-identity-weight 0.75 --antishuffle-train-donors 12 --antishuffle-donor-weight 0.50 --antishuffle-null-weight 0.75 --antishuffle-generic-weight 0.10 --antishuffle-carrier-mode sum --min-decision-score 0.30",
    "./venv_arm64/bin/python scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py --output-dir results/source_private_train_donor_antishuffle_seed47_n512_budget12_cross --budgets 12 --directions core_to_holdout holdout_to_core --train-examples 1024 --eval-examples 512 --seed 47 --candidate-atom-view heldout_synonym --calibration-atom-view synonym_stress --candidate-calibration train_only --packet-builder-calibration train_only --calibration-examples 1024 --packet-builder-examples 1024 --feature-dim 384 --ridge 0.05 --packet-builder-ridge 0.1 --top-k 8 --min-score 0.0 --packet-min-score 0.0 --text-feature-mode semantic_anchor --adapter-target-mode semantic_anchor_teacher --decoder-score-mode candidate_local_permuted_null_gap_residual_norm --permuted-null-weight 0.75 --packet-builder-target-mode answer_minus_candidate_mean --packet-builder-composition train_donor_antishuffle_innovation --source-identity-weight 0.75 --antishuffle-train-donors 12 --antishuffle-donor-weight 0.50 --antishuffle-null-weight 0.75 --antishuffle-generic-weight 0.10 --antishuffle-carrier-mode sum --min-decision-score 0.30 --bootstrap-samples 500",
    "./venv_arm64/bin/python scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py --output-dir results/source_private_train_donor_antishuffle_seed53_n512_budget12_14 --budgets 12 14 --train-examples 1024 --eval-examples 512 --seed 53 --candidate-atom-view heldout_synonym --calibration-atom-view synonym_stress --candidate-calibration train_only --packet-builder-calibration train_only --calibration-examples 1024 --packet-builder-examples 1024 --feature-dim 384 --ridge 0.05 --packet-builder-ridge 0.1 --top-k 8 --min-score 0.0 --packet-min-score 0.0 --text-feature-mode semantic_anchor --adapter-target-mode semantic_anchor_teacher --decoder-score-mode candidate_local_permuted_null_gap_residual_norm --permuted-null-weight 0.75 --packet-builder-target-mode answer_minus_candidate_mean --packet-builder-composition train_donor_antishuffle_innovation --source-identity-weight 0.75 --antishuffle-train-donors 12 --antishuffle-donor-weight 0.50 --antishuffle-null-weight 0.75 --antishuffle-generic-weight 0.10 --antishuffle-carrier-mode sum --min-decision-score 0.30",
    "./venv_arm64/bin/python scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py --output-dir results/source_private_train_donor_antishuffle_seed59_n512_budget12_14 --budgets 12 14 --train-examples 1024 --eval-examples 512 --seed 59 --candidate-atom-view heldout_synonym --calibration-atom-view synonym_stress --candidate-calibration train_only --packet-builder-calibration train_only --calibration-examples 1024 --packet-builder-examples 1024 --feature-dim 384 --ridge 0.05 --packet-builder-ridge 0.1 --top-k 8 --min-score 0.0 --packet-min-score 0.0 --text-feature-mode semantic_anchor --adapter-target-mode semantic_anchor_teacher --decoder-score-mode candidate_local_permuted_null_gap_residual_norm --permuted-null-weight 0.75 --packet-builder-target-mode answer_minus_candidate_mean --packet-builder-composition train_donor_antishuffle_innovation --source-identity-weight 0.75 --antishuffle-train-donors 12 --antishuffle-donor-weight 0.50 --antishuffle-null-weight 0.75 --antishuffle-generic-weight 0.10 --antishuffle-carrier-mode sum --min-decision-score 0.30",
    "./venv_arm64/bin/python scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py --output-dir results/source_private_train_donor_antishuffle_locked_frontier_seed47_n128 --budgets 10 12 14 16 --directions core_to_holdout holdout_to_core --train-examples 256 --eval-examples 128 --seed 47 --candidate-atom-view heldout_synonym --calibration-atom-view synonym_stress --candidate-calibration train_only --packet-builder-calibration train_only --calibration-examples 256 --packet-builder-examples 256 --feature-dim 384 --ridge 0.05 --packet-builder-ridge 0.1 --top-k 8 --min-score 0.0 --packet-min-score 0.0 --text-feature-mode semantic_anchor --adapter-target-mode semantic_anchor_teacher --decoder-score-mode candidate_local_permuted_null_gap_residual_norm --permuted-null-weight 0.75 --packet-builder-target-mode answer_minus_candidate_mean --packet-builder-composition train_donor_antishuffle_innovation --source-identity-weight 0.75 --antishuffle-train-donors 12 --antishuffle-donor-weight 0.50 --antishuffle-null-weight 0.75 --antishuffle-generic-weight 0.10 --antishuffle-carrier-mode sum --min-decision-score 0.30 --bootstrap-samples 500",
    "./venv_arm64/bin/python scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py --output-dir results/source_private_train_donor_antishuffle_seed53_n128_budget10 --budgets 10 --directions core_to_holdout holdout_to_core --train-examples 256 --eval-examples 128 --seed 53 --candidate-atom-view heldout_synonym --calibration-atom-view synonym_stress --candidate-calibration train_only --packet-builder-calibration train_only --calibration-examples 256 --packet-builder-examples 256 --feature-dim 384 --ridge 0.05 --packet-builder-ridge 0.1 --top-k 8 --min-score 0.0 --packet-min-score 0.0 --text-feature-mode semantic_anchor --adapter-target-mode semantic_anchor_teacher --decoder-score-mode candidate_local_permuted_null_gap_residual_norm --permuted-null-weight 0.75 --packet-builder-target-mode answer_minus_candidate_mean --packet-builder-composition train_donor_antishuffle_innovation --source-identity-weight 0.75 --antishuffle-train-donors 12 --antishuffle-donor-weight 0.50 --antishuffle-null-weight 0.75 --antishuffle-generic-weight 0.10 --antishuffle-carrier-mode sum --min-decision-score 0.30 --bootstrap-samples 500",
    "./venv_arm64/bin/python scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py --output-dir results/source_private_train_donor_antishuffle_seed59_n128_budget10 --budgets 10 --directions core_to_holdout holdout_to_core --train-examples 256 --eval-examples 128 --seed 59 --candidate-atom-view heldout_synonym --calibration-atom-view synonym_stress --candidate-calibration train_only --packet-builder-calibration train_only --calibration-examples 256 --packet-builder-examples 256 --feature-dim 384 --ridge 0.05 --packet-builder-ridge 0.1 --top-k 8 --min-score 0.0 --packet-min-score 0.0 --text-feature-mode semantic_anchor --adapter-target-mode semantic_anchor_teacher --decoder-score-mode candidate_local_permuted_null_gap_residual_norm --permuted-null-weight 0.75 --packet-builder-target-mode answer_minus_candidate_mean --packet-builder-composition train_donor_antishuffle_innovation --source-identity-weight 0.75 --antishuffle-train-donors 12 --antishuffle-donor-weight 0.50 --antishuffle-null-weight 0.75 --antishuffle-generic-weight 0.10 --antishuffle-carrier-mode sum --min-decision-score 0.30 --bootstrap-samples 500",
    "./venv_arm64/bin/python scripts/build_source_private_train_donor_locked_rate_frontier.py --output-dir results/source_private_train_donor_antishuffle_locked_rate_frontier_20260501 --selection-scope per_seed",
    "./venv_arm64/bin/python scripts/build_source_private_train_donor_locked_rate_frontier.py --output-dir results/source_private_train_donor_antishuffle_stable_gap_seed47_53_59_20260501 --selection-scope global --validation-control-scope source_private_gap --validation-selector stable_interior --validation-run results/source_private_train_donor_antishuffle_train_family_disjoint_seed47_n128 --validation-run results/source_private_train_donor_antishuffle_train_family_disjoint_seed53_n128_train256 --validation-run results/source_private_train_donor_antishuffle_train_family_disjoint_seed59_n128_train256 --eval-run results/source_private_train_donor_antishuffle_seed47_n512_budget12_cross --eval-run results/source_private_train_donor_antishuffle_seed53_n512_budget12_14 --eval-run results/source_private_train_donor_antishuffle_seed59_n512_budget12_14",
    "./venv_arm64/bin/python scripts/build_source_private_train_donor_fixed_budget_eval_audit.py --output-dir results/source_private_train_donor_antishuffle_fixed12b_eval_audit_20260501",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers ./venv_arm64/bin/python scripts/build_source_private_arc_challenge_bridge_contract.py --output-dir results/source_private_arc_challenge_bridge_contract_20260501 --materialize-official --run-date 2026-05-01",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers ./venv_arm64/bin/python scripts/run_source_private_arc_challenge_fixed_packet_gate.py --output-dir results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_bge_validation --feature-mode hf_last_mean --feature-model /Users/sujeethjinesh/.cache/huggingface/hub/models--BAAI--bge-small-en/snapshots/2275a7bdee235e9b4f01fa73aa60d3311983cfea --feature-device auto --feature-dtype float32 --feature-max-length 128 --feature-dim 384 --code-dim 96 --budget-bytes 12 --source-score-mode lm_choice_loglikelihood --source-lm-model /Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 --source-lm-device auto_cpu --source-lm-dtype float32 --source-lm-max-length 256 --source-lm-normalization mean --seed 47 --bootstrap-samples 500",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers ./venv_arm64/bin/python scripts/run_source_private_arc_challenge_fixed_packet_gate.py --output-dir results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_bge_test --eval-path results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_test.jsonl --feature-mode hf_last_mean --feature-model /Users/sujeethjinesh/.cache/huggingface/hub/models--BAAI--bge-small-en/snapshots/2275a7bdee235e9b4f01fa73aa60d3311983cfea --feature-device auto --feature-dtype float32 --feature-max-length 128 --feature-dim 384 --code-dim 96 --budget-bytes 12 --source-score-mode lm_choice_loglikelihood --source-lm-model /Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 --source-lm-device auto_cpu --source-lm-dtype float32 --source-lm-max-length 256 --source-lm-normalization mean --seed 47 --bootstrap-samples 500",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers ./venv_arm64/bin/python scripts/run_source_private_arc_challenge_fixed_packet_gate.py --output-dir results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_validation --feature-mode hashed --feature-dim 384 --code-dim 96 --budget-bytes 12 --source-score-mode lm_choice_loglikelihood --source-lm-model /Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 --source-lm-device auto_cpu --source-lm-dtype float32 --source-lm-max-length 256 --source-lm-normalization mean --seed 47 --bootstrap-samples 500",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers ./venv_arm64/bin/python scripts/run_source_private_arc_challenge_fixed_packet_gate.py --output-dir results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_test --eval-path results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_test.jsonl --feature-mode hashed --feature-dim 384 --code-dim 96 --budget-bytes 12 --source-score-mode lm_choice_loglikelihood --source-lm-model /Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 --source-lm-device auto_cpu --source-lm-dtype float32 --source-lm-max-length 256 --source-lm-normalization mean --seed 47 --bootstrap-samples 500",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers ./venv_arm64/bin/python scripts/build_source_private_arc_challenge_seed_stability.py --output-dir results/source_private_arc_challenge_seed_stability_20260501_qwen05_bge_validation --split-name validation --anchor-predictions results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_bge_validation/predictions.jsonl --feature-mode hf_last_mean --feature-model /Users/sujeethjinesh/.cache/huggingface/hub/models--BAAI--bge-small-en/snapshots/2275a7bdee235e9b4f01fa73aa60d3311983cfea --feature-device auto --feature-dtype float32 --feature-max-length 128 --feature-dim 384 --code-dim 96 --budget-bytes 12 --seeds 47,53,59,61,67 --bootstrap-samples 500",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers ./venv_arm64/bin/python scripts/build_source_private_arc_challenge_seed_stability.py --output-dir results/source_private_arc_challenge_seed_stability_20260501_qwen05_bge_test --split-name test --eval-path results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_test.jsonl --anchor-predictions results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_bge_test/predictions.jsonl --feature-mode hf_last_mean --feature-model /Users/sujeethjinesh/.cache/huggingface/hub/models--BAAI--bge-small-en/snapshots/2275a7bdee235e9b4f01fa73aa60d3311983cfea --feature-device auto --feature-dtype float32 --feature-max-length 128 --feature-dim 384 --code-dim 96 --budget-bytes 12 --seeds 47,53,59,61,67 --bootstrap-samples 500",
    "./venv_arm64/bin/python scripts/build_source_private_arc_challenge_seed_stability.py --output-dir results/source_private_arc_challenge_seed_stability_20260501_qwen05_hashed_validation --split-name validation_hashed_common_basis --anchor-predictions results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_validation/predictions.jsonl --feature-mode hashed --feature-dim 384 --code-dim 96 --budget-bytes 12 --seeds 47,53,59,61,67 --bootstrap-samples 500",
    "./venv_arm64/bin/python scripts/build_source_private_arc_challenge_seed_stability.py --output-dir results/source_private_arc_challenge_seed_stability_20260501_qwen05_hashed_test --split-name test_hashed_common_basis --eval-path results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_test.jsonl --anchor-predictions results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_test/predictions.jsonl --feature-mode hashed --feature-dim 384 --code-dim 96 --budget-bytes 12 --seeds 47,53,59,61,67 --bootstrap-samples 500",
    "./venv_arm64/bin/python scripts/build_source_private_arc_challenge_seed_stability.py --output-dir results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_relative_validation --split-name validation_anchor_relative_common_basis --anchor-predictions results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_validation/predictions.jsonl --feature-mode anchor_relative_hashed --feature-dim 384 --code-dim 96 --budget-bytes 12 --seeds 47,53,59,61,67 --bootstrap-samples 500",
    "./venv_arm64/bin/python scripts/build_source_private_arc_challenge_seed_stability.py --output-dir results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_relative_test --split-name test_anchor_relative_common_basis --eval-path results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_test.jsonl --anchor-predictions results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_test/predictions.jsonl --feature-mode anchor_relative_hashed --feature-dim 384 --code-dim 96 --budget-bytes 12 --seeds 47,53,59,61,67 --bootstrap-samples 500",
    "./venv_arm64/bin/python scripts/build_source_private_arc_challenge_seed_stability.py --output-dir results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_id_shuffle_validation --split-name validation_anchor_id_shuffle --anchor-predictions results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_validation/predictions.jsonl --feature-mode anchor_relative_hashed --anchor-control anchor_id_shuffle --feature-dim 384 --code-dim 96 --budget-bytes 12 --seeds 47,53,59,61,67 --bootstrap-samples 500",
    "./venv_arm64/bin/python scripts/build_source_private_arc_challenge_seed_stability.py --output-dir results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_id_shuffle_test --split-name test_anchor_id_shuffle --eval-path results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_test.jsonl --anchor-predictions results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_test/predictions.jsonl --feature-mode anchor_relative_hashed --anchor-control anchor_id_shuffle --feature-dim 384 --code-dim 96 --budget-bytes 12 --seeds 47,53,59,61,67 --bootstrap-samples 500",
    "./venv_arm64/bin/python scripts/build_source_private_arc_challenge_seed_stability.py --output-dir results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_value_shuffle_validation --split-name validation_anchor_value_shuffle --anchor-predictions results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_validation/predictions.jsonl --feature-mode anchor_relative_hashed --anchor-control anchor_value_shuffle --feature-dim 384 --code-dim 96 --budget-bytes 12 --seeds 47,53,59,61,67 --bootstrap-samples 500",
    "./venv_arm64/bin/python scripts/build_source_private_arc_challenge_seed_stability.py --output-dir results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_value_shuffle_test --split-name test_anchor_value_shuffle --eval-path results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_test.jsonl --anchor-predictions results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_test/predictions.jsonl --feature-mode anchor_relative_hashed --anchor-control anchor_value_shuffle --feature-dim 384 --code-dim 96 --budget-bytes 12 --seeds 47,53,59,61,67 --bootstrap-samples 500",
    "./venv_arm64/bin/python scripts/build_source_private_arc_challenge_seed_stability.py --output-dir results/source_private_arc_challenge_seed_stability_20260501_qwen05_random_anchors_validation --split-name validation_random_anchors_same_count --anchor-predictions results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_validation/predictions.jsonl --feature-mode anchor_relative_hashed --anchor-control random_anchors_same_count --feature-dim 384 --code-dim 96 --budget-bytes 12 --seeds 47,53,59,61,67 --bootstrap-samples 500",
    "./venv_arm64/bin/python scripts/build_source_private_arc_challenge_seed_stability.py --output-dir results/source_private_arc_challenge_seed_stability_20260501_qwen05_random_anchors_test --split-name test_random_anchors_same_count --eval-path results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_test.jsonl --anchor-predictions results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_test/predictions.jsonl --feature-mode anchor_relative_hashed --anchor-control random_anchors_same_count --feature-dim 384 --code-dim 96 --budget-bytes 12 --seeds 47,53,59,61,67 --bootstrap-samples 500",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers ./venv_arm64/bin/python scripts/run_source_private_arc_challenge_source_latent_endpoint_gate.py --output-dir results/source_private_arc_challenge_source_latent_endpoint_gate_20260501_qwen05_bge_validation --target-feature-mode hf_last_mean --target-feature-model /Users/sujeethjinesh/.cache/huggingface/hub/models--BAAI--bge-small-en/snapshots/2275a7bdee235e9b4f01fa73aa60d3311983cfea --target-feature-device auto --target-feature-dtype float32 --target-feature-max-length 128 --target-feature-dim 384 --code-dim 96 --budget-bytes 12 --source-feature-mode lm_hidden_last_mean --source-lm-model /Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 --source-lm-device auto_cpu --source-lm-dtype float32 --source-lm-max-length 256 --source-lm-normalization mean --source-hidden-layer -1 --source-score-mode lm_choice_loglikelihood --alignment-ridge 10.0 --seed 47 --bootstrap-samples 500",
    "./venv_arm64/bin/python scripts/build_source_private_arc_challenge_systems_trace.py --output-dir results/source_private_arc_challenge_systems_trace_20260501",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers ./venv_arm64/bin/python scripts/build_source_private_sciq_bridge_contract.py --output-dir results/source_private_sciq_bridge_contract_20260501 --materialize-official --run-date 2026-05-01",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers ./venv_arm64/bin/python scripts/run_source_private_arc_challenge_fixed_packet_gate.py --output-dir results/source_private_sciq_fixed_packet_gate_20260501_qwen05_hashed_validation --train-path results/source_private_sciq_bridge_contract_20260501/official_splits/sciq_train.jsonl --eval-path results/source_private_sciq_bridge_contract_20260501/official_splits/sciq_validation.jsonl --feature-mode hashed --feature-dim 384 --code-dim 96 --budget-bytes 12 --source-score-mode lm_choice_loglikelihood --source-lm-model /Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 --source-lm-device auto_cpu --source-lm-dtype float32 --source-lm-max-length 256 --source-lm-normalization mean --seed 47 --bootstrap-samples 500",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers ./venv_arm64/bin/python scripts/build_source_private_openbookqa_bridge_contract.py --output-dir results/source_private_openbookqa_bridge_contract_20260501 --materialize-official --run-date 2026-05-01",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers ./venv_arm64/bin/python scripts/run_source_private_arc_challenge_fixed_packet_gate.py --output-dir results/source_private_openbookqa_fixed_packet_gate_20260501_qwen05_hashed_validation --train-path results/source_private_openbookqa_bridge_contract_20260501/official_splits/openbookqa_train.jsonl --eval-path results/source_private_openbookqa_bridge_contract_20260501/official_splits/openbookqa_validation.jsonl --feature-mode hashed --feature-dim 384 --code-dim 96 --budget-bytes 12 --source-score-mode lm_choice_loglikelihood --source-lm-model /Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 --source-lm-device auto_cpu --source-lm-dtype float32 --source-lm-max-length 256 --source-lm-normalization mean --seed 47 --bootstrap-samples 500 --min-gap-over-text 0.02",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers ./venv_arm64/bin/python scripts/run_source_private_arc_challenge_fixed_packet_gate.py --output-dir results/source_private_openbookqa_fixed_packet_gate_20260501_qwen05_hashed_test_4b --train-path results/source_private_openbookqa_bridge_contract_20260501/official_splits/openbookqa_train.jsonl --eval-path results/source_private_openbookqa_bridge_contract_20260501/official_splits/openbookqa_test.jsonl --feature-mode hashed --feature-dim 384 --code-dim 96 --budget-bytes 4 --source-score-mode lm_choice_loglikelihood --source-lm-model /Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 --source-lm-device auto_cpu --source-lm-dtype float32 --source-lm-max-length 256 --source-lm-normalization mean --seed 47 --bootstrap-samples 500 --min-gap-over-text 0.02",
    "./venv_arm64/bin/python scripts/build_source_private_arc_challenge_seed_stability.py --output-dir results/source_private_openbookqa_seed_stability_20260501_qwen05_hashed_validation_3b --train-path results/source_private_openbookqa_bridge_contract_20260501/official_splits/openbookqa_train.jsonl --eval-path results/source_private_openbookqa_bridge_contract_20260501/official_splits/openbookqa_validation.jsonl --anchor-predictions results/source_private_openbookqa_fixed_packet_gate_20260501_qwen05_hashed_validation/predictions.jsonl --split-name openbookqa_validation_hashed_3b --feature-mode hashed --feature-dim 384 --code-dim 96 --budget-bytes 3 --seeds 47,53,59,61,67 --bootstrap-samples 500 --min-gap-over-text 0.02",
    "./venv_arm64/bin/python scripts/build_source_private_arc_challenge_seed_stability.py --output-dir results/source_private_openbookqa_seed_stability_20260501_qwen05_hashed_test_3b --train-path results/source_private_openbookqa_bridge_contract_20260501/official_splits/openbookqa_train.jsonl --eval-path results/source_private_openbookqa_bridge_contract_20260501/official_splits/openbookqa_test.jsonl --anchor-predictions results/source_private_openbookqa_fixed_packet_gate_20260501_qwen05_hashed_test_4b/predictions.jsonl --split-name openbookqa_test_hashed_3b --feature-mode hashed --feature-dim 384 --code-dim 96 --budget-bytes 3 --seeds 47,53,59,61,67 --bootstrap-samples 500 --min-gap-over-text 0.02",
    "./venv_arm64/bin/python scripts/build_source_private_openbookqa_receiver_headroom_gate.py --output-dir results/source_private_openbookqa_receiver_headroom_gate_20260502",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers ./venv_arm64/bin/python scripts/build_source_private_commonsenseqa_bridge_contract.py --output-dir results/source_private_commonsenseqa_bridge_contract_20260501 --run-date 2026-05-01",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./venv_arm64/bin/python scripts/run_source_private_arc_challenge_fixed_packet_gate.py --output-dir results/source_private_commonsenseqa_fixed_packet_gate_20260501_qwen05_hashed_validation_12b --train-path results/source_private_commonsenseqa_bridge_contract_20260501/official_splits/commonsenseqa_train.jsonl --eval-path results/source_private_commonsenseqa_bridge_contract_20260501/official_splits/commonsenseqa_validation.jsonl --feature-mode hashed --feature-dim 384 --code-dim 96 --budget-bytes 12 --source-score-mode lm_choice_loglikelihood --source-lm-model /Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 --source-lm-device auto_cpu --source-lm-dtype float32 --source-lm-max-length 256 --source-lm-normalization mean --seed 47 --bootstrap-samples 500 --min-gap-over-text 0.02",
    "./venv_arm64/bin/python scripts/build_source_private_arc_challenge_seed_stability.py --output-dir results/source_private_commonsenseqa_seed_stability_20260501_qwen05_hashed_validation_2b --train-path results/source_private_commonsenseqa_bridge_contract_20260501/official_splits/commonsenseqa_train.jsonl --eval-path results/source_private_commonsenseqa_bridge_contract_20260501/official_splits/commonsenseqa_validation.jsonl --anchor-predictions results/source_private_commonsenseqa_fixed_packet_gate_20260501_qwen05_hashed_validation_12b/predictions.jsonl --split-name commonsenseqa_validation_hashed_2b --feature-mode hashed --feature-dim 384 --code-dim 96 --budget-bytes 2 --seeds 47,53,59,61,67 --bootstrap-samples 500 --min-gap-over-text 0.02",
    "./venv_arm64/bin/python scripts/build_source_private_arc_challenge_seed_stability.py --output-dir results/source_private_commonsenseqa_seed_stability_20260501_qwen05_hashed_validation_2b_gap001 --train-path results/source_private_commonsenseqa_bridge_contract_20260501/official_splits/commonsenseqa_train.jsonl --eval-path results/source_private_commonsenseqa_bridge_contract_20260501/official_splits/commonsenseqa_validation.jsonl --anchor-predictions results/source_private_commonsenseqa_fixed_packet_gate_20260501_qwen05_hashed_validation_12b/predictions.jsonl --split-name commonsenseqa_validation_hashed_2b_gap001 --feature-mode hashed --feature-dim 384 --code-dim 96 --budget-bytes 2 --seeds 47,53,59,61,67 --bootstrap-samples 500 --min-gap-over-text 0.01",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers ./venv_arm64/bin/python scripts/build_source_private_hellaswag_bridge_contract.py --output-dir results/source_private_hellaswag_bridge_contract_20260501 --hf-cache-dir .debug/hf_datasets --validation-slice-rows 1024 --run-date 2026-05-01",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./venv_arm64/bin/python scripts/run_source_private_arc_challenge_fixed_packet_gate.py --output-dir results/source_private_hellaswag_fixed_packet_gate_20260501_qwen05_hashed_validation1024_2b --train-path results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_train.jsonl --eval-path results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_validation_first1024.jsonl --budget-bytes 2 --feature-dim 384 --code-dim 96 --feature-mode hashed --source-score-mode lm_choice_loglikelihood --source-lm-model /Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 --source-lm-device auto_cpu --source-lm-dtype float32 --source-lm-max-length 256 --source-lm-normalization mean --source-lm-prompt-mode continuation --bootstrap-samples 500 --min-lift-over-target 0.03 --min-gap-over-control 0.03 --min-gap-over-text 0.02",
    "./venv_arm64/bin/python scripts/build_source_private_arc_challenge_seed_stability.py --output-dir results/source_private_hellaswag_seed_stability_20260501_qwen05_hashed_validation1024_2b_5seed --train-path results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_train.jsonl --eval-path results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_validation_first1024.jsonl --anchor-predictions results/source_private_hellaswag_fixed_packet_gate_20260501_qwen05_hashed_validation1024_2b/predictions.jsonl --split-name hellaswag_validation_first1024_hashed_2b --feature-mode hashed --feature-dim 384 --code-dim 96 --budget-bytes 2 --seeds 47,53,59,61,67 --bootstrap-samples 500 --min-lift-over-target 0.03 --min-gap-over-control 0.03 --min-gap-over-text 0.02",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers ./venv_arm64/bin/python scripts/build_source_private_hellaswag_control_suite.py --output-dir results/source_private_hellaswag_control_suite_20260501 --hf-cache-dir .debug/hf_datasets --bootstrap-samples 500 --run-date 2026-05-01",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./venv_arm64/bin/python scripts/build_source_private_hellaswag_score_packet_headroom.py --output-dir results/source_private_hellaswag_score_packet_headroom_20260501_qwen05_validation1024 --eval-path results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_validation_first1024.jsonl --score-cache results/source_private_hellaswag_score_packet_headroom_20260501_qwen05_validation1024/source_score_cache.json --source-lm-model /Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 --source-lm-device auto_cpu --source-lm-dtype float32 --source-lm-max-length 256 --source-lm-normalization mean --source-lm-prompt-mode continuation --local-files-only --run-date 2026-05-01",
    "./venv_arm64/bin/python scripts/build_source_private_hellaswag_public_receiver_repair_probe.py --output-dir results/source_private_hellaswag_public_receiver_repair_probe_20260501_qwen05_validation1024 --run-date 2026-05-01",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./venv_arm64/bin/python scripts/build_source_private_hellaswag_train_source_score_repair_probe.py --output-dir results/source_private_hellaswag_train_source_score_repair_probe_20260501_qwen05_train512_validation1024 --train-score-rows 512 --source-lm-model /Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 --source-lm-device auto_cpu --source-lm-dtype float32 --source-lm-max-length 256 --source-lm-normalization mean --source-lm-prompt-mode continuation --local-files-only --run-date 2026-05-01",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./venv_arm64/bin/python scripts/build_source_private_hellaswag_hidden_summary_repair_probe.py --output-dir results/source_private_hellaswag_hidden_summary_repair_probe_20260501_qwen05_train512_validation1024 --train-hidden-rows 512 --source-lm-model /Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 --source-lm-device auto_cpu --source-lm-dtype float32 --source-lm-max-length 256 --source-lm-normalization mean --source-lm-prompt-mode continuation --local-files-only --bootstrap-samples 500 --run-date 2026-05-01",
    "./venv_arm64/bin/python scripts/build_source_private_hellaswag_top2_contrastive_repair_probe.py --output-dir results/source_private_hellaswag_top2_contrastive_repair_probe_20260501_qwen05_train512_validation1024 --run-date 2026-05-01",
    "./venv_arm64/bin/python scripts/build_source_private_hellaswag_hidden_innovation_repair_probe.py --output-dir results/source_private_hellaswag_hidden_innovation_repair_probe_20260501_qwen05_train512_validation1024 --run-date 2026-05-01",
    "./venv_arm64/bin/python scripts/build_source_private_hellaswag_hidden_innovation_stability_gate.py --output-dir results/source_private_hellaswag_hidden_innovation_stability_gate_20260501_qwen05_train512_validation1024 --run-date 2026-05-01",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./venv_arm64/bin/python scripts/build_source_private_hellaswag_hidden_innovation_train_sample_stress.py --output-dir results/source_private_hellaswag_hidden_innovation_train_sample_stress_20260501_qwen05_train512_validation1024 --train-sample-seeds 1729,2027 --split-seeds 1729,1731,1733 --bootstrap-samples 500 --run-date 2026-05-01",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./venv_arm64/bin/python scripts/build_source_private_hellaswag_hidden_innovation_bagged_gate.py --output-dir results/source_private_hellaswag_hidden_innovation_bagged_gate_third_sample_20260501_qwen05_train512_validation1024 --train-sample-cache-dir results/source_private_hellaswag_hidden_innovation_train_sample_stress_20260501_qwen05_train512_validation1024 --train-sample-seeds 1729,2027,2039 --split-seeds 1729,1731,1733 --bootstrap-samples 500 --run-date 2026-05-01",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./venv_arm64/bin/python scripts/build_source_private_hellaswag_hidden_innovation_eval_slice_stress.py --output-dir results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation1024_2048 --eval-slice-start 1024 --eval-slice-rows 1024 --train-sample-cache-dir results/source_private_hellaswag_hidden_innovation_train_sample_stress_20260501_qwen05_train512_validation1024 --train-sample-seeds 1729,2027,2039 --split-seeds 1729,1731,1733 --bootstrap-samples 500 --run-date 2026-05-01",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./venv_arm64/bin/python scripts/build_source_private_hellaswag_hidden_innovation_eval_slice_stress.py --output-dir results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation2048_3072 --eval-slice-start 2048 --eval-slice-rows 1024 --train-sample-cache-dir results/source_private_hellaswag_hidden_innovation_train_sample_stress_20260501_qwen05_train512_validation1024 --train-sample-seeds 1729,2027,2039 --split-seeds 1729,1731,1733 --bootstrap-samples 500 --run-date 2026-05-01",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./venv_arm64/bin/python scripts/build_source_private_hellaswag_hidden_innovation_eval_slice_stress.py --output-dir results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation3072_4096 --eval-slice-start 3072 --eval-slice-rows 1024 --train-sample-cache-dir results/source_private_hellaswag_hidden_innovation_train_sample_stress_20260501_qwen05_train512_validation1024 --train-sample-seeds 1729,2027,2039 --split-seeds 1729,1731,1733 --bootstrap-samples 500 --run-date 2026-05-01",
    "HF_HOME=.debug/hf_home HF_DATASETS_CACHE=.debug/hf_datasets TRANSFORMERS_CACHE=.debug/hf_transformers OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./venv_arm64/bin/python scripts/build_source_private_hellaswag_hidden_innovation_eval_slice_stress.py --output-dir results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation4096_5120 --eval-slice-start 4096 --eval-slice-rows 1024 --train-sample-cache-dir results/source_private_hellaswag_hidden_innovation_train_sample_stress_20260501_qwen05_train512_validation1024 --train-sample-seeds 1729,2027,2039 --split-seeds 1729,1731,1733 --bootstrap-samples 500 --run-date 2026-05-01",
    "./venv_arm64/bin/python scripts/build_source_private_hellaswag_hidden_innovation_eval_slice_stress.py --output-dir results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation5120_6144 --eval-slice-start 5120 --eval-slice-rows 1024 --bootstrap-samples 300 --run-date 2026-05-01",
    "./venv_arm64/bin/python scripts/build_source_private_hellaswag_hidden_innovation_eval_slice_stress.py --output-dir results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation6144_7168 --eval-slice-start 6144 --eval-slice-rows 1024 --bootstrap-samples 300 --run-date 2026-05-01",
    "./venv_arm64/bin/python scripts/build_source_private_hellaswag_hidden_innovation_eval_slice_stress.py --output-dir results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation7168_8192 --eval-slice-start 7168 --eval-slice-rows 1024 --bootstrap-samples 300 --run-date 2026-05-01",
    "./venv_arm64/bin/python scripts/build_source_private_hellaswag_hidden_innovation_eval_slice_stress.py --output-dir results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation8192_9216 --eval-slice-start 8192 --eval-slice-rows 1024 --bootstrap-samples 300 --run-date 2026-05-01",
    "./venv_arm64/bin/python scripts/build_source_private_hellaswag_hidden_innovation_eval_slice_stress.py --output-dir results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation9216_10042 --eval-slice-start 9216 --eval-slice-rows 826 --bootstrap-samples 300 --run-date 2026-05-01",
    "./venv_arm64/bin/python scripts/build_source_private_hellaswag_hidden_innovation_multi_slice_stress.py --output-dir results/source_private_hellaswag_hidden_innovation_multi_slice_stress_20260501_qwen05_validation0_9216 --slice-artifacts results/source_private_hellaswag_hidden_innovation_bagged_gate_third_sample_20260501_qwen05_train512_validation1024/hellaswag_hidden_innovation_bagged_gate.json,results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation1024_2048/hellaswag_hidden_innovation_eval_slice_stress.json,results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation2048_3072/hellaswag_hidden_innovation_eval_slice_stress.json,results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation3072_4096/hellaswag_hidden_innovation_eval_slice_stress.json,results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation4096_5120/hellaswag_hidden_innovation_eval_slice_stress.json,results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation5120_6144/hellaswag_hidden_innovation_eval_slice_stress.json,results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation6144_7168/hellaswag_hidden_innovation_eval_slice_stress.json,results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation7168_8192/hellaswag_hidden_innovation_eval_slice_stress.json,results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation8192_9216/hellaswag_hidden_innovation_eval_slice_stress.json --run-date 2026-05-01",
    "./venv_arm64/bin/python scripts/build_source_private_hellaswag_hidden_innovation_multi_slice_stress.py --output-dir results/source_private_hellaswag_hidden_innovation_multi_slice_stress_20260501_qwen05_validation0_10042 --slice-artifacts results/source_private_hellaswag_hidden_innovation_bagged_gate_third_sample_20260501_qwen05_train512_validation1024/hellaswag_hidden_innovation_bagged_gate.json,results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation1024_2048/hellaswag_hidden_innovation_eval_slice_stress.json,results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation2048_3072/hellaswag_hidden_innovation_eval_slice_stress.json,results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation3072_4096/hellaswag_hidden_innovation_eval_slice_stress.json,results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation4096_5120/hellaswag_hidden_innovation_eval_slice_stress.json,results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation5120_6144/hellaswag_hidden_innovation_eval_slice_stress.json,results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation6144_7168/hellaswag_hidden_innovation_eval_slice_stress.json,results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation7168_8192/hellaswag_hidden_innovation_eval_slice_stress.json,results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation8192_9216/hellaswag_hidden_innovation_eval_slice_stress.json,results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation9216_10042/hellaswag_hidden_innovation_eval_slice_stress.json --run-date 2026-05-01",
    "./venv_arm64/bin/python scripts/build_source_private_hellaswag_anchor_relative_hidden_innovation_gate.py --output-dir results/source_private_hellaswag_anchor_relative_hidden_innovation_gate_20260501_qwen05_train512_validation0_1024 --eval-path results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_validation_first1024.jsonl --eval-score-cache results/source_private_hellaswag_score_packet_headroom_20260501_qwen05_validation1024/source_score_cache.json --eval-hidden-cache results/source_private_hellaswag_hidden_summary_repair_probe_20260501_qwen05_train512_validation1024/source_eval_hidden_cache.npz --train-sample-cache-dir results/source_private_hellaswag_hidden_innovation_train_sample_stress_20260501_qwen05_train512_validation1024 --train-sample-seeds 1729,2027,2039 --split-seeds 1729,1731,1733 --anchor-count 128 --bootstrap-samples 500 --run-date 2026-05-01",
    "./venv_arm64/bin/python scripts/build_source_private_hellaswag_anchor_relative_hidden_innovation_gate.py --output-dir results/source_private_hellaswag_anchor_relative_hidden_innovation_gate_20260501_qwen05_train512_validation1024_2048 --eval-path results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation1024_2048/hellaswag_validation_rows_1024_2048.jsonl --eval-score-cache results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation1024_2048/source_eval_score_cache.json --eval-hidden-cache results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation1024_2048/source_eval_hidden_cache.npz --train-sample-cache-dir results/source_private_hellaswag_hidden_innovation_train_sample_stress_20260501_qwen05_train512_validation1024 --train-sample-seeds 1729,2027,2039 --split-seeds 1729,1731,1733 --anchor-count 128 --bootstrap-samples 500 --run-date 2026-05-01",
    "./venv_arm64/bin/python scripts/build_source_private_hellaswag_anchor_relative_hidden_innovation_gate.py --output-dir results/source_private_hellaswag_anchor_relative_hidden_innovation_gate_20260501_qwen05_train512_validation2048_3072 --eval-path results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation2048_3072/hellaswag_validation_rows_2048_3072.jsonl --eval-score-cache results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation2048_3072/source_eval_score_cache.json --eval-hidden-cache results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation2048_3072/source_eval_hidden_cache.npz --train-sample-cache-dir results/source_private_hellaswag_hidden_innovation_train_sample_stress_20260501_qwen05_train512_validation1024 --train-sample-seeds 1729,2027,2039 --split-seeds 1729,1731,1733 --anchor-count 128 --bootstrap-samples 500 --run-date 2026-05-01",
    "./venv_arm64/bin/python scripts/build_source_private_hellaswag_anchor_relative_hidden_innovation_gate.py --output-dir results/source_private_hellaswag_anchor_relative_hidden_innovation_gate_20260501_qwen05_train512_validation3072_4096 --eval-path results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation3072_4096/hellaswag_validation_rows_3072_4096.jsonl --eval-score-cache results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation3072_4096/source_eval_score_cache.json --eval-hidden-cache results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation3072_4096/source_eval_hidden_cache.npz --train-sample-cache-dir results/source_private_hellaswag_hidden_innovation_train_sample_stress_20260501_qwen05_train512_validation1024 --train-sample-seeds 1729,2027,2039 --split-seeds 1729,1731,1733 --anchor-count 128 --bootstrap-samples 500 --run-date 2026-05-01",
    "./venv_arm64/bin/python scripts/build_source_private_hellaswag_anchor_relative_hidden_innovation_gate.py --output-dir results/source_private_hellaswag_anchor_relative_hidden_innovation_gate_20260501_qwen05_train512_validation4096_5120 --eval-path results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation4096_5120/hellaswag_validation_rows_4096_5120.jsonl --eval-score-cache results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation4096_5120/source_eval_score_cache.json --eval-hidden-cache results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation4096_5120/source_eval_hidden_cache.npz --train-sample-cache-dir results/source_private_hellaswag_hidden_innovation_train_sample_stress_20260501_qwen05_train512_validation1024 --train-sample-seeds 1729,2027,2039 --split-seeds 1729,1731,1733 --anchor-count 128 --bootstrap-samples 500 --run-date 2026-05-01",
    "./venv_arm64/bin/python scripts/build_source_private_hellaswag_anchor_relative_hidden_innovation_multi_slice_stress.py --output-dir results/source_private_hellaswag_anchor_relative_hidden_innovation_multi_slice_stress_20260501_qwen05_validation0_5120 --run-date 2026-05-01",
    "./venv_arm64/bin/python scripts/build_source_private_hellaswag_repair_systems_acceptance_card.py --output-dir results/source_private_hellaswag_repair_systems_acceptance_card_20260501 --run-date 2026-05-01",
    "./venv_arm64/bin/python scripts/build_source_private_hellaswag_pq_hidden_innovation_codec_gate.py --output-dir results/source_private_hellaswag_pq_hidden_innovation_codec_gate_20260502_tinyllama_validation1024_2048",
    "./venv_arm64/bin/python scripts/build_source_private_mac_packet_ring_transport_microbench.py --output-dir results/source_private_mac_packet_ring_transport_microbench_20260501 --target-bytes 33554432 --repeats 5 --min-iterations 128",
    "./venv_arm64/bin/python scripts/build_source_private_serving_slo_envelope.py --output-dir results/source_private_serving_slo_envelope_20260501",
    "./venv_arm64/bin/python scripts/build_source_private_systems_rate_assumption_frontier.py --output-dir results/source_private_systems_rate_assumption_frontier_20260501",
    "./venv_arm64/bin/python scripts/build_source_private_cross_benchmark_systems_comparator.py --output-dir results/source_private_cross_benchmark_systems_comparator_20260501",
    "./venv_arm64/bin/python scripts/build_source_private_native_readiness_ledger.py --output-dir results/source_private_native_readiness_ledger_20260501",
    "./venv_arm64/bin/python scripts/build_source_private_native_systems_benchmark_plan.py --output-dir results/source_private_native_systems_benchmark_plan_20260501 --run-date 2026-05-01",
    "./venv_arm64/bin/python scripts/build_source_private_rate_frontier.py --output-dir results/source_private_rate_frontier_20260429",
    "./venv_arm64/bin/python scripts/build_source_private_kv_cache_baseline_table.py --output-dir results/source_private_kv_cache_baseline_table_20260429",
    "./venv_arm64/bin/python scripts/run_source_private_coded_label_risk_gate.py --examples 160 --candidates 4 --family-set all --seeds 29,31,37 --budget 2 --output-dir results/source_private_coded_label_risk_gate_20260429",
    "./venv_arm64/bin/python scripts/build_source_private_pass_fail_ledger.py --output-dir results/source_private_pass_fail_ledger_20260429",
    "find final -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 shasum -a 256 > final/MANIFEST.sha256",
    "shasum -a 256 -c final/MANIFEST.sha256",
    "./venv_arm64/bin/python -m pytest tests/test_build_source_private_rate_frontier.py tests/test_build_source_private_kv_cache_baseline_table.py tests/test_run_source_private_coded_label_risk_gate.py tests/test_build_source_private_pass_fail_ledger.py -q",
]


REQUIRED_ARTIFACTS = {
    "rate_frontier": "results/source_private_rate_frontier_20260429/rate_frontier.json",
    "kv_cache_baseline": "results/source_private_kv_cache_baseline_table_20260429/kv_cache_baseline_table.json",
    "coded_label_risk": "results/source_private_coded_label_risk_gate_20260429/summary.json",
    "pass_fail_ledger": "results/source_private_pass_fail_ledger_20260429/pass_fail_ledger.json",
    "endpoint_uncertainty_core": "results/source_private_endpoint_uncertainty_20260429/core_label_strict_n160/summary.json",
    "endpoint_uncertainty_holdout": "results/source_private_endpoint_uncertainty_20260429/label_strict_n160/summary.json",
    "systems_summary": "results/source_private_systems_summary_20260428/systems_summary.json",
    "final_table_doc": "paper/source_private_tool_trace_final_table_20260429.md",
    "readiness_doc": "paper/repo_readiness_review_20260426.md",
    "final_manifest": "final/MANIFEST.sha256",
    "candidate_local_competitor_basis": "results/source_private_candidate_local_competitor_basis_table_20260430/candidate_local_competitor_basis_table.json",
    "candidate_local_cross_family": "results/source_private_candidate_local_cross_family_gate_20260430/candidate_local_cross_family_gate.json",
    "candidate_local_systems_boundary": "results/source_private_candidate_local_systems_boundary_trace_20260430/candidate_local_systems_boundary_trace.json",
    "candidate_local_threshold_frontier": "results/source_private_candidate_local_threshold_frontier_20260501/candidate_local_threshold_frontier.json",
    "candidate_local_margin_atlas": "results/source_private_candidate_local_margin_atlas_20260501/margin_atlas.json",
    "candidate_conditioned_packet_builder_seed47": "results/source_private_candidate_conditioned_packet_builder_smoke_20260501/candidate_conditioned_packet_builder_smoke.json",
    "candidate_conditioned_packet_builder_seed53": "results/source_private_candidate_conditioned_packet_builder_smoke_20260501_seed53/candidate_conditioned_packet_builder_smoke.json",
    "candidate_conditioned_packet_builder_seed59": "results/source_private_candidate_conditioned_packet_builder_smoke_20260501_seed59/candidate_conditioned_packet_builder_smoke.json",
    "source_prioritized_packet_builder_loo_seed47": "results/source_private_candidate_conditioned_packet_builder_loo_hybrid_w075_20260501/candidate_conditioned_packet_builder_smoke.json",
    "source_prioritized_packet_builder_loo_seed53": "results/source_private_candidate_conditioned_packet_builder_loo_hybrid_w075_20260501_seed53/candidate_conditioned_packet_builder_smoke.json",
    "source_prioritized_packet_builder_loo_seed59": "results/source_private_candidate_conditioned_packet_builder_loo_hybrid_w075_20260501_seed59_serial/candidate_conditioned_packet_builder_smoke.json",
    "train_sender_packet_builder_seed47": "results/source_private_candidate_conditioned_packet_builder_train_builder_hybrid_w075_20260501/candidate_conditioned_packet_builder_smoke.json",
    "train_sender_packet_builder_seed53": "results/source_private_candidate_conditioned_packet_builder_train_builder_hybrid_w075_20260501_seed53/candidate_conditioned_packet_builder_smoke.json",
    "train_sender_packet_builder_seed59": "results/source_private_candidate_conditioned_packet_builder_train_builder_hybrid_w075_20260501_seed59/candidate_conditioned_packet_builder_smoke.json",
    "train_sender_packet_builder_rate": "results/source_private_candidate_conditioned_packet_builder_train_builder_hybrid_w075_rate_20260501/candidate_conditioned_packet_builder_smoke.json",
    "train_receiver_permuted_null_gap_seed47": "results/source_private_train_only_receiver_permuted_null_gap_20260501_seed47_n512/learned_synonym_dictionary_packet_gate.json",
    "train_receiver_permuted_null_gap_seed53": "results/source_private_train_only_receiver_permuted_null_gap_20260501_seed53_n512/learned_synonym_dictionary_packet_gate.json",
    "train_receiver_permuted_null_gap_seed59": "results/source_private_train_only_receiver_permuted_null_gap_20260501_seed59_n512/learned_synonym_dictionary_packet_gate.json",
    "train_donor_antishuffle_seed47_n128": "results/source_private_train_donor_antishuffle_seed47_n128_budget14/candidate_conditioned_packet_builder_smoke.json",
    "train_donor_antishuffle_seed53_n128": "results/source_private_train_donor_antishuffle_seed53_n128_budget12/candidate_conditioned_packet_builder_smoke.json",
    "train_donor_antishuffle_seed59_n128": "results/source_private_train_donor_antishuffle_seed59_n128_budget12/candidate_conditioned_packet_builder_smoke.json",
    "train_donor_antishuffle_seed47_n512": "results/source_private_train_donor_antishuffle_seed47_n512_budget14/candidate_conditioned_packet_builder_smoke.json",
    "train_donor_antishuffle_seed47_n512_budget12": "results/source_private_train_donor_antishuffle_seed47_n512_budget12_cross/candidate_conditioned_packet_builder_smoke.json",
    "train_donor_antishuffle_seed53_n512": "results/source_private_train_donor_antishuffle_seed53_n512_budget12_14/candidate_conditioned_packet_builder_smoke.json",
    "train_donor_antishuffle_seed59_n512": "results/source_private_train_donor_antishuffle_seed59_n512_budget12_14/candidate_conditioned_packet_builder_smoke.json",
    "train_donor_locked_rate_frontier": "results/source_private_train_donor_antishuffle_locked_rate_frontier_20260501/train_donor_locked_rate_frontier.json",
    "train_donor_stable_gap_selector": "results/source_private_train_donor_antishuffle_stable_gap_seed47_53_59_20260501/train_donor_locked_rate_frontier.json",
    "train_donor_fixed12_eval_audit": "results/source_private_train_donor_antishuffle_fixed12b_eval_audit_20260501/fixed_budget_eval_audit.json",
    "arc_challenge_bridge_contract": "results/source_private_arc_challenge_bridge_contract_20260501/arc_challenge_bridge_contract.json",
    "arc_challenge_fixed_packet_validation": "results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_bge_validation/arc_challenge_fixed_packet_gate.json",
    "arc_challenge_fixed_packet_test": "results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_bge_test/arc_challenge_fixed_packet_gate.json",
    "arc_challenge_seed_stability_validation": "results/source_private_arc_challenge_seed_stability_20260501_qwen05_bge_validation/arc_challenge_seed_stability.json",
    "arc_challenge_seed_stability_test": "results/source_private_arc_challenge_seed_stability_20260501_qwen05_bge_test/arc_challenge_seed_stability.json",
    "arc_challenge_common_basis_validation": "results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_validation/arc_challenge_fixed_packet_gate.json",
    "arc_challenge_common_basis_test": "results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_test/arc_challenge_fixed_packet_gate.json",
    "arc_challenge_common_basis_seed_validation": "results/source_private_arc_challenge_seed_stability_20260501_qwen05_hashed_validation/arc_challenge_seed_stability.json",
    "arc_challenge_common_basis_seed_test": "results/source_private_arc_challenge_seed_stability_20260501_qwen05_hashed_test/arc_challenge_seed_stability.json",
    "arc_challenge_anchor_relative_seed_validation": "results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_relative_validation/arc_challenge_seed_stability.json",
    "arc_challenge_anchor_relative_seed_test": "results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_relative_test/arc_challenge_seed_stability.json",
    "arc_challenge_anchor_id_shuffle_validation": "results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_id_shuffle_validation/arc_challenge_seed_stability.json",
    "arc_challenge_anchor_id_shuffle_test": "results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_id_shuffle_test/arc_challenge_seed_stability.json",
    "arc_challenge_anchor_value_shuffle_validation": "results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_value_shuffle_validation/arc_challenge_seed_stability.json",
    "arc_challenge_anchor_value_shuffle_test": "results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_value_shuffle_test/arc_challenge_seed_stability.json",
    "arc_challenge_random_anchors_validation": "results/source_private_arc_challenge_seed_stability_20260501_qwen05_random_anchors_validation/arc_challenge_seed_stability.json",
    "arc_challenge_random_anchors_test": "results/source_private_arc_challenge_seed_stability_20260501_qwen05_random_anchors_test/arc_challenge_seed_stability.json",
    "arc_challenge_source_latent_endpoint_validation": "results/source_private_arc_challenge_source_latent_endpoint_gate_20260501_qwen05_bge_validation/source_latent_endpoint_gate.json",
    "arc_challenge_systems_trace": "results/source_private_arc_challenge_systems_trace_20260501/arc_challenge_systems_trace.json",
    "sciq_bridge_contract": "results/source_private_sciq_bridge_contract_20260501/sciq_bridge_contract.json",
    "sciq_fixed_packet_validation": "results/source_private_sciq_fixed_packet_gate_20260501_qwen05_hashed_validation/arc_challenge_fixed_packet_gate.json",
    "openbookqa_bridge_contract": "results/source_private_openbookqa_bridge_contract_20260501/openbookqa_bridge_contract.json",
    "openbookqa_fixed_packet_validation": "results/source_private_openbookqa_fixed_packet_gate_20260501_qwen05_hashed_validation/arc_challenge_fixed_packet_gate.json",
    "openbookqa_fixed_packet_test_4b": "results/source_private_openbookqa_fixed_packet_gate_20260501_qwen05_hashed_test_4b/arc_challenge_fixed_packet_gate.json",
    "openbookqa_seed_stability_validation_3b": "results/source_private_openbookqa_seed_stability_20260501_qwen05_hashed_validation_3b/arc_challenge_seed_stability.json",
    "openbookqa_seed_stability_test_3b": "results/source_private_openbookqa_seed_stability_20260501_qwen05_hashed_test_3b/arc_challenge_seed_stability.json",
    "openbookqa_receiver_headroom": "results/source_private_openbookqa_receiver_headroom_gate_20260502/openbookqa_receiver_headroom_gate.json",
    "commonsenseqa_bridge_contract": "results/source_private_commonsenseqa_bridge_contract_20260501/commonsenseqa_bridge_contract.json",
    "commonsenseqa_fixed_packet_validation_12b": "results/source_private_commonsenseqa_fixed_packet_gate_20260501_qwen05_hashed_validation_12b/arc_challenge_fixed_packet_gate.json",
    "commonsenseqa_seed_validation_2b_strict": "results/source_private_commonsenseqa_seed_stability_20260501_qwen05_hashed_validation_2b/arc_challenge_seed_stability.json",
    "commonsenseqa_seed_validation_2b_gap001": "results/source_private_commonsenseqa_seed_stability_20260501_qwen05_hashed_validation_2b_gap001/arc_challenge_seed_stability.json",
    "hellaswag_bridge_contract": "results/source_private_hellaswag_bridge_contract_20260501/hellaswag_bridge_contract.json",
    "hellaswag_fixed_packet_validation1024_2b": "results/source_private_hellaswag_fixed_packet_gate_20260501_qwen05_hashed_validation1024_2b/arc_challenge_fixed_packet_gate.json",
    "hellaswag_seed_validation1024_2b_5seed": "results/source_private_hellaswag_seed_stability_20260501_qwen05_hashed_validation1024_2b_5seed/arc_challenge_seed_stability.json",
    "hellaswag_control_suite": "results/source_private_hellaswag_control_suite_20260501/hellaswag_control_suite.json",
    "hellaswag_score_packet_headroom": "results/source_private_hellaswag_score_packet_headroom_20260501_qwen05_validation1024/hellaswag_score_packet_headroom.json",
    "hellaswag_public_receiver_repair": "results/source_private_hellaswag_public_receiver_repair_probe_20260501_qwen05_validation1024/hellaswag_public_receiver_repair_probe.json",
    "hellaswag_train_source_score_repair": "results/source_private_hellaswag_train_source_score_repair_probe_20260501_qwen05_train512_validation1024/hellaswag_train_source_score_repair_probe.json",
    "hellaswag_hidden_summary_repair": "results/source_private_hellaswag_hidden_summary_repair_probe_20260501_qwen05_train512_validation1024/hellaswag_hidden_summary_repair_probe.json",
    "hellaswag_top2_contrastive_repair": "results/source_private_hellaswag_top2_contrastive_repair_probe_20260501_qwen05_train512_validation1024/hellaswag_top2_contrastive_repair_probe.json",
    "hellaswag_hidden_innovation_repair": "results/source_private_hellaswag_hidden_innovation_repair_probe_20260501_qwen05_train512_validation1024/hellaswag_hidden_innovation_repair_probe.json",
    "hellaswag_hidden_innovation_stability": "results/source_private_hellaswag_hidden_innovation_stability_gate_20260501_qwen05_train512_validation1024/hellaswag_hidden_innovation_stability_gate.json",
    "hellaswag_hidden_innovation_train_sample_stress": "results/source_private_hellaswag_hidden_innovation_train_sample_stress_20260501_qwen05_train512_validation1024/hellaswag_hidden_innovation_train_sample_stress.json",
    "hellaswag_hidden_innovation_bagged_gate": "results/source_private_hellaswag_hidden_innovation_bagged_gate_third_sample_20260501_qwen05_train512_validation1024/hellaswag_hidden_innovation_bagged_gate.json",
    "hellaswag_hidden_innovation_eval_slice_stress": "results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation1024_2048/hellaswag_hidden_innovation_eval_slice_stress.json",
    "hellaswag_hidden_innovation_eval_slice_stress_2048_3072": "results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation2048_3072/hellaswag_hidden_innovation_eval_slice_stress.json",
    "hellaswag_hidden_innovation_eval_slice_stress_3072_4096": "results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation3072_4096/hellaswag_hidden_innovation_eval_slice_stress.json",
    "hellaswag_hidden_innovation_eval_slice_stress_4096_5120": "results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation4096_5120/hellaswag_hidden_innovation_eval_slice_stress.json",
    "hellaswag_hidden_innovation_multi_slice_stress": "results/source_private_hellaswag_hidden_innovation_multi_slice_stress_20260501_qwen05_validation0_9216/hellaswag_hidden_innovation_multi_slice_stress.json",
    "hellaswag_hidden_innovation_terminal_tail": "results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation9216_10042/hellaswag_hidden_innovation_eval_slice_stress.json",
    "hellaswag_hidden_innovation_full_validation_multi_slice": "results/source_private_hellaswag_hidden_innovation_multi_slice_stress_20260501_qwen05_validation0_10042/hellaswag_hidden_innovation_multi_slice_stress.json",
    "hellaswag_anchor_relative_hidden_innovation_multi_slice": "results/source_private_hellaswag_anchor_relative_hidden_innovation_multi_slice_stress_20260501_qwen05_validation0_5120/hellaswag_anchor_relative_hidden_innovation_multi_slice_stress.json",
    "hellaswag_repair_systems_acceptance": "results/source_private_hellaswag_repair_systems_acceptance_card_20260501/hellaswag_repair_systems_acceptance_card.json",
    "hellaswag_pq_hidden_innovation_codec": "results/source_private_hellaswag_pq_hidden_innovation_codec_gate_20260502_tinyllama_validation1024_2048/hellaswag_pq_hidden_innovation_codec_gate.json",
    "mac_packet_ring_transport": "results/source_private_mac_packet_ring_transport_microbench_20260501/packet_ring_transport_microbench.json",
    "serving_slo_envelope": "results/source_private_serving_slo_envelope_20260501/serving_slo_envelope.json",
    "systems_rate_assumption_frontier": "results/source_private_systems_rate_assumption_frontier_20260501/systems_rate_assumption_frontier.json",
    "cross_benchmark_systems_comparator": "results/source_private_cross_benchmark_systems_comparator_20260501/cross_benchmark_systems_comparator.json",
    "native_readiness_ledger": "results/source_private_native_readiness_ledger_20260501/native_readiness_ledger.json",
    "native_systems_benchmark_plan": "results/source_private_native_systems_benchmark_plan_20260501/native_systems_benchmark_plan.json",
}


NOVELTY_MATRIX = [
    {
        "comparison": "LatentWire source-private packet",
        "source": "this work",
        "communicated_object": "rate-capped private evidence packet decoded with target candidate side information",
        "source_private": True,
        "requires_model_internals": False,
        "extreme_byte_rate": True,
        "source_destroying_controls": True,
        "systems_axis": "bytes, local latency, candidate accuracy, controls",
        "paper_role": "headline method",
    },
    {
        "comparison": "C2C cache-to-cache communication",
        "source": "https://arxiv.org/abs/2510.03215",
        "communicated_object": "projected/fused source KV cache",
        "source_private": "partly",
        "requires_model_internals": True,
        "extreme_byte_rate": False,
        "source_destroying_controls": "not same threat model",
        "systems_axis": "cache transfer accuracy and latency",
        "paper_role": "closest high-rate internal-state baseline/framing",
    },
    {
        "comparison": "KVComm selective KV sharing",
        "source": "https://openreview.net/forum?id=F7rUng23nw",
        "communicated_object": "selected KV pairs/layers",
        "source_private": "partly",
        "requires_model_internals": True,
        "extreme_byte_rate": False,
        "source_destroying_controls": "not same threat model",
        "systems_axis": "fraction of KV cache transmitted",
        "paper_role": "high-rate KV communication baseline/framing",
    },
    {
        "comparison": "TurboQuant / vector-KV quantization",
        "source": "https://arxiv.org/abs/2504.19874",
        "communicated_object": "quantized vectors or KV/cache states",
        "source_private": False,
        "requires_model_internals": True,
        "extreme_byte_rate": False,
        "source_destroying_controls": False,
        "systems_axis": "bits per vector/cache element",
        "paper_role": "systems byte-floor comparator and future vector-packet ablation",
    },
    {
        "comparison": "QJL 1-bit sign sketch",
        "source": "https://arxiv.org/abs/2406.03482",
        "communicated_object": "JL-projected sign sketches for inner products/KV",
        "source_private": False,
        "requires_model_internals": True,
        "extreme_byte_rate": "low-bit but high-dimensional",
        "source_destroying_controls": False,
        "systems_axis": "1-bit cache/vector compression",
        "paper_role": "matched-byte vector sketch baseline if latent branch is promoted",
    },
    {
        "comparison": "Prefix / prompt tuning",
        "source": "https://arxiv.org/abs/2101.00190; https://arxiv.org/abs/2104.08691; https://arxiv.org/abs/2110.07602",
        "communicated_object": "learned hard/soft prompt or prefix state inserted into the target context/model",
        "source_private": False,
        "requires_model_internals": "soft/prefix variants require learned continuous state or prefix KV-like state",
        "extreme_byte_rate": False,
        "source_destroying_controls": False,
        "systems_axis": "extra prompt/prefix tokens or vectors, target context/KV growth, tuning storage",
        "paper_role": "conditioning contrast; the source-private packet is a per-example discrete boundary record, not a token, soft prompt, or prefix cache",
    },
    {
        "comparison": "Prompt/text compression such as LLMLingua-family methods",
        "source": "https://arxiv.org/abs/2310.05736",
        "communicated_object": "compressed visible prompt/context tokens",
        "source_private": False,
        "requires_model_internals": False,
        "extreme_byte_rate": "token-level",
        "source_destroying_controls": False,
        "systems_axis": "prompt tokens, quality, latency",
        "paper_role": "structured text/compression framing; query-aware text relay is the local control",
    },
    {
        "comparison": "Slepian-Wolf / Wyner-Ziv source coding",
        "source": "https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources",
        "communicated_object": "syndrome/source code with decoder side information",
        "source_private": True,
        "requires_model_internals": False,
        "extreme_byte_rate": True,
        "source_destroying_controls": "theory, not benchmark controls",
        "systems_axis": "rate-distortion/coding limit",
        "paper_role": "theory framing, not empirical LLM baseline",
    },
    {
        "comparison": "JEPA / diffusion-transformer latent prediction",
        "source": "https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf",
        "communicated_object": "predicted latent/representation state",
        "source_private": "not primary",
        "requires_model_internals": True,
        "extreme_byte_rate": False,
        "source_destroying_controls": False,
        "systems_axis": "latent prediction quality",
        "paper_role": "inspiration for future learned receiver; not current claim",
    },
]


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_status() -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for name, relative in REQUIRED_ARTIFACTS.items():
        path = ROOT / relative
        status[name] = {
            "path": relative,
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else None,
            "sha256": _sha256_file(path) if path.exists() and path.is_file() else None,
        }
    return status


def _packet_builder_headline(packet_builder_runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for run in packet_builder_runs for row in run["rows"]]
    return {
        "seed_count": len(packet_builder_runs),
        "row_count": len(rows),
        "pass_rows": sum(1 for row in rows if row["pass_gate"]),
        "all_seed_cross_family_pass": all(run["pass_gate"] for run in packet_builder_runs),
        "min_candidate_accuracy": min(row["candidate_conditioned_packet_accuracy"] for row in rows),
        "max_candidate_accuracy": max(row["candidate_conditioned_packet_accuracy"] for row in rows),
        "min_base_accuracy": min(row["base_matched_accuracy"] for row in rows),
        "max_base_accuracy": max(row["base_matched_accuracy"] for row in rows),
        "min_candidate_minus_base": min(row["candidate_minus_base"] for row in rows),
        "max_best_control_accuracy": max(row["best_control_accuracy"] for row in rows),
        "min_passing_ci95_low_vs_base": min(row["paired_ci95_low_vs_base"] for row in rows if row["pass_gate"]),
        "max_target_accuracy": max(row["target_accuracy"] for row in rows),
    }


def _packet_builder_cross_family_headline(packet_builder_runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        row
        for run in packet_builder_runs
        for row in run["rows"]
        if row["direction"] in {"core_to_holdout", "holdout_to_core"}
    ]
    passing = [row for row in rows if row["pass_gate"]]
    cross_family_pass_runs = [run for run in packet_builder_runs if run["headline"]["cross_family_pass"]]
    return {
        "seed_count": len(packet_builder_runs),
        "cross_family_pass_seed_count": len(cross_family_pass_runs),
        "row_count": len(rows),
        "pass_rows": len(passing),
        "all_seed_cross_family_pass": all(run["headline"]["cross_family_pass"] for run in packet_builder_runs),
        "min_candidate_accuracy": min(row["candidate_conditioned_packet_accuracy"] for row in rows),
        "max_candidate_accuracy": max(row["candidate_conditioned_packet_accuracy"] for row in rows),
        "min_base_accuracy": min(row["base_matched_accuracy"] for row in rows),
        "max_base_accuracy": max(row["base_matched_accuracy"] for row in rows),
        "min_candidate_minus_base": min(row["candidate_minus_base"] for row in rows),
        "max_best_control_accuracy": max(row["best_control_accuracy"] for row in rows),
        "max_passing_best_control_accuracy": max(row["best_control_accuracy"] for row in passing),
        "max_target_accuracy": max(row["target_accuracy"] for row in rows),
        "min_passing_ci95_low_vs_base": min(row["paired_ci95_low_vs_base"] for row in passing),
    }


def _train_receiver_headline(receiver_runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for run in receiver_runs for row in run["rows"]]
    cross_rows = [row for row in rows if row["direction"] in {"core_to_holdout", "holdout_to_core"}]
    passing_cross = [row for row in cross_rows if row["pass_gate"]]
    return {
        "seed_count": len(receiver_runs),
        "row_count": len(rows),
        "pass_rows": sum(1 for row in rows if row["pass_gate"]),
        "cross_row_count": len(cross_rows),
        "cross_pass_rows": len(passing_cross),
        "all_seed_cross_family_pass": all(run["headline"]["cross_family_pass"] for run in receiver_runs),
        "min_cross_accuracy": min(row["learned_synonym_dictionary_accuracy"] for row in cross_rows),
        "max_cross_accuracy": max(row["learned_synonym_dictionary_accuracy"] for row in cross_rows),
        "max_cross_best_control_accuracy": max(row["best_control_accuracy"] for row in cross_rows),
        "max_cross_target_accuracy": max(row["target_accuracy"] for row in cross_rows),
        "min_cross_minus_best_control": min(row["learned_minus_best_control"] for row in cross_rows),
        "min_passing_cross_ci95_low_vs_target": min(row["paired_ci95_low_vs_target"] for row in passing_cross),
        "same_family_pass_rows": sum(1 for row in rows if row["direction"] == "same_family_all" and row["pass_gate"]),
    }


def _contribution_rows(
    *,
    rate: dict[str, Any],
    kv: dict[str, Any],
    coded: dict[str, Any],
    ledger: dict[str, Any],
    endpoint_core: dict[str, Any],
    endpoint_holdout: dict[str, Any],
    candidate_competitor: dict[str, Any],
    candidate_cross_family: dict[str, Any],
    candidate_systems: dict[str, Any],
    candidate_threshold: dict[str, Any],
    candidate_margin: dict[str, Any],
    train_sender_packet_builder_headline: dict[str, Any],
    train_sender_rate_headline: dict[str, Any],
    public_packet_builder_headline: dict[str, Any],
    loo_packet_builder_headline: dict[str, Any],
    train_receiver_headline: dict[str, Any],
    train_donor_antishuffle_headline: dict[str, Any],
    train_donor_antishuffle_n512_headline: dict[str, Any],
    train_donor_locked_frontier: dict[str, Any],
    train_donor_stable_gap: dict[str, Any],
    train_donor_fixed12_eval: dict[str, Any],
    arc_contract: dict[str, Any],
    arc_fixed_validation: dict[str, Any],
    arc_fixed_test: dict[str, Any],
    arc_seed_validation: dict[str, Any],
    arc_seed_test: dict[str, Any],
    arc_common_validation: dict[str, Any],
    arc_common_test: dict[str, Any],
    arc_common_seed_validation: dict[str, Any],
    arc_common_seed_test: dict[str, Any],
    arc_anchor_relative_seed_validation: dict[str, Any],
    arc_anchor_relative_seed_test: dict[str, Any],
    arc_anchor_id_shuffle_validation: dict[str, Any],
    arc_anchor_id_shuffle_test: dict[str, Any],
    arc_anchor_value_shuffle_validation: dict[str, Any],
    arc_anchor_value_shuffle_test: dict[str, Any],
    arc_random_anchors_validation: dict[str, Any],
    arc_random_anchors_test: dict[str, Any],
    arc_source_latent_validation: dict[str, Any],
    arc_systems_trace: dict[str, Any],
    sciq_contract: dict[str, Any],
    sciq_fixed_validation: dict[str, Any],
    openbookqa_contract: dict[str, Any],
    openbookqa_fixed_validation: dict[str, Any],
    openbookqa_fixed_test_4b: dict[str, Any],
    openbookqa_seed_validation_3b: dict[str, Any],
    openbookqa_seed_test_3b: dict[str, Any],
    openbookqa_receiver_headroom: dict[str, Any],
    commonsenseqa_contract: dict[str, Any],
    commonsenseqa_fixed_validation_12b: dict[str, Any],
    commonsenseqa_seed_validation_2b_strict: dict[str, Any],
    commonsenseqa_seed_validation_2b_gap001: dict[str, Any],
    hellaswag_contract: dict[str, Any],
    hellaswag_fixed_validation1024_2b: dict[str, Any],
    hellaswag_seed_validation1024_2b_5seed: dict[str, Any],
    hellaswag_control_suite: dict[str, Any],
    hellaswag_score_packet_headroom: dict[str, Any],
    hellaswag_public_receiver_repair: dict[str, Any],
    hellaswag_train_source_score_repair: dict[str, Any],
    hellaswag_hidden_summary_repair: dict[str, Any],
    hellaswag_hidden_innovation_repair: dict[str, Any],
    hellaswag_hidden_innovation_stability: dict[str, Any],
    hellaswag_hidden_innovation_train_sample_stress: dict[str, Any],
    hellaswag_hidden_innovation_bagged_gate: dict[str, Any],
    hellaswag_hidden_innovation_eval_slice_stress: dict[str, Any],
    hellaswag_hidden_innovation_eval_slice_stress_2048_3072: dict[str, Any],
    hellaswag_hidden_innovation_multi_slice_stress: dict[str, Any],
    hellaswag_hidden_innovation_terminal_tail: dict[str, Any],
    hellaswag_hidden_innovation_full_validation_multi_slice: dict[str, Any],
    hellaswag_anchor_relative_hidden_innovation_multi_slice: dict[str, Any],
    hellaswag_repair_systems_acceptance: dict[str, Any],
    hellaswag_pq_hidden_innovation_codec: dict[str, Any],
    mac_packet_ring: dict[str, Any],
    cross_benchmark_systems: dict[str, Any],
    native_readiness: dict[str, Any],
    native_systems_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "contribution": "OpenBookQA 3B shared-basis second public benchmark",
            "status": "new second public-benchmark positive gate and stronger rate point",
            "headline_evidence": (
                f"validation seed stability="
                f"{openbookqa_seed_validation_3b['aggregate']['pass_count']}/"
                f"{openbookqa_seed_validation_3b['aggregate']['seed_count']} with matched/target/text mean="
                f"{openbookqa_seed_validation_3b['aggregate']['matched_accuracy_mean']:.3f}/"
                f"{openbookqa_seed_validation_3b['aggregate']['target_accuracy']:.3f}/"
                f"{openbookqa_seed_validation_3b['aggregate']['same_byte_structured_text_accuracy']:.3f}; "
                f"test seed stability="
                f"{openbookqa_seed_test_3b['aggregate']['pass_count']}/"
                f"{openbookqa_seed_test_3b['aggregate']['seed_count']} with matched/target/text mean="
                f"{openbookqa_seed_test_3b['aggregate']['matched_accuracy_mean']:.3f}/"
                f"{openbookqa_seed_test_3b['aggregate']['target_accuracy']:.3f}/"
                f"{openbookqa_seed_test_3b['aggregate']['same_byte_structured_text_accuracy']:.3f}"
            ),
            "main_metric": (
                f"{openbookqa_seed_test_3b['budget_bytes']}B payload; "
                f"test min lift over target="
                f"{openbookqa_seed_test_3b['aggregate']['matched_minus_target_min']:.3f}; "
                f"min lift over same-byte text="
                f"{openbookqa_seed_test_3b['aggregate']['matched_minus_same_byte_text_min']:.3f}; "
                f"min lift over best destructive="
                f"{openbookqa_seed_test_3b['aggregate']['matched_minus_best_destructive_min']:.3f}; "
                f"min CI95 low vs target="
                f"{openbookqa_seed_test_3b['aggregate']['paired_ci95_low_vs_target_min']:.3f}"
            ),
            "remaining_gap": (
                "This promotes OpenBookQA as a second public benchmark beyond ARC and shows the public-basis "
                "packet still beats target-only, same-byte text, and destructive controls at 3B. The source "
                "decision is still a Qwen choice-loglikelihood cache, so the next ICLR gate is a more native "
                "latent/source endpoint and GPU systems measurement."
            ),
        },
        {
            "contribution": "OpenBookQA train-only packet/target receiver",
            "status": "new positive receiver-fusion method gate",
            "headline_evidence": (
                f"default seed receiver={openbookqa_receiver_headroom['headline']['default_seed_matched']['receiver_accuracy']:.3f} "
                f"vs packet={openbookqa_receiver_headroom['headline']['default_seed_matched']['base_accuracy']:.3f} "
                f"and target-public="
                f"{openbookqa_receiver_headroom['headline']['default_seed_matched']['target_public_accuracy']:.3f}; "
                f"aggregate seed-row delta="
                f"{openbookqa_receiver_headroom['headline']['aggregate_seed_row_ci_vs_packet']['mean']:.3f} "
                f"with CI95 low="
                f"{openbookqa_receiver_headroom['headline']['aggregate_seed_row_ci_vs_packet']['ci95_low']:.4f}"
            ),
            "main_metric": (
                f"{openbookqa_receiver_headroom['budget_bytes']}B source packet; "
                f"default receiver-packet delta="
                f"{openbookqa_receiver_headroom['headline']['default_seed_matched']['receiver_minus_base']:.3f}; "
                f"default receiver-target delta="
                f"{openbookqa_receiver_headroom['headline']['default_seed_matched']['receiver_minus_target_public']:.3f}; "
                f"best receiver control="
                f"{openbookqa_receiver_headroom['headline']['default_best_receiver_control']} at "
                f"{openbookqa_receiver_headroom['headline']['default_best_receiver_control_accuracy']:.3f}; "
                f"strict per-seed CI pass count="
                f"{openbookqa_receiver_headroom['headline']['strict_per_seed_ci_pass_count']}/"
                f"{openbookqa_receiver_headroom['headline']['seed_count']}"
            ),
            "remaining_gap": (
                "This is the first held-out positive receiver row after benchmark selection: validation selects "
                "a packet/target fusion rule and test improves over packet-only. The safe framing is "
                "source-private evidence fusion. For ICLR, we still need stronger seed/control stability, "
                "an ARC replication, and a less label-copy-like common-basis or learned connector."
            ),
        },
        {
            "contribution": "SciQ text-saturation diagnostic",
            "status": "documented benchmark limitation, not a promoted headline benchmark",
            "headline_evidence": (
                f"official SciQ splits materialized "
                f"{sciq_contract['official_summaries']['train']['n']}/"
                f"{sciq_contract['official_summaries']['validation']['n']}/"
                f"{sciq_contract['official_summaries']['test']['n']} train/validation/test; "
                f"validation matched/target/text="
                f"{sciq_fixed_validation['headline']['matched_accuracy']:.3f}/"
                f"{sciq_fixed_validation['headline']['target_accuracy']:.3f}/"
                f"{sciq_fixed_validation['headline']['same_byte_structured_text_accuracy']:.3f}"
            ),
            "main_metric": (
                f"matched-source minus same-byte text="
                f"{sciq_fixed_validation['headline']['matched_minus_same_byte_text']:.3f}; "
                f"matched-source minus target="
                f"{sciq_fixed_validation['headline']['matched_minus_target']:.3f}; "
                f"CI95 low vs target="
                f"{sciq_fixed_validation['headline']['paired_ci95_vs_target']['ci95_low']:.3f}"
            ),
            "remaining_gap": (
                "SciQ confirms the source signal is strong, but short answer strings let the matched-byte text "
                "control nearly catch up, so using it as a headline benchmark would weaken the fairness story. "
                "Keep it as a reviewer-facing limitation and benchmark-selection rationale."
            ),
        },
        {
            "contribution": "CommonsenseQA non-science validation probe",
            "status": "live non-science diagnostic, not a strict headline benchmark yet",
            "headline_evidence": (
                f"labeled train/validation materialized as "
                f"{commonsenseqa_contract['labeled_summaries']['train']['n']}/"
                f"{commonsenseqa_contract['labeled_summaries']['validation']['n']} with no overlap; "
                f"12B validation matched/target/text="
                f"{commonsenseqa_fixed_validation_12b['headline']['matched_accuracy']:.3f}/"
                f"{commonsenseqa_fixed_validation_12b['headline']['target_accuracy']:.3f}/"
                f"{commonsenseqa_fixed_validation_12b['headline']['same_byte_structured_text_accuracy']:.3f}; "
                f"2B relaxed-margin seed stability="
                f"{commonsenseqa_seed_validation_2b_gap001['aggregate']['pass_count']}/"
                f"{commonsenseqa_seed_validation_2b_gap001['aggregate']['seed_count']}"
            ),
            "main_metric": (
                f"2B min lift over target="
                f"{commonsenseqa_seed_validation_2b_gap001['aggregate']['matched_minus_target_min']:.3f}; "
                f"min lift over same-byte text="
                f"{commonsenseqa_seed_validation_2b_gap001['aggregate']['matched_minus_same_byte_text_min']:.3f}; "
                f"strict 0.02 text-margin pass count="
                f"{commonsenseqa_seed_validation_2b_strict['aggregate']['pass_count']}/"
                f"{commonsenseqa_seed_validation_2b_strict['aggregate']['seed_count']}"
            ),
            "remaining_gap": (
                "CommonsenseQA confirms the source signal is not science-only, but same-byte text is too close "
                "to the packet under the strict 0.02 margin and Hugging Face test labels are unavailable. "
                "Use this as the next live method-improvement target, not as an ICLR headline row yet."
            ),
        },
        {
            "contribution": "HellaSwag 2B non-science adversarial continuation gate",
            "status": "strong non-science slice, weakened by top-label-copy control",
            "headline_evidence": (
                f"labeled train/validation materialized as "
                f"{hellaswag_contract['labeled_summaries']['train']['n']}/"
                f"{hellaswag_contract['labeled_summaries']['validation']['n']} with no overlap; "
                f"1024-row fixed run matched/target/text="
                f"{hellaswag_fixed_validation1024_2b['headline']['matched_accuracy']:.3f}/"
                f"{hellaswag_fixed_validation1024_2b['headline']['target_accuracy']:.3f}/"
                f"{hellaswag_fixed_validation1024_2b['headline']['same_byte_structured_text_accuracy']:.3f}; "
                f"5-seed stability="
                f"{hellaswag_seed_validation1024_2b_5seed['aggregate']['pass_count']}/"
                f"{hellaswag_seed_validation1024_2b_5seed['aggregate']['seed_count']}; "
                f"source-label copy="
                f"{hellaswag_control_suite['headline']['source_label_text_copy_accuracy']:.3f}"
            ),
            "main_metric": (
                f"{hellaswag_seed_validation1024_2b_5seed['budget_bytes']}B payload; "
                f"{hellaswag_fixed_validation1024_2b['systems_trace']['record_bytes_with_header_crc']}B framed record; "
                f"min lift over target="
                f"{hellaswag_seed_validation1024_2b_5seed['aggregate']['matched_minus_target_min']:.3f}; "
                f"min lift over same-byte text="
                f"{hellaswag_seed_validation1024_2b_5seed['aggregate']['matched_minus_same_byte_text_min']:.3f}; "
                f"matched minus source-label copy="
                f"{hellaswag_control_suite['headline']['matched_minus_source_label_text_copy']:.3f}; "
                f"min lift over best destructive="
                f"{hellaswag_seed_validation1024_2b_5seed['aggregate']['matched_minus_best_destructive_min']:.3f}; "
                f"min CI95 low vs target="
                f"{hellaswag_seed_validation1024_2b_5seed['aggregate']['paired_ci95_low_vs_target_min']:.3f}"
            ),
            "remaining_gap": (
                "The packet clears target-only, destructive controls, and same-byte choice-prefix text, but it does "
                "not beat the stronger one-byte top-label-copy control. Treat HellaSwag as a method-improvement "
                "surface until a packet carries more than the source top choice and then widen to full validation."
            ),
        },
        {
            "contribution": "HellaSwag label-copy and score-packet headroom diagnostic",
            "status": "new reviewer-risk diagnostic / branch weakened",
            "headline_evidence": (
                f"metadata controls clean="
                f"{hellaswag_control_suite['headline']['metadata_controls_clean']}; "
                f"matched/source-label-copy="
                f"{hellaswag_control_suite['headline']['matched_accuracy']:.3f}/"
                f"{hellaswag_control_suite['headline']['source_label_text_copy_accuracy']:.3f}; "
                f"best rank-bin heldout/source-label heldout="
                f"{hellaswag_score_packet_headroom['headline']['best_rank_bin_packet_heldout_accuracy']:.3f}/"
                f"{hellaswag_score_packet_headroom['headline']['source_label_text_heldout_accuracy']:.3f}"
            ),
            "main_metric": (
                f"matched minus best metadata/activity control="
                f"{hellaswag_control_suite['headline']['matched_minus_best_metadata_or_activity_control']:.3f}; "
                f"matched minus source-label copy="
                f"{hellaswag_control_suite['headline']['matched_minus_source_label_text_copy']:.3f}; "
                f"top-2 oracle heldout="
                f"{hellaswag_score_packet_headroom['headline']['top2_oracle_heldout_accuracy']:.3f}; "
                f"score packet delta="
                f"{hellaswag_score_packet_headroom['headline']['score_packet_minus_source_label_text_heldout']:.3f}"
            ),
            "remaining_gap": (
                "The source top-2 contains substantial headroom, but confidence margin alone does not recover it. "
                "Next method branch should learn or calibrate a second-choice repair signal rather than simply "
                "quantizing the top-vs-runner-up margin."
            ),
        },
        {
            "contribution": "HellaSwag train-only public receiver repair falsification",
            "status": "new negative method gate / branch pruned",
            "headline_evidence": (
                f"train-only public receiver dev accuracy="
                f"{hellaswag_public_receiver_repair['selected_dev_accuracy']:.3f}; "
                f"source-label/public-only/top2-rerank/gated="
                f"{hellaswag_public_receiver_repair['headline']['source_label_copy_accuracy']:.3f}/"
                f"{hellaswag_public_receiver_repair['headline']['public_target_only_accuracy']:.3f}/"
                f"{hellaswag_public_receiver_repair['headline']['top2_public_rerank_accuracy']:.3f}/"
                f"{hellaswag_public_receiver_repair['headline']['public_if_in_source_top2_accuracy']:.3f}"
            ),
            "main_metric": (
                f"best repair condition="
                f"{hellaswag_public_receiver_repair['headline']['best_repair_condition']}; "
                f"best repair minus source-label copy="
                f"{hellaswag_public_receiver_repair['headline']['best_repair_minus_source_label_copy']:.3f}; "
                f"source top-2 oracle="
                f"{hellaswag_public_receiver_repair['headline']['source_top2_oracle_accuracy']:.3f}; "
                f"public prediction in source top-2 rate="
                f"{hellaswag_public_receiver_repair['headline']['public_prediction_in_source_top2_rate']:.3f}"
            ),
            "remaining_gap": (
                "A public train-only receiver scorer cannot recover the HellaSwag top-2 headroom. This prunes "
                "the eval-only/public-feature repair branch and makes the next valid gate train-split source-score "
                "calibration or a true source hidden-summary repair packet."
            ),
        },
        {
            "contribution": "HellaSwag train-source-score repair falsification",
            "status": "new negative source-score method gate / branch weakened",
            "headline_evidence": (
                f"scored train rows="
                f"{hellaswag_train_source_score_repair['scored_train_rows']}; "
                f"selected policy="
                f"{hellaswag_train_source_score_repair['headline']['selected_policy']}; "
                f"dev/eval="
                f"{hellaswag_train_source_score_repair['headline']['selected_internal_dev_accuracy']:.3f}/"
                f"{hellaswag_train_source_score_repair['headline']['selected_eval_accuracy']:.3f}; "
                f"source-label/trained-label="
                f"{hellaswag_train_source_score_repair['headline']['source_label_copy_eval_accuracy']:.3f}/"
                f"{hellaswag_train_source_score_repair['headline']['trained_choice_bias_label_copy_eval_accuracy']:.3f}"
            ),
            "main_metric": (
                f"selected minus best label-copy="
                f"{hellaswag_train_source_score_repair['headline']['selected_minus_best_label_copy']:.3f}; "
                f"source top-2/top-4 oracle="
                f"{hellaswag_train_source_score_repair['headline']['source_top2_oracle_accuracy']:.3f}/"
                f"{hellaswag_train_source_score_repair['headline']['source_top4_oracle_accuracy']:.3f}; "
                f"train source scoring latency="
                f"{hellaswag_train_source_score_repair['source_model']['train']['latency_s']:.2f}s; "
                f"record bytes="
                f"{hellaswag_train_source_score_repair['packet_contract']['framed_record_bytes']}B"
            ),
            "remaining_gap": (
                "Train-source score shape does not beat source-label copy, despite large oracle headroom. This "
                "weakens score-only HellaSwag repair and makes source hidden-summary or residual-code repair "
                "the next live branch."
            ),
        },
        {
            "contribution": "HellaSwag source-hidden summary repair falsification",
            "status": "new negative hidden-summary method gate / branch weakened",
            "headline_evidence": (
                f"scored/hidden train rows="
                f"{hellaswag_hidden_summary_repair['scored_train_rows']}; "
                f"selected layer/ridge="
                f"{hellaswag_hidden_summary_repair['hidden_model_selection']['selected_layer']}/"
                f"{hellaswag_hidden_summary_repair['hidden_model_selection']['selected_ridge']}; "
                f"hidden-label/source-label eval="
                f"{hellaswag_hidden_summary_repair['headline']['hidden_label_copy_eval_accuracy']:.3f}/"
                f"{hellaswag_hidden_summary_repair['headline']['source_label_copy_eval_accuracy']:.3f}; "
                f"hidden-packet eval="
                f"{hellaswag_hidden_summary_repair['headline']['hidden_packet_eval_accuracy']:.3f}"
            ),
            "main_metric": (
                f"hidden packet minus source-label copy="
                f"{hellaswag_hidden_summary_repair['headline']['hidden_packet_minus_source_label_copy']:.3f}; "
                f"hidden packet minus same-byte text="
                f"{hellaswag_hidden_summary_repair['headline']['hidden_packet_minus_same_byte_text']:.3f}; "
                f"CI95 vs source-label="
                f"{hellaswag_hidden_summary_repair['headline']['paired_ci95_hidden_packet_vs_source_label_copy']['ci95_low']:.3f}/"
                f"{hellaswag_hidden_summary_repair['headline']['paired_ci95_hidden_packet_vs_source_label_copy']['ci95_high']:.3f}; "
                f"source top-2 oracle="
                f"{hellaswag_hidden_summary_repair['headline']['source_top2_oracle_accuracy']:.3f}; "
                f"hidden train/eval forward="
                f"{hellaswag_hidden_summary_repair['source_model']['hidden_train']['latency_s']:.1f}/"
                f"{hellaswag_hidden_summary_repair['source_model']['hidden_eval']['latency_s']:.1f}s; "
                f"record bytes="
                f"{hellaswag_hidden_summary_repair['packet_contract']['framed_record_bytes']}B"
            ),
            "remaining_gap": (
                "A last-layer train-only source-hidden ridge scorer overfits train and underperforms raw source-label "
                "copy on frozen validation, even though the packet still beats same-byte text and destructive controls. "
                "This prunes simple hidden-label/hidden-residual repair; any revived hidden branch must use a "
                "different common-basis learner, layer sweep, denoising/OT objective, or cross-model supervision."
            ),
        },
        {
            "contribution": "HellaSwag source-hidden innovation repair",
            "status": "new positive hard-surface method gate feeding anchored stability",
            "headline_evidence": (
                f"scored/hidden train rows="
                f"{hellaswag_hidden_innovation_repair['scored_train_rows']}; "
                f"selected view/ridge="
                f"{hellaswag_hidden_innovation_repair['headline']['selected_view']}/"
                f"{hellaswag_hidden_innovation_repair['headline']['selected_ridge']}; "
                f"dev/eval="
                f"{hellaswag_hidden_innovation_repair['headline']['selected_internal_dev_accuracy']:.3f}/"
                f"{hellaswag_hidden_innovation_repair['headline']['selected_eval_accuracy']:.3f}; "
                f"best label-copy="
                f"{hellaswag_hidden_innovation_repair['headline']['best_label_copy_eval_accuracy']:.3f}"
            ),
            "main_metric": (
                f"selected minus best label-copy="
                f"{hellaswag_hidden_innovation_repair['headline']['selected_minus_best_label_copy']:.3f}; "
                f"CI95 low/high="
                f"{hellaswag_hidden_innovation_repair['headline']['paired_ci95_selected_vs_best_label_copy']['ci95_low']:.3f}/"
                f"{hellaswag_hidden_innovation_repair['headline']['paired_ci95_selected_vs_best_label_copy']['ci95_high']:.3f}; "
                f"zero/wrong/roll hidden controls="
                f"{hellaswag_hidden_innovation_repair['headline']['zero_hidden_control_accuracy']:.3f}/"
                f"{hellaswag_hidden_innovation_repair['headline']['wrong_example_hidden_control_accuracy']:.3f}/"
                f"{hellaswag_hidden_innovation_repair['headline']['candidate_roll_hidden_control_accuracy']:.3f}; "
                f"record bytes="
                f"{hellaswag_hidden_innovation_repair['packet_contract']['framed_record_bytes']}B"
            ),
            "remaining_gap": (
                "This is the first HellaSwag repair row that beats source-label copy with positive paired "
                "uncertainty while preserving the fixed-byte source-private packet boundary. A separate anchored "
                "split-stability gate now tests whether the score+hidden-residual method survives multiple train/dev "
                "splits; this row remains the single-split method discovery artifact."
            ),
        },
        {
            "contribution": "HellaSwag anchored hidden-innovation split stability",
            "status": "positive cached split gate, weakened by train-sample stress",
            "headline_evidence": (
                f"anchored split seeds pass="
                f"{hellaswag_hidden_innovation_stability['headline']['pass_count']}/"
                f"{hellaswag_hidden_innovation_stability['headline']['split_seed_count']}; "
                f"eval mean/min/max="
                f"{hellaswag_hidden_innovation_stability['headline']['selected_eval_accuracy_mean']:.3f}/"
                f"{hellaswag_hidden_innovation_stability['headline']['selected_eval_accuracy_min']:.3f}/"
                f"{hellaswag_hidden_innovation_stability['headline']['selected_eval_accuracy_max']:.3f}; "
                f"unrestricted diagnostic pass="
                f"{hellaswag_hidden_innovation_stability['unrestricted_model_selection_diagnostic']['pass_count']}/"
                f"{hellaswag_hidden_innovation_stability['unrestricted_model_selection_diagnostic']['split_seed_count']}"
            ),
            "main_metric": (
                f"min delta vs best label-copy="
                f"{hellaswag_hidden_innovation_stability['headline']['delta_vs_best_label_copy_min']:.3f}; "
                f"min CI95 low="
                f"{hellaswag_hidden_innovation_stability['headline']['paired_ci95_low_vs_best_label_copy_min']:.3f}; "
                f"min zero-hidden delta="
                f"{hellaswag_hidden_innovation_stability['headline']['selected_minus_zero_hidden_control_min']:.3f}; "
                f"wrong/candidate-roll max="
                f"{hellaswag_hidden_innovation_stability['headline']['wrong_example_hidden_control_accuracy_max']:.3f}/"
                f"{hellaswag_hidden_innovation_stability['headline']['candidate_roll_hidden_control_accuracy_max']:.3f}; "
                f"record bytes="
                f"{hellaswag_hidden_innovation_stability['packet_contract']['framed_record_bytes']}B"
            ),
            "remaining_gap": (
                "This upgrades HellaSwag from a single split to a cached split-stable anchored method, but a "
                "new train-row-sample stress now shows the branch is not yet robust enough for headline promotion."
            ),
        },
        {
            "contribution": "HellaSwag hidden-innovation train-sample stress",
            "status": "new negative robustness gate / HellaSwag branch demoted from headline",
            "headline_evidence": (
                f"train sample pass map="
                f"{hellaswag_hidden_innovation_train_sample_stress['headline']['sample_pass']}; "
                f"split rows pass="
                f"{hellaswag_hidden_innovation_train_sample_stress['headline']['pass_count']}/"
                f"{hellaswag_hidden_innovation_train_sample_stress['headline']['split_rows']}; "
                f"new sample pass count="
                f"{hellaswag_hidden_innovation_train_sample_stress['headline']['new_sample_pass_count']}"
            ),
            "main_metric": (
                f"mean/min delta vs best label-copy="
                f"{hellaswag_hidden_innovation_train_sample_stress['headline']['delta_vs_best_label_copy_mean']:.3f}/"
                f"{hellaswag_hidden_innovation_train_sample_stress['headline']['delta_vs_best_label_copy_min']:.3f}; "
                f"min CI95 low="
                f"{hellaswag_hidden_innovation_train_sample_stress['headline']['paired_ci95_low_vs_best_label_copy_min']:.3f}; "
                f"min zero-hidden delta="
                f"{hellaswag_hidden_innovation_train_sample_stress['headline']['selected_minus_zero_hidden_control_min']:.3f}; "
                f"record bytes="
                f"{hellaswag_hidden_innovation_train_sample_stress['packet_contract']['framed_record_bytes']}B"
            ),
            "remaining_gap": (
                "The original train sample remains positive, but the fresh 2027 train sample fails one split row. "
                "This weakens the current dense hidden-residual denoiser and points to either stronger "
                "train-sample aggregation, a stability-selected ridge policy, or a new SAE/common-basis packet."
            ),
        },
        {
            "contribution": "HellaSwag bagged hidden-innovation packet",
            "status": "new positive third-sample robustness gate / live ICLR method candidate",
            "headline_evidence": (
                f"aggregation="
                f"{hellaswag_hidden_innovation_bagged_gate['headline']['aggregation_policy']}; "
                f"component models="
                f"{hellaswag_hidden_innovation_bagged_gate['headline']['component_model_count']}; "
                f"train samples="
                f"{hellaswag_hidden_innovation_bagged_gate['headline']['train_sample_seed_count']} "
                f"with new samples="
                f"{hellaswag_hidden_innovation_bagged_gate['headline']['new_train_sample_seed_count']}; "
                f"eval/source-label/best-label="
                f"{hellaswag_hidden_innovation_bagged_gate['headline']['selected_eval_accuracy']:.3f}/"
                f"{hellaswag_hidden_innovation_bagged_gate['headline']['source_label_copy_eval_accuracy']:.3f}/"
                f"{hellaswag_hidden_innovation_bagged_gate['headline']['best_label_copy_eval_accuracy']:.3f}; "
                f"jackknife subbags pass="
                f"{hellaswag_hidden_innovation_bagged_gate['jackknife_summary']['pass_count']}/"
                f"{hellaswag_hidden_innovation_bagged_gate['jackknife_summary']['row_count']}"
            ),
            "main_metric": (
                f"delta vs best label-copy="
                f"{hellaswag_hidden_innovation_bagged_gate['headline']['selected_minus_best_label_copy']:.3f}; "
                f"CI95 low/high="
                f"{hellaswag_hidden_innovation_bagged_gate['headline']['paired_ci95_low_vs_best_label_copy']:.3f}/"
                f"{hellaswag_hidden_innovation_bagged_gate['headline']['paired_ci95_high_vs_best_label_copy']:.3f}; "
                f"delta vs score-only bagged="
                f"{hellaswag_hidden_innovation_bagged_gate['headline']['selected_minus_score_only_bagged_control']:.3f}; "
                f"zero/wrong/roll controls="
                f"{hellaswag_hidden_innovation_bagged_gate['headline']['zero_hidden_control_accuracy']:.3f}/"
                f"{hellaswag_hidden_innovation_bagged_gate['headline']['wrong_example_hidden_control_accuracy']:.3f}/"
                f"{hellaswag_hidden_innovation_bagged_gate['headline']['candidate_roll_hidden_control_accuracy']:.3f}; "
                f"jackknife min delta/CI low="
                f"{hellaswag_hidden_innovation_bagged_gate['jackknife_summary']['selected_minus_best_label_copy_min']:.3f}/"
                f"{hellaswag_hidden_innovation_bagged_gate['jackknife_summary']['paired_ci95_low_vs_best_label_copy_min']:.3f}; "
                f"record bytes="
                f"{hellaswag_hidden_innovation_bagged_gate['packet_contract']['framed_record_bytes']}B"
            ),
            "remaining_gap": (
                "This directly addresses the fresh train-sample failure by bagging over train-row samples and "
                "split seeds without increasing packet bytes. The third train sample and 2-of-3 jackknife subbags "
                "now pass, so the next ICLR gate is a frozen larger/full-validation slice plus native systems rows."
            ),
        },
        {
            "contribution": "HellaSwag heldout-slice hidden-innovation stress",
            "status": "new positive frozen heldout-slice gate / HellaSwag headline-candidate",
            "headline_evidence": (
                f"slice="
                f"{hellaswag_hidden_innovation_eval_slice_stress['headline']['eval_slice_start']}:"
                f"{hellaswag_hidden_innovation_eval_slice_stress['headline']['eval_slice_end_exclusive']}; "
                f"eval/source-label/best-label="
                f"{hellaswag_hidden_innovation_eval_slice_stress['headline']['selected_eval_accuracy']:.3f}/"
                f"{hellaswag_hidden_innovation_eval_slice_stress['headline']['source_label_copy_eval_accuracy']:.3f}/"
                f"{hellaswag_hidden_innovation_eval_slice_stress['headline']['best_label_copy_eval_accuracy']:.3f}; "
                f"train samples="
                f"{hellaswag_hidden_innovation_eval_slice_stress['headline']['train_sample_seed_count']}; "
                f"jackknife subbags pass="
                f"{hellaswag_hidden_innovation_eval_slice_stress['headline']['jackknife_pass_count']}/"
                f"{hellaswag_hidden_innovation_eval_slice_stress['headline']['jackknife_row_count']}"
            ),
            "main_metric": (
                f"delta vs best label-copy="
                f"{hellaswag_hidden_innovation_eval_slice_stress['headline']['selected_minus_best_label_copy']:.3f}; "
                f"CI95 low/high="
                f"{hellaswag_hidden_innovation_eval_slice_stress['headline']['paired_ci95_low_vs_best_label_copy']:.3f}/"
                f"{hellaswag_hidden_innovation_eval_slice_stress['headline']['paired_ci95_high_vs_best_label_copy']:.3f}; "
                f"delta vs score-only bagged="
                f"{hellaswag_hidden_innovation_eval_slice_stress['headline']['selected_minus_score_only_bagged_control']:.3f}; "
                f"zero/wrong/roll controls="
                f"{hellaswag_hidden_innovation_eval_slice_stress['headline']['zero_hidden_control_accuracy']:.3f}/"
                f"{hellaswag_hidden_innovation_eval_slice_stress['headline']['wrong_example_hidden_control_accuracy']:.3f}/"
                f"{hellaswag_hidden_innovation_eval_slice_stress['headline']['candidate_roll_hidden_control_accuracy']:.3f}; "
                f"record bytes="
                f"{hellaswag_hidden_innovation_eval_slice_stress['headline']['framed_record_bytes']}B"
            ),
            "remaining_gap": (
                "This is the first frozen post-first1024 validation-slice pass for the bagged hidden-innovation "
                "packet and directly reduces slice-overfit risk. It upgrades HellaSwag to a headline-candidate "
                "hard benchmark, but the comfortable ICLR gate is still full validation or predeclared multi-slice "
                "stress plus native systems rows."
            ),
        },
        {
            "contribution": "HellaSwag multi-slice hidden-innovation stress",
            "status": "new positive 9-slice gate / stronger HellaSwag headline-candidate",
            "headline_evidence": (
                f"slices="
                f"{hellaswag_hidden_innovation_multi_slice_stress['headline']['pass_slice_count']}/"
                f"{hellaswag_hidden_innovation_multi_slice_stress['headline']['slice_count']} contiguous; "
                f"rows="
                f"{hellaswag_hidden_innovation_multi_slice_stress['headline']['total_eval_rows']}; "
                f"weighted selected/best-label/score-only="
                f"{hellaswag_hidden_innovation_multi_slice_stress['headline']['weighted_selected_eval_accuracy']:.3f}/"
                f"{hellaswag_hidden_innovation_multi_slice_stress['headline']['weighted_best_label_copy_eval_accuracy']:.3f}/"
                f"{hellaswag_hidden_innovation_multi_slice_stress['headline']['weighted_score_only_bagged_control_accuracy']:.3f}; "
                f"latest heldout slice="
                f"{hellaswag_hidden_innovation_multi_slice_stress['slice_rows'][-1]['eval_slice_start']}:"
                f"{hellaswag_hidden_innovation_multi_slice_stress['slice_rows'][-1]['eval_slice_end_exclusive']}"
            ),
            "main_metric": (
                f"min delta vs best label-copy="
                f"{hellaswag_hidden_innovation_multi_slice_stress['headline']['min_delta_vs_best_label_copy']:.3f}; "
                f"min CI95 low="
                f"{hellaswag_hidden_innovation_multi_slice_stress['headline']['min_ci95_low_vs_best_label_copy']:.3f}; "
                f"min delta vs score-only/zero-hidden="
                f"{hellaswag_hidden_innovation_multi_slice_stress['headline']['min_delta_vs_score_only_bagged']:.3f}/"
                f"{hellaswag_hidden_innovation_multi_slice_stress['headline']['min_delta_vs_zero_hidden']:.3f}; "
                f"wrong/roll max="
                f"{hellaswag_hidden_innovation_multi_slice_stress['headline']['max_wrong_example_hidden_control_accuracy']:.3f}/"
                f"{hellaswag_hidden_innovation_multi_slice_stress['headline']['max_candidate_roll_hidden_control_accuracy']:.3f}; "
                f"record bytes="
                f"{hellaswag_hidden_innovation_multi_slice_stress['headline']['framed_record_bytes']}B"
            ),
            "remaining_gap": (
                "This is the strongest Mac-local HellaSwag evidence so far: the same 2B raw / 5B framed "
                "source-private hidden-innovation packet clears validation rows 0:9216 under label-copy, "
                "score-only, zero-hidden, corrupted-hidden, and jackknife checks. It still does not replace "
                "the remaining validation slices, a strict cross-family falsification pair, or native NVIDIA "
                "systems rows."
            ),
        },
        {
            "contribution": "HellaSwag terminal-tail full-validation stress",
            "status": "new soft-fail full-validation diagnostic / not overclaimed",
            "headline_evidence": (
                f"terminal slice="
                f"{hellaswag_hidden_innovation_terminal_tail['headline']['eval_slice_start']}:"
                f"{hellaswag_hidden_innovation_terminal_tail['headline']['eval_slice_end_exclusive']}; "
                f"pass={hellaswag_hidden_innovation_terminal_tail['pass_gate']}; "
                f"selected/best-label/score-only="
                f"{hellaswag_hidden_innovation_terminal_tail['headline']['selected_eval_accuracy']:.3f}/"
                f"{hellaswag_hidden_innovation_terminal_tail['headline']['best_label_copy_eval_accuracy']:.3f}/"
                f"{hellaswag_hidden_innovation_terminal_tail['headline']['score_only_bagged_control_accuracy']:.3f}; "
                f"jackknife="
                f"{hellaswag_hidden_innovation_terminal_tail['headline']['jackknife_pass_count']}/"
                f"{hellaswag_hidden_innovation_terminal_tail['headline']['jackknife_row_count']}"
            ),
            "main_metric": (
                f"tail delta vs best label-copy="
                f"{hellaswag_hidden_innovation_terminal_tail['headline']['selected_minus_best_label_copy']:.3f}; "
                f"tail CI95 low="
                f"{hellaswag_hidden_innovation_terminal_tail['headline']['paired_ci95_low_vs_best_label_copy']:.3f}; "
                f"jackknife min CI95 low="
                f"{hellaswag_hidden_innovation_terminal_tail['headline']['jackknife_min_ci95_low_vs_best_label_copy']:.3f}; "
                f"full rows="
                f"{hellaswag_hidden_innovation_full_validation_multi_slice['headline']['total_eval_rows']}; "
                f"full weighted selected/best-label="
                f"{hellaswag_hidden_innovation_full_validation_multi_slice['headline']['weighted_selected_eval_accuracy']:.3f}/"
                f"{hellaswag_hidden_innovation_full_validation_multi_slice['headline']['weighted_best_label_copy_eval_accuracy']:.3f}"
            ),
            "remaining_gap": (
                "The terminal tail has a positive overall paired margin, but the strict slice gate fails because "
                "one jackknife subbag has a non-positive lower bound. This records that HellaSwag is not a "
                "full-validation strict pass; the defensible headline remains validation[0:9216] until either "
                "a predeclared tail-repeat rule or a stronger repair packet clears the short terminal slice."
            ),
        },
        {
            "contribution": "HellaSwag anchor-relative common-basis stress",
            "status": "new negative common-basis diagnostic / HellaSwag branch blocker",
            "headline_evidence": (
                f"slices="
                f"{hellaswag_anchor_relative_hidden_innovation_multi_slice['headline']['pass_slice_count']}/"
                f"{hellaswag_anchor_relative_hidden_innovation_multi_slice['headline']['slice_count']} strict; "
                f"rows="
                f"{hellaswag_anchor_relative_hidden_innovation_multi_slice['headline']['total_eval_rows']}; "
                f"weighted selected/best-label/score-only="
                f"{hellaswag_anchor_relative_hidden_innovation_multi_slice['headline']['weighted_selected_eval_accuracy']:.3f}/"
                f"{hellaswag_anchor_relative_hidden_innovation_multi_slice['headline']['weighted_best_label_copy_eval_accuracy']:.3f}/"
                f"{hellaswag_anchor_relative_hidden_innovation_multi_slice['headline']['weighted_score_only_bagged_control_accuracy']:.3f}; "
                f"dense gap="
                f"{hellaswag_anchor_relative_hidden_innovation_multi_slice['headline']['weighted_delta_vs_dense_hidden_innovation']:.3f}"
            ),
            "main_metric": (
                f"weighted lift over best label-copy="
                f"{hellaswag_anchor_relative_hidden_innovation_multi_slice['headline']['weighted_delta_vs_best_label_copy']:.3f}; "
                f"min slice lift/CI low="
                f"{hellaswag_anchor_relative_hidden_innovation_multi_slice['headline']['min_delta_vs_best_label_copy']:.3f}/"
                f"{hellaswag_anchor_relative_hidden_innovation_multi_slice['headline']['min_ci95_low_vs_best_label_copy']:.3f}; "
                f"anchor controls below label-copy="
                f"{hellaswag_anchor_relative_hidden_innovation_multi_slice['headline']['anchor_controls_below_label_copy']}; "
                f"record bytes="
                f"{hellaswag_anchor_relative_hidden_innovation_multi_slice['headline']['framed_record_bytes']}B"
            ),
            "remaining_gap": (
                "The train-only anchor-relative common-basis bottleneck preserves a small aggregate lift but fails "
                "every strict slice and trails the dense hidden-innovation packet by more than three accuracy "
                "points. This blocks any strong common-basis claim; the next method branch should test top-k/RBF/"
                "spectral anchor features or a learned shared sparse code, while keeping this negative result as "
                "a reviewer-facing anti-overclaim."
            ),
        },
        {
            "contribution": "HellaSwag PQ hidden-code branch kill",
            "status": "new negative hidden-code/codebook gate / HellaSwag receiver-improvement cut",
            "headline_evidence": (
                f"eval slice="
                f"{hellaswag_pq_hidden_innovation_codec['headline']['eval_slice_start']}:"
                f"{hellaswag_pq_hidden_innovation_codec['headline']['eval_slice_end_exclusive']}; "
                f"default/packet-only="
                f"{hellaswag_pq_hidden_innovation_codec['headline']['default_accuracy']:.3f}/"
                f"{hellaswag_pq_hidden_innovation_codec['headline']['packet_only_accuracy']:.3f}; "
                f"best scout="
                f"{hellaswag_pq_hidden_innovation_codec['headline']['best_scout_accuracy']:.3f}; "
                f"packet="
                f"{hellaswag_pq_hidden_innovation_codec['headline']['raw_payload_bytes']}B raw/"
                f"{hellaswag_pq_hidden_innovation_codec['headline']['framed_record_bytes']}B framed"
            ),
            "main_metric": (
                f"default delta vs packet-only="
                f"{hellaswag_pq_hidden_innovation_codec['headline']['default_delta_vs_packet_only']:.3f}; "
                f"default CI95 low="
                f"{hellaswag_pq_hidden_innovation_codec['headline']['default_ci95_low_vs_packet_only']:.3f}; "
                f"best scout delta/CI95 low="
                f"{hellaswag_pq_hidden_innovation_codec['headline']['best_scout_delta_vs_packet_only']:.3f}/"
                f"{hellaswag_pq_hidden_innovation_codec['headline']['best_scout_ci95_low_vs_packet_only']:.3f}; "
                f"control max delta="
                f"{hellaswag_pq_hidden_innovation_codec['headline']['control_max_delta_vs_packet_only']:.3f}"
            ),
            "remaining_gap": (
                "This is the newest Mac-local HellaSwag decision. Product-quantized TinyLlama hidden residuals "
                "are sensitive to destructive controls but do not add stable utility beyond the compact candidate "
                "packet. Cut HellaSwag receiver-improvement from the headline story; keep the benchmark for "
                "systems/headroom/negative ablation unless a true learned query/cache connector revives it."
            ),
        },
        {
            "contribution": "HellaSwag repair systems acceptance card",
            "status": "superseded local acceptance row / demoted by later HellaSwag branch-kill gates",
            "headline_evidence": (
                f"{hellaswag_repair_systems_acceptance['headline']['rows']} HellaSwag repair rows audited; "
                f"systems audit={hellaswag_repair_systems_acceptance['headline']['systems_audit_pass']}; "
                f"method gate={hellaswag_repair_systems_acceptance['headline']['method_gate_pass']}; "
                f"native queue allowed={hellaswag_repair_systems_acceptance['headline']['native_queue_allowed']}"
            ),
            "main_metric": (
                f"best repair={hellaswag_repair_systems_acceptance['headline']['best_repair_row_id']} with "
                f"delta vs source-label copy="
                f"{hellaswag_repair_systems_acceptance['headline']['best_delta_vs_source_label_copy']:.3f}; "
                f"strict required delta="
                f"{hellaswag_repair_systems_acceptance['headline']['strict_delta_required']:.2f}; "
                f"max oracle gap remaining="
                f"{hellaswag_repair_systems_acceptance['headline']['max_oracle_gap_remaining']:.3f}"
            ),
            "remaining_gap": (
                "This historical acceptance card explains why the dense hidden-innovation branch was worth "
                "stress-testing, but it is no longer the current paper headline after terminal-tail, "
                "anchor-relative common-basis, switch-observability, discrete-query, and PQ hidden-code failures. "
                "Use it as a diagnostic, not as an ICLR method claim."
            ),
        },
        {
            "contribution": "ARC-Challenge public anchor-relative coordinate ablation",
            "status": "positive common-basis robustness evidence, not a hashed-basis superiority claim",
            "headline_evidence": (
                f"validation seed stability="
                f"{arc_anchor_relative_seed_validation['aggregate']['pass_count']}/"
                f"{arc_anchor_relative_seed_validation['aggregate']['seed_count']} with matched mean/min="
                f"{arc_anchor_relative_seed_validation['aggregate']['matched_accuracy_mean']:.3f}/"
                f"{arc_anchor_relative_seed_validation['aggregate']['matched_accuracy_min']:.3f}; "
                f"test seed stability="
                f"{arc_anchor_relative_seed_test['aggregate']['pass_count']}/"
                f"{arc_anchor_relative_seed_test['aggregate']['seed_count']} with matched mean/min="
                f"{arc_anchor_relative_seed_test['aggregate']['matched_accuracy_mean']:.3f}/"
                f"{arc_anchor_relative_seed_test['aggregate']['matched_accuracy_min']:.3f}"
            ),
            "main_metric": (
                f"test min lift over target="
                f"{arc_anchor_relative_seed_test['aggregate']['matched_minus_target_min']:.3f}; "
                f"min lift over best destructive="
                f"{arc_anchor_relative_seed_test['aggregate']['matched_minus_best_destructive_min']:.3f}; "
                f"min lift over same-byte text="
                f"{arc_anchor_relative_seed_test['aggregate']['matched_minus_same_byte_text_min']:.3f}; "
                f"min CI95 low vs target="
                f"{arc_anchor_relative_seed_test['aggregate']['paired_ci95_low_vs_target_min']:.3f}"
            ),
            "remaining_gap": (
                "This strengthens the common-basis story by replacing direct hashed coordinates with public "
                "train-anchor-relative similarities while preserving the ARC pass. It does not beat the direct "
                "hashed basis by enough to claim anchor-relative superiority; anchor-identity/value-shuffle and "
                "random-anchor controls are still needed. The source decision is still a Qwen choice cache; a "
                "second public benchmark or stronger learned hidden-state endpoint remains needed for ICLR."
            ),
        },
        {
            "contribution": "ARC-Challenge anchor-coordinate falsification controls",
            "status": "new reviewer-facing control surface",
            "headline_evidence": (
                f"ID-shuffle validation/test pass counts="
                f"{arc_anchor_id_shuffle_validation['aggregate']['pass_count']}/"
                f"{arc_anchor_id_shuffle_validation['aggregate']['seed_count']} and "
                f"{arc_anchor_id_shuffle_test['aggregate']['pass_count']}/"
                f"{arc_anchor_id_shuffle_test['aggregate']['seed_count']}; "
                f"value-shuffle validation/test pass counts="
                f"{arc_anchor_value_shuffle_validation['aggregate']['pass_count']}/"
                f"{arc_anchor_value_shuffle_validation['aggregate']['seed_count']} and "
                f"{arc_anchor_value_shuffle_test['aggregate']['pass_count']}/"
                f"{arc_anchor_value_shuffle_test['aggregate']['seed_count']}; "
                f"random shared anchors validation/test="
                f"{arc_random_anchors_validation['aggregate']['pass_count']}/"
                f"{arc_random_anchors_validation['aggregate']['seed_count']} and "
                f"{arc_random_anchors_test['aggregate']['pass_count']}/"
                f"{arc_random_anchors_test['aggregate']['seed_count']}"
            ),
            "main_metric": (
                f"test matched mean ID/value/random="
                f"{arc_anchor_id_shuffle_test['aggregate']['matched_accuracy_mean']:.3f}/"
                f"{arc_anchor_value_shuffle_test['aggregate']['matched_accuracy_mean']:.3f}/"
                f"{arc_random_anchors_test['aggregate']['matched_accuracy_mean']:.3f}; "
                f"target/text="
                f"{arc_anchor_relative_seed_test['aggregate']['target_accuracy']:.3f}/"
                f"{arc_anchor_relative_seed_test['aggregate']['same_byte_structured_text_accuracy']:.3f}"
            ),
            "remaining_gap": (
                "ID/value mismatch collapse shows sender and receiver need the same coordinate chart. "
                "Random shared anchors passing shows semantic train anchors are not necessary, so the safe claim is "
                "public common-basis packet communication rather than semantic-anchor superiority."
            ),
        },
        {
            "contribution": "ARC-Challenge shared-basis source-computable endpoint",
            "status": "new strongest public benchmark endpoint",
            "headline_evidence": (
                f"validation matched/target/text="
                f"{arc_common_validation['headline']['matched_accuracy']:.3f}/"
                f"{arc_common_validation['headline']['target_accuracy']:.3f}/"
                f"{arc_common_validation['headline']['same_byte_structured_text_accuracy']:.3f}; "
                f"test matched/target/text="
                f"{arc_common_test['headline']['matched_accuracy']:.3f}/"
                f"{arc_common_test['headline']['target_accuracy']:.3f}/"
                f"{arc_common_test['headline']['same_byte_structured_text_accuracy']:.3f}; "
                f"seed stability validation/test="
                f"{arc_common_seed_validation['aggregate']['pass_count']}/"
                f"{arc_common_seed_validation['aggregate']['seed_count']} and "
                f"{arc_common_seed_test['aggregate']['pass_count']}/"
                f"{arc_common_seed_test['aggregate']['seed_count']}"
            ),
            "main_metric": (
                f"test lift over target={arc_common_test['headline']['matched_minus_target']:.3f}; "
                f"lift over best destructive={arc_common_test['headline']['matched_minus_best_destructive']:.3f}; "
                f"lift over same-byte text={arc_common_test['headline']['matched_minus_same_byte_text']:.3f}; "
                f"seed-stability min CI95 low vs target="
                f"{arc_common_seed_test['aggregate']['paired_ci95_low_vs_target_min']:.3f}"
            ),
            "remaining_gap": (
                "This removes the receiver-BGE packet-construction caveat by using an agreed public hashed basis, "
                "but the source decision is still Qwen choice log-likelihood rather than a learned hidden-state "
                "communication endpoint; native systems rows remain pending."
            ),
        },
        {
            "contribution": "ARC-Challenge shared-basis systems trace",
            "status": "new Mac-local systems trace for the public endpoint",
            "headline_evidence": (
                f"test source scoring={arc_systems_trace['headline']['test_source_scoring_ms_per_question']:.1f} "
                f"ms/question; receiver decode p50/p95="
                f"{arc_systems_trace['headline']['test_receiver_decode_p50_us']:.1f}/"
                f"{arc_systems_trace['headline']['test_receiver_decode_p95_us']:.1f} us; "
                f"peak RSS={arc_systems_trace['headline']['test_peak_rss_mib']:.1f} MiB"
            ),
            "main_metric": (
                f"12B payload / {arc_systems_trace['headline']['record_bytes']:.0f}B framed record; "
                f"single-request cacheline/DMA="
                f"{arc_systems_trace['headline']['single_request_cacheline_bytes']:.0f}/"
                f"{arc_systems_trace['headline']['single_request_dma_bytes']:.0f}B; "
                f"batch64 line/DMA per request="
                f"{arc_systems_trace['headline']['batch64_cacheline_bytes_per_request']:.1f}/"
                f"{arc_systems_trace['headline']['batch64_dma_bytes_per_request']:.1f}B"
            ),
            "remaining_gap": (
                "This closes a Mac-local phase/RSS/byte trace gap for ARC, but native NVIDIA/vLLM "
                "TTFT/TPOT/goodput/HBM rows and source-KV baselines remain pending."
            ),
        },
        {
            "contribution": "Qwen-hidden to BGE source-latent endpoint diagnostic",
            "status": "negative diagnostic / branch weakened",
            "headline_evidence": (
                f"validation matched/target/text="
                f"{arc_source_latent_validation['headline']['matched_accuracy']:.3f}/"
                f"{arc_source_latent_validation['headline']['target_accuracy']:.3f}/"
                f"{arc_source_latent_validation['headline']['same_byte_structured_text_accuracy']:.3f}; "
                f"source accuracy before packet="
                f"{arc_source_latent_validation['source_model']['source_eval_accuracy_before_packet']:.3f}"
            ),
            "main_metric": (
                f"matched-source-latent minus same-byte text="
                f"{arc_source_latent_validation['headline']['matched_minus_same_byte_text']:.3f}; "
                f"CI95 low vs target="
                f"{arc_source_latent_validation['headline']['paired_ci95_vs_target']['ci95_low']:.3f}; "
                f"train alignment cosine mean="
                f"{arc_source_latent_validation['alignment']['train_alignment_cosine_mean']:.3f}"
            ),
            "remaining_gap": (
                "Direct ridge alignment from Qwen hidden summaries to BGE residuals is not strong enough; future "
                "latent endpoints need better common-basis learning, denoising receivers, or OT/Procrustes-style "
                "regularization before promotion."
            ),
        },
        {
            "contribution": "ARC-Challenge projection-seed stability",
            "status": "new public-benchmark robustness gate",
            "headline_evidence": (
                f"validation {arc_seed_validation['aggregate']['pass_count']}/"
                f"{arc_seed_validation['aggregate']['seed_count']} seeds pass; "
                f"test {arc_seed_test['aggregate']['pass_count']}/"
                f"{arc_seed_test['aggregate']['seed_count']} seeds pass"
            ),
            "main_metric": (
                f"test mean/min matched="
                f"{arc_seed_test['aggregate']['matched_accuracy_mean']:.3f}/"
                f"{arc_seed_test['aggregate']['matched_accuracy_min']:.3f}; "
                f"min lift over target="
                f"{arc_seed_test['aggregate']['matched_minus_target_min']:.3f}; "
                f"min lift over same-byte text="
                f"{arc_seed_test['aggregate']['matched_minus_same_byte_text_min']:.3f}; "
                f"min CI95 low vs target="
                f"{arc_seed_test['aggregate']['paired_ci95_low_vs_target_min']:.3f}"
            ),
            "remaining_gap": (
                "This closes the random-projection luck objection for ARC, but it still reuses the Qwen source-choice "
                "cache and does not replace the need for a cleaner model-to-model endpoint or a second public task."
            ),
        },
        {
            "contribution": "ARC-Challenge fixed-12B public benchmark transfer",
            "status": "new positive public-benchmark gate",
            "headline_evidence": (
                f"validation matched/target/text="
                f"{arc_fixed_validation['headline']['matched_accuracy']:.3f}/"
                f"{arc_fixed_validation['headline']['target_accuracy']:.3f}/"
                f"{arc_fixed_validation['headline']['same_byte_structured_text_accuracy']:.3f}; "
                f"test matched/target/text="
                f"{arc_fixed_test['headline']['matched_accuracy']:.3f}/"
                f"{arc_fixed_test['headline']['target_accuracy']:.3f}/"
                f"{arc_fixed_test['headline']['same_byte_structured_text_accuracy']:.3f}"
            ),
            "main_metric": (
                f"test lift over target={arc_fixed_test['headline']['matched_minus_target']:.3f}; "
                f"lift over best destructive={arc_fixed_test['headline']['matched_minus_best_destructive']:.3f}; "
                f"CI95 low vs target="
                f"{arc_fixed_test['headline']['paired_ci95_vs_target']['ci95_low']:.3f}; "
                f"candidate derangement="
                f"{arc_fixed_test['condition_metrics']['candidate_derangement']['accuracy']:.3f}"
            ),
            "remaining_gap": (
                "This is the first positive public ARC row, but the source scorer is a local Qwen log-likelihood "
                "oracle over candidate text rather than the original train-donor synthetic source; ICLR still needs "
                "a cleaner cross-model source/target story and native serving measurements."
            ),
        },
        {
            "contribution": "ARC-Challenge public bridge contract",
            "status": "new public-benchmark readiness gate",
            "headline_evidence": (
                f"official ARC-Challenge splits materialized "
                f"{arc_contract['official_summaries']['train']['n']}/"
                f"{arc_contract['official_summaries']['validation']['n']}/"
                f"{arc_contract['official_summaries']['test']['n']} train/validation/test; "
                f"local smoke validation/eval overlap="
                f"{arc_contract['local_overlap_matrix']['validation_smoke']['evaluation_smoke']}"
            ),
            "main_metric": (
                f"fixed {arc_contract['method_contract']['fixed_packet_budget_bytes']}B contract; "
                f"{len(arc_contract['method_contract']['required_controls'])} required controls; "
                f"public result ready={arc_contract['public_benchmark_result_ready']}"
            ),
            "remaining_gap": (
                "This freezes the public benchmark and leakage controls, but it is deliberately not a positive "
                "ARC result until the fixed-12B packet beats target-only and destructive controls."
            ),
        },
        {
            "contribution": "Global stable-gap validation selector",
            "status": "new strongest model-selection audit",
            "headline_evidence": (
                f"{train_donor_stable_gap['policies']['global']['pass_rows']}/"
                f"{train_donor_stable_gap['policies']['global']['row_count']} selected n512 rows pass after "
                f"global validation-selected budget "
                f"{train_donor_stable_gap['policies']['global']['selected_budget_by_seed']}"
            ),
            "main_metric": (
                f"selected accuracy min="
                f"{train_donor_stable_gap['policies']['global']['min_selected_candidate_accuracy']:.3f}; "
                f"max best control="
                f"{train_donor_stable_gap['policies']['global']['max_selected_best_control_accuracy']:.3f}; "
                f"min CI95 lower bound vs base="
                f"{train_donor_stable_gap['policies']['global']['min_selected_ci95_low_vs_base']:.3f}"
            ),
            "remaining_gap": (
                "This gives a global pre-eval 12B selector, but it uses source-private gap validation and reports "
                "visible structured text as a separate access-model comparator; public benchmark and native systems "
                "evidence remain pending."
            ),
        },
        {
            "contribution": "Global fixed-12B train-donor eval frontier",
            "status": "new strongest eval-surface result",
            "headline_evidence": (
                f"{train_donor_fixed12_eval['headline']['pass_rows']}/"
                f"{train_donor_fixed12_eval['headline']['rows']} n512 cross-family rows pass at one fixed 12B rate "
                f"across seeds {train_donor_fixed12_eval['headline']['seeds']}"
            ),
            "main_metric": (
                f"min accuracy={train_donor_fixed12_eval['headline']['min_candidate_accuracy']:.3f}; "
                f"max best control={train_donor_fixed12_eval['headline']['max_best_control_accuracy']:.3f}; "
                f"min CI95 lower bound vs base="
                f"{train_donor_fixed12_eval['headline']['min_paired_ci95_low_vs_base']:.3f}"
            ),
            "remaining_gap": (
                "This removes the need for per-seed eval budgets, but it is an eval-only audit; ICLR still needs "
                "a train-only validation rule or predeclared operating-point argument for choosing 12B."
            ),
        },
        {
            "contribution": "Validation-locked train-donor rate frontier",
            "status": "new reviewer-facing model-selection audit",
            "headline_evidence": (
                f"{train_donor_locked_frontier['policies']['per_seed']['pass_rows']}/"
                f"{train_donor_locked_frontier['policies']['per_seed']['row_count']} selected n512 rows pass after "
                f"validation-selected budgets {train_donor_locked_frontier['policies']['per_seed']['selected_budget_by_seed']}"
            ),
            "main_metric": (
                f"selected accuracy min="
                f"{train_donor_locked_frontier['policies']['per_seed']['min_selected_candidate_accuracy']:.3f}; "
                f"max best control="
                f"{train_donor_locked_frontier['policies']['per_seed']['max_selected_best_control_accuracy']:.3f}; "
                f"min CI95 lower bound vs base="
                f"{train_donor_locked_frontier['policies']['per_seed']['min_selected_ci95_low_vs_base']:.3f}"
            ),
            "remaining_gap": (
                "This removes the worst hand-picked-budget concern for the current seed-repeat surface, but the "
                "global fixed-budget policy still does not pass; final ICLR needs a larger train-only validation "
                "split or a method adjustment that makes one budget clear every seed."
            ),
        },
        {
            "contribution": "Unified train-only train-donor anti-shuffle packet method",
            "status": "strongest live ICLR method branch",
            "headline_evidence": (
                f"{train_donor_antishuffle_headline['pass_rows']}/"
                f"{train_donor_antishuffle_headline['row_count']} n128 cross-family seed-repeat rows pass; "
                f"{train_donor_antishuffle_n512_headline['cross_family_pass_seed_count']}/"
                f"{train_donor_antishuffle_n512_headline['seed_count']} n512 seeds pass under the 12-14B frontier"
            ),
            "main_metric": (
                f"12-14B accuracy {train_donor_antishuffle_headline['min_candidate_accuracy']:.3f}-"
                f"{train_donor_antishuffle_headline['max_candidate_accuracy']:.3f} vs base "
                f"{train_donor_antishuffle_headline['min_base_accuracy']:.3f}-"
                f"{train_donor_antishuffle_headline['max_base_accuracy']:.3f}; "
                f"n512 min CI95 lower bound vs base="
                f"{train_donor_antishuffle_n512_headline['min_passing_ci95_low_vs_base']:.3f}"
            ),
            "remaining_gap": (
                "This removes the eval-donor caveat for the unified train-only stack and clears n512 seed repeats, "
                "but still needs a locked validation-based byte-selection rule, public benchmark transfer, "
                "and native NVIDIA/vLLM systems measurements."
            ),
        },
        {
            "contribution": "Train-only receiver permuted-null gap decoder",
            "status": "new live receiver-basis method candidate",
            "headline_evidence": (
                f"{train_receiver_headline['cross_pass_rows']}/"
                f"{train_receiver_headline['cross_row_count']} n512 cross-family seed-repeat rows pass; "
                f"{train_receiver_headline['seed_count']}/3 seeds cross-family pass"
            ),
            "main_metric": (
                f"12B train-only receiver accuracy {train_receiver_headline['min_cross_accuracy']:.3f}-"
                f"{train_receiver_headline['max_cross_accuracy']:.3f} vs target "
                f"{train_receiver_headline['max_cross_target_accuracy']:.3f}; "
                f"max best control={train_receiver_headline['max_cross_best_control_accuracy']:.3f}; "
                f"min CI95 lower bound vs target="
                f"{train_receiver_headline['min_passing_cross_ci95_low_vs_target']:.3f}"
            ),
            "remaining_gap": (
                "This clears the cross-family train-only receiver gate for the source-atom packet, but same-family "
                "is still limited by a structured-text control; use it as the receiver component inside the new "
                "train-donor anti-shuffle stack."
            ),
        },
        {
            "contribution": "Train-only sender source-prioritized packet builder",
            "status": "strongest current generalization-facing method",
            "headline_evidence": (
                f"{train_sender_packet_builder_headline['pass_rows']}/"
                f"{train_sender_packet_builder_headline['row_count']} n512 seed-repeat rows pass; "
                f"{train_sender_packet_builder_headline['seed_count']}/3 seeds cross-family pass"
            ),
            "main_metric": (
                f"12B accuracy {train_sender_packet_builder_headline['min_candidate_accuracy']:.3f}-"
                f"{train_sender_packet_builder_headline['max_candidate_accuracy']:.3f} vs live base "
                f"{train_sender_packet_builder_headline['min_base_accuracy']:.3f}-"
                f"{train_sender_packet_builder_headline['max_base_accuracy']:.3f}; "
                f"min lift over base={train_sender_packet_builder_headline['min_candidate_minus_base']:.3f}; "
                f"rate pass rows={train_sender_rate_headline['pass_rows']}/"
                f"{train_sender_rate_headline['row_count']}"
            ),
            "remaining_gap": (
                "The sender packet builder is train-only, but the receiver dictionary still uses public eval-disjoint "
                "calibration; keep as a separate sender contribution now that train-donor anti-shuffle is the unified branch."
            ),
        },
        {
            "contribution": "Source-prioritized innovation packet builder",
            "status": "new strict held-out-family positive method",
            "headline_evidence": (
                f"{loo_packet_builder_headline['pass_rows']}/{loo_packet_builder_headline['row_count']} n512 seed-repeat rows pass; "
                f"{loo_packet_builder_headline['seed_count']}/3 seeds cross-family pass"
            ),
            "main_metric": (
                f"12B accuracy {loo_packet_builder_headline['min_candidate_accuracy']:.3f}-"
                f"{loo_packet_builder_headline['max_candidate_accuracy']:.3f} vs live base "
                f"{loo_packet_builder_headline['min_base_accuracy']:.3f}-"
                f"{loo_packet_builder_headline['max_base_accuracy']:.3f}; "
                f"min lift over base={loo_packet_builder_headline['min_candidate_minus_base']:.3f}"
            ),
            "remaining_gap": (
                "Packet builder excludes the eval family from builder calibration, but the candidate dictionary is still "
                "public eval-disjoint; true train-only cross-family generalization and real benchmarks remain ICLR gates."
            ),
        },
        {
            "contribution": "Public-disjoint source-to-candidate packet builder",
            "status": "high-accuracy adaptation row",
            "headline_evidence": (
                f"{public_packet_builder_headline['pass_rows']}/{public_packet_builder_headline['row_count']} n512 seed-repeat rows pass; "
                f"{public_packet_builder_headline['seed_count']}/3 seeds cross-family pass"
            ),
            "main_metric": (
                f"8B accuracy {public_packet_builder_headline['min_candidate_accuracy']:.3f}-"
                f"{public_packet_builder_headline['max_candidate_accuracy']:.3f} vs live base "
                f"{public_packet_builder_headline['min_base_accuracy']:.3f}-"
                f"{public_packet_builder_headline['max_base_accuracy']:.3f}; "
                f"min lift over base={public_packet_builder_headline['min_candidate_minus_base']:.3f}"
            ),
            "remaining_gap": (
                "This is stronger in accuracy but weaker in generalization than leave-one-family-out; use as adaptation "
                "evidence, not the headline ICLR generalization claim."
            ),
        },
        {
            "contribution": "Source-private candidate-local residual packet",
            "status": "current live positive method",
            "headline_evidence": (
                f"{candidate_competitor['headline']['live_pass_rows']}/"
                f"{candidate_competitor['headline']['live_rows']} n512 rows pass strict controls"
            ),
            "main_metric": (
                f"{candidate_systems['headline']['live_packet_payload_bytes']}B payload / "
                f"{candidate_systems['headline']['live_packet_record_bytes']}B record; "
                f"accuracy {candidate_systems['headline']['live_accuracy_min']:.3f}-"
                f"{candidate_systems['headline']['live_accuracy_max']:.3f}"
            ),
            "remaining_gap": "Still benchmark-specific; needs broader tasks and native systems rows for comfortable ICLR.",
        },
        {
            "contribution": "Same-family versus cross-family separation gate",
            "status": "reviewer-facing generalization evidence",
            "headline_evidence": (
                f"live cross-family rows pass {candidate_cross_family['headline']['live_cross_family_pass_rows']}/"
                f"{candidate_cross_family['headline']['live_cross_family_rows']} and same-family rows pass "
                f"{candidate_cross_family['headline']['live_same_family_pass_rows']}/"
                f"{candidate_cross_family['headline']['live_same_family_rows']}"
            ),
            "main_metric": (
                f"cross-family min matched={candidate_cross_family['headline']['live_cross_family_min_matched_accuracy']:.3f}; "
                f"max best control={candidate_cross_family['headline']['live_cross_family_best_control_accuracy_max']:.3f}"
            ),
            "remaining_gap": "This is strict within the current synthetic family split, not yet unrelated real LLM families.",
        },
        {
            "contribution": "Candidate-local threshold frontier",
            "status": "mechanistic robustness diagnostic",
            "headline_evidence": (
                f"live clean threshold range "
                f"{candidate_threshold['headline']['live_clean_threshold_range']['min']:.2f}-"
                f"{candidate_threshold['headline']['live_clean_threshold_range']['max']:.2f}; "
                f"RR/sign clean threshold exists="
                f"{candidate_threshold['headline']['rr_clean_threshold_range']['exists']}/"
                f"{candidate_threshold['headline']['random_rotation_sign_clean_threshold_range']['exists']}"
            ),
            "main_metric": (
                f"threshold 0.48 clean rows="
                f"{candidate_threshold['headline']['live_threshold_0_48_clean_rows']}/"
                f"{candidate_threshold['headline']['live_threshold_0_48_rows']}"
            ),
            "remaining_gap": "Explains the live row; does not create a new learned packet interface.",
        },
        {
            "contribution": "Candidate-local margin atlas",
            "status": "reviewer-facing confidence and failure-mode diagnostic",
            "headline_evidence": (
                f"live matched positive-margin rate="
                f"{candidate_margin['headline']['live_matched_positive_margin_rate']:.3f}; "
                f"best destructive control="
                f"{candidate_margin['headline']['live_best_control_positive_margin_rate']:.3f}; "
                f"Procrustes matched/control="
                f"{candidate_margin['headline']['procrustes_matched_positive_margin_rate']:.3f}/"
                f"{candidate_margin['headline']['procrustes_best_control_positive_margin_rate']:.3f}"
            ),
            "main_metric": (
                f"live matched p50 margin={candidate_margin['headline']['live_matched_margin_p50']:.3f}; "
                f"best-control condition={candidate_margin['headline']['live_best_control_condition']}"
            ),
            "remaining_gap": "Diagnostic evidence only; the next method branch should use this margin surface to train or prune residual codes.",
        },
        {
            "contribution": "Candidate-local systems boundary trace",
            "status": "strong Mac-local systems positioning",
            "headline_evidence": (
                f"no source text/KV exposure; min KV proxy/live-record ratio "
                f"{candidate_systems['headline']['min_kv_native_proxy_record_ratio_vs_live']:.1f}x"
            ),
            "main_metric": (
                f"resident sparse decode p50="
                f"{candidate_systems['headline']['live_resident_sparse_decode_p50_us']:.3f} us/request; "
                f"packet-ring batch64 p95={mac_packet_ring['headline']['packet_batch64_p95_ns_per_request']:.3f} ns/request"
            ),
            "remaining_gap": "Native NVIDIA/vLLM C2C/KVComm/TurboQuant rows remain pending.",
        },
        {
            "contribution": "Cross-benchmark source-state byte-floor systems comparator",
            "status": "new cross-benchmark systems/accounting contribution",
            "headline_evidence": (
                f"{cross_benchmark_systems['headline']['headline_eligible_benchmarks']} headline public "
                f"benchmarks and {cross_benchmark_systems['headline']['diagnostic_benchmarks']} diagnostic "
                f"benchmark; framed packets="
                f"{cross_benchmark_systems['headline']['min_framed_packet_bytes']:.0f}-"
                f"{cross_benchmark_systems['headline']['max_framed_packet_bytes']:.0f}B; "
                f"native systems complete={cross_benchmark_systems['headline']['native_systems_complete']}"
            ),
            "main_metric": (
                f"min one-token QJL 1-bit KV floor="
                f"{cross_benchmark_systems['headline']['min_qjl_1bit_ratio_vs_framed']:.1f}x framed packet; "
                f"min 30%-layer QJL floor="
                f"{cross_benchmark_systems['headline']['min_qjl_30pct_ratio_vs_framed']:.1f}x; "
                f"min 30%-layer fp16 KVComm floor="
                f"{cross_benchmark_systems['headline']['min_kvcomm30_fp16_ratio_vs_framed']:.1f}x"
            ),
            "remaining_gap": (
                "This makes the source-state exposure cost explicit across ARC, OpenBookQA, and HellaSwag while "
                "marking HellaSwag as diagnostic due to label-copy. It is still a byte-floor/accounting comparator, "
                "not a native C2C/KVComm/QJL/TurboQuant/vLLM win."
            ),
        },
        {
            "contribution": "Native-readiness systems ledger",
            "status": "new systems non-claim/readiness table",
            "headline_evidence": (
                f"{native_readiness['headline']['local_measured_rows']} Mac-local measured rows; "
                f"{native_readiness['headline']['pending_native_rows']} pending native rows"
            ),
            "main_metric": (
                f"native ready={native_readiness['headline']['native_ready']}; "
                f"source-private local rows={native_readiness['headline']['source_private_local_rows']}"
            ),
            "remaining_gap": (
                "Strengthens systems framing by defining exactly what to measure next, but it is not a native "
                "throughput result and should not be counted as closing the NVIDIA/vLLM gate."
            ),
        },
        {
            "contribution": "Native vLLM/SGLang systems benchmark plan",
            "status": "new native systems runbook and acceptance schema",
            "headline_evidence": (
                f"{native_systems_plan['headline']['required_baseline_count']} required rows across "
                f"{'/'.join(native_systems_plan['headline']['serving_substrates'])}; "
                f"headline benchmarks={','.join(native_systems_plan['headline']['headline_benchmarks'])}; "
                f"native systems complete={native_systems_plan['headline']['native_systems_complete']}"
            ),
            "main_metric": (
                f"{native_systems_plan['headline']['required_metric_count']} required metrics including "
                "TTFT, TPOT, goodput, peak GPU memory, HBM/PCIe-or-NVLink traffic, payload bytes, "
                "and source-exposure flags"
            ),
            "remaining_gap": (
                "This converts the systems blocker into a reproducible GPU measurement table, but it is still "
                "a plan artifact until NVIDIA/vLLM/SGLang rows are executed and ingested."
            ),
        },
        {
            "contribution": "Source-private evidence-packet benchmark and controls",
            "status": "strong scoped contribution",
            "headline_evidence": (
                f"{coded['examples']} examples x {len(coded['seeds'])} seeds x "
                f"{len(coded['transforms'])} label/code/order stress transforms pass"
            ),
            "main_metric": (
                f"matched={coded['by_transform']['label_code_order_composed']['min_matched_accuracy']:.3f}, "
                f"target={coded['by_transform']['label_code_order_composed']['max_target_accuracy']:.3f}, "
                f"worst_control={coded['by_transform']['label_code_order_composed']['max_best_source_destroying_control']:.3f}"
            ),
            "remaining_gap": "Still a protocol/candidate-decoder task; frame as source-private evidence communication, not universal semantics.",
        },
        {
            "contribution": "Extreme-rate candidate-syndrome packet method",
            "status": "headline method for scoped paper",
            "headline_evidence": (
                f"packet oracle bytes max={rate['headline']['packet_oracle_bytes_max']:.1f}; "
                f"matched-byte text accuracy max={rate['headline']['matched_byte_text_at_packet_accuracy_max']:.3f}"
            ),
            "main_metric": (
                f"packet vs query-aware text >= {rate['headline']['packet_vs_query_aware_oracle_compression_min']:.1f}x; "
                f"packet vs full log >= {rate['headline']['packet_vs_full_log_compression_min']:.1f}x"
            ),
            "remaining_gap": "Text becomes oracle at higher bytes, so claim only the far-left rate frontier.",
        },
        {
            "contribution": "Systems byte/KV-cache accounting frontier",
            "status": "systems contribution with clear caveat",
            "headline_evidence": (
                f"minimum QJL-style 1-bit cache payload is "
                f"{kv['headline']['min_non_packet_qjl_1bit_bytes_vs_packet']:.1f}x packet"
            ),
            "main_metric": (
                f"minimum KIVI-style 2-bit cache payload is "
                f"{kv['headline']['min_non_packet_kivi_2bit_bytes_vs_packet']:.1f}x packet"
            ),
            "remaining_gap": "Derived byte accounting only; no production GPU serving throughput yet.",
        },
        {
            "contribution": "Endpoint paired uncertainty and local target-decoder evidence",
            "status": "paper-ready evidence rows exist, but systems scope is local",
            "headline_evidence": (
                f"endpoint paired rows pass with min packet-vs-target CI lows "
                f"{endpoint_core['min_packet_vs_target_ci95_low']:.3f}/{endpoint_holdout['min_packet_vs_target_ci95_low']:.3f}"
            ),
            "main_metric": f"paper-ready rows in ledger={len(ledger['paper_ready_rows'])}; total audited rows={ledger['total_rows']}",
            "remaining_gap": "Mac-local proxy, not server TTFT/TPOT/throughput.",
        },
        {
            "contribution": "Learned receiver / latent-method diagnostics",
            "status": "bounded diagnostic contribution, not headline cross-family claim",
            "headline_evidence": (
                f"ledger records {ledger['by_contribution'].get('learned target-preserving receiver', {}).get('positive_needs_more_evidence', 0)} "
                "positive learned-receiver rows and explicit failed/pruned rows"
            ),
            "main_metric": "same-distribution positives exist; simple cross-family masked innovation failed",
            "remaining_gap": "Need shared-dictionary/crosscoder-style method with feature knockout before claiming cross-family latent communication.",
        },
    ]


def _pass_checks(
    *,
    artifacts: dict[str, dict[str, Any]],
    rate: dict[str, Any],
    kv: dict[str, Any],
    coded: dict[str, Any],
    ledger: dict[str, Any],
    endpoint_core: dict[str, Any],
    endpoint_holdout: dict[str, Any],
    candidate_competitor: dict[str, Any],
    candidate_cross_family: dict[str, Any],
    candidate_systems: dict[str, Any],
    candidate_threshold: dict[str, Any],
    candidate_margin: dict[str, Any],
    train_sender_packet_builder_headline: dict[str, Any],
    train_sender_rate_headline: dict[str, Any],
    public_packet_builder_headline: dict[str, Any],
    loo_packet_builder_headline: dict[str, Any],
    train_receiver_headline: dict[str, Any],
    train_donor_antishuffle_headline: dict[str, Any],
    train_donor_antishuffle_n512_headline: dict[str, Any],
    train_donor_locked_frontier: dict[str, Any],
    train_donor_stable_gap: dict[str, Any],
    train_donor_fixed12_eval: dict[str, Any],
    arc_contract: dict[str, Any],
    arc_fixed_validation: dict[str, Any],
    arc_fixed_test: dict[str, Any],
    arc_seed_validation: dict[str, Any],
    arc_seed_test: dict[str, Any],
    arc_common_validation: dict[str, Any],
    arc_common_test: dict[str, Any],
    arc_common_seed_validation: dict[str, Any],
    arc_common_seed_test: dict[str, Any],
    arc_anchor_relative_seed_validation: dict[str, Any],
    arc_anchor_relative_seed_test: dict[str, Any],
    arc_anchor_id_shuffle_validation: dict[str, Any],
    arc_anchor_id_shuffle_test: dict[str, Any],
    arc_anchor_value_shuffle_validation: dict[str, Any],
    arc_anchor_value_shuffle_test: dict[str, Any],
    arc_random_anchors_validation: dict[str, Any],
    arc_random_anchors_test: dict[str, Any],
    arc_source_latent_validation: dict[str, Any],
    arc_systems_trace: dict[str, Any],
    sciq_contract: dict[str, Any],
    sciq_fixed_validation: dict[str, Any],
    openbookqa_contract: dict[str, Any],
    openbookqa_fixed_validation: dict[str, Any],
    openbookqa_fixed_test_4b: dict[str, Any],
    openbookqa_seed_validation_3b: dict[str, Any],
    openbookqa_seed_test_3b: dict[str, Any],
    openbookqa_receiver_headroom: dict[str, Any],
    commonsenseqa_contract: dict[str, Any],
    commonsenseqa_fixed_validation_12b: dict[str, Any],
    commonsenseqa_seed_validation_2b_strict: dict[str, Any],
    commonsenseqa_seed_validation_2b_gap001: dict[str, Any],
    hellaswag_contract: dict[str, Any],
    hellaswag_fixed_validation1024_2b: dict[str, Any],
    hellaswag_seed_validation1024_2b_5seed: dict[str, Any],
    hellaswag_control_suite: dict[str, Any],
    hellaswag_score_packet_headroom: dict[str, Any],
    hellaswag_public_receiver_repair: dict[str, Any],
    hellaswag_train_source_score_repair: dict[str, Any],
    hellaswag_hidden_summary_repair: dict[str, Any],
    hellaswag_hidden_innovation_repair: dict[str, Any],
    hellaswag_hidden_innovation_stability: dict[str, Any],
    hellaswag_hidden_innovation_train_sample_stress: dict[str, Any],
    hellaswag_hidden_innovation_bagged_gate: dict[str, Any],
    hellaswag_hidden_innovation_eval_slice_stress: dict[str, Any],
    hellaswag_hidden_innovation_eval_slice_stress_2048_3072: dict[str, Any],
    hellaswag_hidden_innovation_multi_slice_stress: dict[str, Any],
    hellaswag_hidden_innovation_terminal_tail: dict[str, Any],
    hellaswag_hidden_innovation_full_validation_multi_slice: dict[str, Any],
    hellaswag_anchor_relative_hidden_innovation_multi_slice: dict[str, Any],
    hellaswag_repair_systems_acceptance: dict[str, Any],
    hellaswag_pq_hidden_innovation_codec: dict[str, Any],
    mac_packet_ring: dict[str, Any],
    serving_slo: dict[str, Any],
    systems_rate_assumption: dict[str, Any],
    cross_benchmark_systems: dict[str, Any],
    native_readiness: dict[str, Any],
    native_systems_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    checks = [
        ("required_artifacts_exist", all(row["exists"] for row in artifacts.values())),
        (
            "train_donor_antishuffle_3_seed_cross_family_pass",
            train_donor_antishuffle_headline["all_seed_cross_family_pass"],
        ),
        (
            "train_donor_antishuffle_6_of_6_cross_rows_pass",
            train_donor_antishuffle_headline["pass_rows"] == train_donor_antishuffle_headline["row_count"] == 6,
        ),
        (
            "train_donor_antishuffle_controls_clean",
            train_donor_antishuffle_headline["max_passing_best_control_accuracy"]
            <= train_donor_antishuffle_headline["max_target_accuracy"] + 0.03,
        ),
        (
            "train_donor_antishuffle_seed47_n512_cross_family_pass",
            train_donor_antishuffle_n512_headline["all_seed_cross_family_pass"],
        ),
        (
            "train_donor_antishuffle_3_seed_n512_cross_family_pass",
            train_donor_antishuffle_n512_headline["cross_family_pass_seed_count"] == 3,
        ),
        (
            "train_donor_antishuffle_seed47_n512_ci_positive",
            train_donor_antishuffle_n512_headline["min_passing_ci95_low_vs_base"] > 0.05,
        ),
        ("train_donor_locked_rate_frontier_passes", bool(train_donor_locked_frontier["pass_gate"])),
        (
            "train_donor_locked_rate_frontier_6_of_6_selected_eval_rows",
            train_donor_locked_frontier["policies"]["per_seed"]["pass_rows"]
            == train_donor_locked_frontier["policies"]["per_seed"]["row_count"]
            == 6,
        ),
        (
            "train_donor_locked_rate_frontier_ci_positive",
            train_donor_locked_frontier["policies"]["per_seed"]["min_selected_ci95_low_vs_base"] > 0.05,
        ),
        ("train_donor_stable_gap_selector_passes", bool(train_donor_stable_gap["pass_gate"])),
        (
            "train_donor_stable_gap_selector_selects_global_12b",
            set(train_donor_stable_gap["policies"]["global"]["selected_budget_by_seed"].values()) == {12},
        ),
        (
            "train_donor_stable_gap_selector_6_of_6_selected_eval_rows",
            train_donor_stable_gap["policies"]["global"]["pass_rows"]
            == train_donor_stable_gap["policies"]["global"]["row_count"]
            == 6,
        ),
        ("train_donor_fixed12_eval_audit_passes", bool(train_donor_fixed12_eval["pass_gate"])),
        (
            "train_donor_fixed12_eval_6_of_6_cross_rows_pass",
            train_donor_fixed12_eval["headline"]["pass_rows"]
            == train_donor_fixed12_eval["headline"]["rows"]
            == 6,
        ),
        (
            "train_donor_fixed12_eval_ci_positive",
            train_donor_fixed12_eval["headline"]["min_paired_ci95_low_vs_base"] > 0.05,
        ),
        ("arc_challenge_bridge_contract_passes", bool(arc_contract["pass_gate"])),
        (
            "arc_challenge_bridge_has_official_splits",
            bool(arc_contract["checks"]["official_splits_materialized"])
            and arc_contract["checks"]["official_counts_match_expected"] is True
            and arc_contract["checks"]["official_splits_disjoint"] is True,
        ),
        (
            "arc_challenge_public_result_not_overclaimed",
            arc_contract["public_benchmark_result_ready"] is False,
        ),
        ("arc_challenge_fixed_packet_validation_passes", bool(arc_fixed_validation["pass_gate"])),
        ("arc_challenge_fixed_packet_test_passes", bool(arc_fixed_test["pass_gate"])),
        (
            "arc_challenge_fixed_packet_test_beats_target",
            arc_fixed_test["headline"]["matched_minus_target"] >= 0.05
            and arc_fixed_test["headline"]["paired_ci95_vs_target"]["ci95_low"] > 0.0,
        ),
        (
            "arc_challenge_fixed_packet_test_beats_same_byte_text",
            arc_fixed_test["headline"]["matched_minus_same_byte_text"] > 0.0,
        ),
        (
            "arc_challenge_fixed_packet_test_derangement_collapses",
            arc_fixed_test["condition_metrics"]["candidate_derangement"]["accuracy"]
            <= arc_fixed_test["headline"]["target_accuracy"] + 0.05,
        ),
        ("arc_challenge_seed_stability_validation_passes", bool(arc_seed_validation["pass_gate"])),
        ("arc_challenge_seed_stability_test_passes", bool(arc_seed_test["pass_gate"])),
        (
            "arc_challenge_seed_stability_test_5_of_5",
            arc_seed_test["aggregate"]["pass_count"] == arc_seed_test["aggregate"]["seed_count"] == 5,
        ),
        (
            "arc_challenge_seed_stability_test_lift_stable",
            arc_seed_test["aggregate"]["matched_minus_target_min"] >= 0.05
            and arc_seed_test["aggregate"]["matched_minus_best_destructive_min"] >= 0.03,
        ),
        (
            "arc_challenge_seed_stability_test_beats_same_byte_text",
            arc_seed_test["aggregate"]["matched_minus_same_byte_text_min"] > 0.0,
        ),
        (
            "arc_challenge_seed_stability_test_ci_positive",
            arc_seed_test["aggregate"]["paired_ci95_low_vs_target_min"] > 0.0,
        ),
        ("arc_challenge_common_basis_validation_passes", bool(arc_common_validation["pass_gate"])),
        ("arc_challenge_common_basis_test_passes", bool(arc_common_test["pass_gate"])),
        (
            "arc_challenge_common_basis_test_beats_target",
            arc_common_test["headline"]["matched_minus_target"] >= 0.05
            and arc_common_test["headline"]["paired_ci95_vs_target"]["ci95_low"] > 0.0,
        ),
        (
            "arc_challenge_common_basis_test_beats_same_byte_text",
            arc_common_test["headline"]["matched_minus_same_byte_text"] > 0.0,
        ),
        ("arc_challenge_common_basis_seed_validation_passes", bool(arc_common_seed_validation["pass_gate"])),
        ("arc_challenge_common_basis_seed_test_passes", bool(arc_common_seed_test["pass_gate"])),
        (
            "arc_challenge_common_basis_seed_test_5_of_5",
            arc_common_seed_test["aggregate"]["pass_count"]
            == arc_common_seed_test["aggregate"]["seed_count"]
            == 5,
        ),
        (
            "arc_challenge_common_basis_seed_test_ci_positive",
            arc_common_seed_test["aggregate"]["paired_ci95_low_vs_target_min"] > 0.0,
        ),
        ("arc_challenge_anchor_relative_seed_validation_passes", bool(arc_anchor_relative_seed_validation["pass_gate"])),
        ("arc_challenge_anchor_relative_seed_test_passes", bool(arc_anchor_relative_seed_test["pass_gate"])),
        (
            "arc_challenge_anchor_relative_seed_test_5_of_5",
            arc_anchor_relative_seed_test["aggregate"]["pass_count"]
            == arc_anchor_relative_seed_test["aggregate"]["seed_count"]
            == 5,
        ),
        (
            "arc_challenge_anchor_relative_seed_test_beats_same_byte_text",
            arc_anchor_relative_seed_test["aggregate"]["matched_minus_same_byte_text_min"] > 0.0,
        ),
        (
            "arc_challenge_anchor_relative_seed_test_ci_positive",
            arc_anchor_relative_seed_test["aggregate"]["paired_ci95_low_vs_target_min"] > 0.0,
        ),
        (
            "arc_challenge_anchor_id_shuffle_validation_collapses",
            arc_anchor_id_shuffle_validation["aggregate"]["pass_count"] == 0
            and arc_anchor_id_shuffle_validation["aggregate"]["matched_accuracy_mean"]
            <= arc_anchor_id_shuffle_validation["aggregate"]["target_accuracy"] + 0.02,
        ),
        (
            "arc_challenge_anchor_id_shuffle_test_collapses",
            arc_anchor_id_shuffle_test["aggregate"]["pass_count"] == 0
            and arc_anchor_id_shuffle_test["aggregate"]["matched_accuracy_mean"]
            <= arc_anchor_id_shuffle_test["aggregate"]["target_accuracy"] + 0.02,
        ),
        (
            "arc_challenge_anchor_value_shuffle_validation_collapses",
            arc_anchor_value_shuffle_validation["aggregate"]["pass_count"] == 0
            and arc_anchor_value_shuffle_validation["aggregate"]["matched_accuracy_mean"]
            <= arc_anchor_value_shuffle_validation["aggregate"]["target_accuracy"] + 0.02,
        ),
        (
            "arc_challenge_anchor_value_shuffle_test_collapses",
            arc_anchor_value_shuffle_test["aggregate"]["pass_count"] == 0
            and arc_anchor_value_shuffle_test["aggregate"]["matched_accuracy_mean"]
            <= arc_anchor_value_shuffle_test["aggregate"]["target_accuracy"] + 0.02,
        ),
        (
            "arc_challenge_random_anchors_validation_passes",
            arc_random_anchors_validation["aggregate"]["pass_count"]
            == arc_random_anchors_validation["aggregate"]["seed_count"]
            == 5,
        ),
        (
            "arc_challenge_random_anchors_test_passes",
            arc_random_anchors_test["aggregate"]["pass_count"]
            == arc_random_anchors_test["aggregate"]["seed_count"]
            == 5,
        ),
        (
            "arc_challenge_random_anchors_test_beats_same_byte_text",
            arc_random_anchors_test["aggregate"]["matched_minus_same_byte_text_min"] > 0.0,
        ),
        (
            "arc_challenge_source_latent_endpoint_not_overclaimed",
            arc_source_latent_validation["pass_gate"] is False
            and arc_source_latent_validation["headline"]["matched_minus_same_byte_text"] < 0.0,
        ),
        ("arc_challenge_systems_trace_passes", bool(arc_systems_trace["pass_gate"])),
        (
            "arc_challenge_systems_trace_scopes_native_blocker",
            arc_systems_trace["headline"]["colm_systems_trace_ready"] is True
            and arc_systems_trace["headline"]["iclr_native_systems_complete"] is False
            and arc_systems_trace["headline"]["pending_native_rows"] >= 5,
        ),
        (
            "arc_challenge_systems_trace_source_private_boundary",
            arc_systems_trace["headline"]["payload_bytes"] == 12.0
            and arc_systems_trace["headline"]["record_bytes"] == 15.0,
        ),
        ("sciq_bridge_contract_passes", bool(sciq_contract["pass_gate"])),
        (
            "sciq_validation_text_saturation_documented",
            bool(sciq_fixed_validation["pass_gate"])
            and sciq_fixed_validation["headline"]["matched_minus_target"] >= 0.40
            and sciq_fixed_validation["headline"]["matched_minus_same_byte_text"] <= 0.01,
        ),
        ("openbookqa_bridge_contract_passes", bool(openbookqa_contract["pass_gate"])),
        (
            "openbookqa_bridge_has_official_splits",
            bool(openbookqa_contract["checks"]["official_splits_materialized"])
            and openbookqa_contract["checks"]["official_counts_match_expected"] is True
            and openbookqa_contract["checks"]["official_splits_disjoint"] is True,
        ),
        (
            "openbookqa_fixed_validation_source_cache_positive",
            openbookqa_fixed_validation["headline"]["matched_minus_target"] >= 0.05
            and openbookqa_fixed_validation["headline"]["matched_minus_same_byte_text"] > 0.0
            and openbookqa_fixed_validation["headline"]["paired_ci95_vs_target"]["ci95_low"] > 0.0,
        ),
        (
            "openbookqa_fixed_test_4b_source_cache_positive",
            openbookqa_fixed_test_4b["headline"]["matched_minus_target"] >= 0.05
            and openbookqa_fixed_test_4b["headline"]["matched_minus_same_byte_text"] > 0.0
            and openbookqa_fixed_test_4b["headline"]["paired_ci95_vs_target"]["ci95_low"] > 0.0,
        ),
        ("openbookqa_seed_validation_3b_passes", bool(openbookqa_seed_validation_3b["pass_gate"])),
        ("openbookqa_seed_test_3b_passes", bool(openbookqa_seed_test_3b["pass_gate"])),
        (
            "openbookqa_seed_test_3b_5_of_5",
            openbookqa_seed_test_3b["aggregate"]["pass_count"]
            == openbookqa_seed_test_3b["aggregate"]["seed_count"]
            == 5,
        ),
        (
            "openbookqa_seed_test_3b_beats_same_byte_text",
            openbookqa_seed_test_3b["aggregate"]["matched_minus_same_byte_text_min"] >= 0.02,
        ),
        (
            "openbookqa_seed_test_3b_ci_positive",
            openbookqa_seed_test_3b["aggregate"]["paired_ci95_low_vs_target_min"] > 0.0,
        ),
        ("openbookqa_receiver_headroom_candidate_passes", bool(openbookqa_receiver_headroom["receiver_candidate_pass"])),
        (
            "openbookqa_receiver_default_beats_packet_and_target",
            openbookqa_receiver_headroom["headline"]["default_seed_matched"]["receiver_minus_base"] >= 0.005
            and openbookqa_receiver_headroom["headline"]["default_seed_matched"][
                "receiver_minus_target_public"
            ]
            >= 0.005
            and openbookqa_receiver_headroom["headline"]["default_seed_matched"][
                "paired_ci95_vs_base"
            ]["ci95_low"]
            > 0.0,
        ),
        (
            "openbookqa_receiver_aggregate_seed_delta_positive",
            openbookqa_receiver_headroom["headline"]["all_seed_deltas_positive"] is True
            and openbookqa_receiver_headroom["headline"]["aggregate_seed_row_ci_vs_packet"]["ci95_low"] > 0.0,
        ),
        ("commonsenseqa_bridge_contract_passes", bool(commonsenseqa_contract["pass_gate"])),
        (
            "commonsenseqa_labeled_splits_disjoint",
            bool(commonsenseqa_contract["checks"]["labeled_splits_materialized"])
            and commonsenseqa_contract["checks"]["labeled_counts_match_expected"] is True
            and commonsenseqa_contract["checks"]["labeled_splits_disjoint"] is True,
        ),
        (
            "commonsenseqa_12b_source_signal_beats_target_and_controls",
            commonsenseqa_fixed_validation_12b["headline"]["matched_minus_target"] >= 0.20
            and commonsenseqa_fixed_validation_12b["headline"]["matched_minus_best_destructive"] >= 0.20
            and commonsenseqa_fixed_validation_12b["headline"]["paired_ci95_vs_target"]["ci95_low"] > 0.0,
        ),
        (
            "commonsenseqa_text_saturation_not_overclaimed",
            commonsenseqa_fixed_validation_12b["pass_gate"] is False
            and commonsenseqa_fixed_validation_12b["headline"]["matched_minus_same_byte_text"] <= 0.001
            and commonsenseqa_seed_validation_2b_strict["pass_gate"] is False,
        ),
        (
            "commonsenseqa_2b_relaxed_margin_seed_stable",
            commonsenseqa_seed_validation_2b_gap001["pass_gate"] is True
            and commonsenseqa_seed_validation_2b_gap001["aggregate"]["pass_count"]
            == commonsenseqa_seed_validation_2b_gap001["aggregate"]["seed_count"]
            == 5,
        ),
        ("hellaswag_bridge_contract_passes", bool(hellaswag_contract["pass_gate"])),
        (
            "hellaswag_bridge_has_labeled_splits",
            bool(hellaswag_contract["checks"]["labeled_splits_materialized"])
            and hellaswag_contract["checks"]["labeled_counts_match_expected"] is True
            and hellaswag_contract["checks"]["labeled_splits_disjoint"] is True,
        ),
        ("hellaswag_fixed_validation1024_2b_passes", bool(hellaswag_fixed_validation1024_2b["pass_gate"])),
        (
            "hellaswag_fixed_validation1024_2b_beats_same_byte_text",
            hellaswag_fixed_validation1024_2b["headline"]["matched_minus_same_byte_text"] >= 0.02,
        ),
        (
            "hellaswag_fixed_validation1024_2b_ci_positive",
            hellaswag_fixed_validation1024_2b["headline"]["paired_ci95_vs_target"]["ci95_low"] > 0.0,
        ),
        ("hellaswag_seed_validation1024_2b_passes", bool(hellaswag_seed_validation1024_2b_5seed["pass_gate"])),
        (
            "hellaswag_seed_validation1024_2b_5_of_5",
            hellaswag_seed_validation1024_2b_5seed["aggregate"]["pass_count"]
            == hellaswag_seed_validation1024_2b_5seed["aggregate"]["seed_count"]
            == 5,
        ),
        (
            "hellaswag_seed_validation1024_2b_beats_same_byte_text",
            hellaswag_seed_validation1024_2b_5seed["aggregate"]["matched_minus_same_byte_text_min"] >= 0.02,
        ),
        (
            "hellaswag_seed_validation1024_2b_controls_clean",
            hellaswag_seed_validation1024_2b_5seed["aggregate"]["matched_minus_best_destructive_min"] >= 0.03
            and hellaswag_seed_validation1024_2b_5seed["aggregate"]["candidate_derangement_accuracy_max"]
            <= hellaswag_seed_validation1024_2b_5seed["aggregate"]["target_accuracy"] + 0.05,
        ),
        (
            "hellaswag_seed_validation1024_2b_record_is_5b",
            hellaswag_fixed_validation1024_2b["systems_trace"]["record_bytes_with_header_crc"] == 5,
        ),
        ("hellaswag_control_suite_metadata_controls_clean", bool(hellaswag_control_suite["pass_gate"])),
        (
            "hellaswag_control_suite_label_copy_threat_documented",
            hellaswag_control_suite["headline"]["label_copy_threat_present"] is True
            and hellaswag_control_suite["strict_non_label_copy_pass_gate"] is False
            and hellaswag_control_suite["headline"]["matched_minus_source_label_text_copy"] < 0.02,
        ),
        (
            "hellaswag_score_packet_headroom_not_overclaimed",
            hellaswag_score_packet_headroom["pass_gate"] is False
            and hellaswag_score_packet_headroom["headline"]["best_rank_bin_packet_minus_source_label_text_heldout"]
            < 0.02,
        ),
        (
            "hellaswag_score_packet_top2_oracle_has_method_headroom",
            hellaswag_score_packet_headroom["headline"]["top2_oracle_heldout_accuracy"]
            - hellaswag_score_packet_headroom["headline"]["source_label_text_heldout_accuracy"]
            >= 0.20,
        ),
        (
            "hellaswag_public_receiver_repair_not_overclaimed",
            hellaswag_public_receiver_repair["pass_gate"] is False
            and hellaswag_public_receiver_repair["headline"]["best_repair_minus_source_label_copy"] < 0.02,
        ),
        (
            "hellaswag_public_receiver_repair_top2_headroom_persists",
            hellaswag_public_receiver_repair["headline"]["source_top2_oracle_accuracy"]
            - hellaswag_public_receiver_repair["headline"]["source_label_copy_accuracy"]
            >= 0.20,
        ),
        (
            "hellaswag_train_source_score_repair_not_overclaimed",
            hellaswag_train_source_score_repair["pass_gate"] is False
            and hellaswag_train_source_score_repair["headline"]["selected_minus_best_label_copy"] < 0.02,
        ),
        (
            "hellaswag_train_source_score_repair_top2_headroom_persists",
            hellaswag_train_source_score_repair["headline"]["source_top2_oracle_accuracy"]
            - hellaswag_train_source_score_repair["headline"]["source_label_copy_eval_accuracy"]
            >= 0.20,
        ),
        (
            "hellaswag_hidden_summary_repair_not_overclaimed",
            hellaswag_hidden_summary_repair["pass_gate"] is False
            and hellaswag_hidden_summary_repair["headline"]["hidden_packet_minus_source_label_copy"] < 0.02
            and hellaswag_hidden_summary_repair["headline"]["paired_ci95_hidden_packet_vs_source_label_copy"]["ci95_high"]
            < 0.0,
        ),
        (
            "hellaswag_hidden_summary_repair_top2_headroom_persists",
            hellaswag_hidden_summary_repair["headline"]["source_top2_oracle_accuracy"]
            - hellaswag_hidden_summary_repair["headline"]["source_label_copy_eval_accuracy"]
            >= 0.20,
        ),
        ("hellaswag_hidden_innovation_repair_passes", bool(hellaswag_hidden_innovation_repair["pass_gate"])),
        (
            "hellaswag_hidden_innovation_repair_beats_label_copy",
            hellaswag_hidden_innovation_repair["headline"]["selected_minus_best_label_copy"] >= 0.02
            and hellaswag_hidden_innovation_repair["headline"]["paired_ci95_selected_vs_best_label_copy"][
                "ci95_low"
            ]
            > 0.0,
        ),
        (
            "hellaswag_hidden_innovation_repair_controls_collapse",
            hellaswag_hidden_innovation_repair["headline"]["selected_minus_zero_hidden_control"] >= 0.02
            and hellaswag_hidden_innovation_repair["headline"]["wrong_example_hidden_control_accuracy"]
            <= hellaswag_hidden_innovation_repair["headline"]["best_label_copy_eval_accuracy"]
            and hellaswag_hidden_innovation_repair["headline"]["candidate_roll_hidden_control_accuracy"]
            <= hellaswag_hidden_innovation_repair["headline"]["best_label_copy_eval_accuracy"],
        ),
        (
            "hellaswag_hidden_innovation_repair_source_private_packet",
            hellaswag_hidden_innovation_repair["packet_contract"]["raw_payload_bytes"] == 2
            and hellaswag_hidden_innovation_repair["packet_contract"]["framed_record_bytes"] == 5
            and hellaswag_hidden_innovation_repair["packet_contract"]["source_text_exposed"] is False
            and hellaswag_hidden_innovation_repair["packet_contract"]["source_kv_exposed"] is False
            and hellaswag_hidden_innovation_repair["packet_contract"]["raw_hidden_vector_transmitted"] is False
            and hellaswag_hidden_innovation_repair["packet_contract"]["raw_scores_transmitted"] is False,
        ),
        (
            "hellaswag_hidden_innovation_stability_passes",
            bool(hellaswag_hidden_innovation_stability["pass_gate"]),
        ),
        (
            "hellaswag_hidden_innovation_stability_5_of_5_cached_splits",
            hellaswag_hidden_innovation_stability["headline"]["pass_count"] == 5
            and hellaswag_hidden_innovation_stability["headline"]["split_seed_count"] == 5
            and hellaswag_hidden_innovation_stability["headline"]["delta_vs_best_label_copy_min"] >= 0.02
            and hellaswag_hidden_innovation_stability["headline"]["paired_ci95_low_vs_best_label_copy_min"] > 0.0,
        ),
        (
            "hellaswag_hidden_innovation_stability_controls_collapse",
            hellaswag_hidden_innovation_stability["headline"]["selected_minus_zero_hidden_control_min"] >= 0.02
            and hellaswag_hidden_innovation_stability["headline"]["wrong_example_hidden_control_accuracy_max"]
            <= max(row["best_label_copy_eval_accuracy"] for row in hellaswag_hidden_innovation_stability["stability_rows"])
            and hellaswag_hidden_innovation_stability["headline"]["candidate_roll_hidden_control_accuracy_max"]
            <= max(row["best_label_copy_eval_accuracy"] for row in hellaswag_hidden_innovation_stability["stability_rows"]),
        ),
        (
            "hellaswag_hidden_innovation_stability_unrestricted_selector_not_overclaimed",
            hellaswag_hidden_innovation_stability["unrestricted_model_selection_diagnostic"]["pass_count"]
            < hellaswag_hidden_innovation_stability["unrestricted_model_selection_diagnostic"]["split_seed_count"]
            and "score_only"
            in hellaswag_hidden_innovation_stability["unrestricted_model_selection_diagnostic"]["selected_view_counts"],
        ),
        (
            "hellaswag_hidden_innovation_stability_source_private_packet",
            hellaswag_hidden_innovation_stability["packet_contract"]["raw_payload_bytes"] == 2
            and hellaswag_hidden_innovation_stability["packet_contract"]["framed_record_bytes"] == 5
            and hellaswag_hidden_innovation_stability["packet_contract"]["source_text_exposed"] is False
            and hellaswag_hidden_innovation_stability["packet_contract"]["source_kv_exposed"] is False
            and hellaswag_hidden_innovation_stability["packet_contract"]["raw_hidden_vector_transmitted"] is False
            and hellaswag_hidden_innovation_stability["packet_contract"]["raw_scores_transmitted"] is False,
        ),
        (
            "hellaswag_hidden_innovation_train_sample_stress_not_overclaimed",
            hellaswag_hidden_innovation_train_sample_stress["pass_gate"] is False
            and hellaswag_hidden_innovation_train_sample_stress["headline"]["new_sample_pass_count"] == 0
            and hellaswag_hidden_innovation_train_sample_stress["headline"]["pass_count"]
            < hellaswag_hidden_innovation_train_sample_stress["headline"]["split_rows"],
        ),
        (
            "hellaswag_hidden_innovation_train_sample_stress_records_fresh_sample",
            hellaswag_hidden_innovation_train_sample_stress["headline"]["new_train_sample_seed_count"] >= 1
            and "2027" in hellaswag_hidden_innovation_train_sample_stress["headline"]["sample_pass"],
        ),
        (
            "hellaswag_hidden_innovation_train_sample_stress_source_private_packet",
            hellaswag_hidden_innovation_train_sample_stress["packet_contract"]["raw_payload_bytes"] == 2
            and hellaswag_hidden_innovation_train_sample_stress["packet_contract"]["framed_record_bytes"] == 5
            and hellaswag_hidden_innovation_train_sample_stress["packet_contract"]["source_text_exposed"] is False
            and hellaswag_hidden_innovation_train_sample_stress["packet_contract"]["source_kv_exposed"] is False
            and hellaswag_hidden_innovation_train_sample_stress["packet_contract"][
                "raw_hidden_vector_transmitted"
            ]
            is False
            and hellaswag_hidden_innovation_train_sample_stress["packet_contract"]["raw_scores_transmitted"]
            is False,
        ),
        (
            "hellaswag_hidden_innovation_bagged_gate_passes",
            bool(hellaswag_hidden_innovation_bagged_gate["pass_gate"]),
        ),
        (
            "hellaswag_hidden_innovation_bagged_gate_beats_label_and_score_only",
            hellaswag_hidden_innovation_bagged_gate["headline"]["selected_minus_best_label_copy"] >= 0.02
            and hellaswag_hidden_innovation_bagged_gate["headline"]["paired_ci95_low_vs_best_label_copy"] > 0.0
            and hellaswag_hidden_innovation_bagged_gate["headline"][
                "selected_minus_score_only_bagged_control"
            ]
            >= 0.02
            and hellaswag_hidden_innovation_bagged_gate["headline"]["paired_ci95_low_vs_score_only_bagged"]
            > 0.0,
        ),
        (
            "hellaswag_hidden_innovation_bagged_gate_controls_collapse",
            hellaswag_hidden_innovation_bagged_gate["headline"]["selected_minus_zero_hidden_control"] >= 0.02
            and hellaswag_hidden_innovation_bagged_gate["headline"]["wrong_example_hidden_control_accuracy"]
            <= hellaswag_hidden_innovation_bagged_gate["headline"]["best_label_copy_eval_accuracy"]
            and hellaswag_hidden_innovation_bagged_gate["headline"]["candidate_roll_hidden_control_accuracy"]
            <= hellaswag_hidden_innovation_bagged_gate["headline"]["best_label_copy_eval_accuracy"],
        ),
        (
            "hellaswag_hidden_innovation_bagged_gate_uses_fresh_train_sample",
            hellaswag_hidden_innovation_bagged_gate["headline"]["new_train_sample_seed_count"] >= 2
            and hellaswag_hidden_innovation_bagged_gate["headline"]["component_model_count"] >= 9,
        ),
        (
            "hellaswag_hidden_innovation_bagged_gate_jackknife_3_of_3",
            hellaswag_hidden_innovation_bagged_gate["jackknife_summary"]["row_count"] == 3
            and hellaswag_hidden_innovation_bagged_gate["jackknife_summary"]["pass_count"] == 3
            and hellaswag_hidden_innovation_bagged_gate["jackknife_summary"][
                "selected_minus_best_label_copy_min"
            ]
            >= 0.02
            and hellaswag_hidden_innovation_bagged_gate["jackknife_summary"][
                "paired_ci95_low_vs_best_label_copy_min"
            ]
            > 0.0
            and hellaswag_hidden_innovation_bagged_gate["jackknife_summary"][
                "selected_minus_score_only_bagged_control_min"
            ]
            >= 0.02
            and hellaswag_hidden_innovation_bagged_gate["jackknife_summary"][
                "paired_ci95_low_vs_score_only_bagged_min"
            ]
            > 0.0,
        ),
        (
            "hellaswag_hidden_innovation_bagged_gate_source_private_packet",
            hellaswag_hidden_innovation_bagged_gate["packet_contract"]["raw_payload_bytes"] == 2
            and hellaswag_hidden_innovation_bagged_gate["packet_contract"]["framed_record_bytes"] == 5
            and hellaswag_hidden_innovation_bagged_gate["packet_contract"]["source_text_exposed"] is False
            and hellaswag_hidden_innovation_bagged_gate["packet_contract"]["source_kv_exposed"] is False
            and hellaswag_hidden_innovation_bagged_gate["packet_contract"]["raw_hidden_vector_transmitted"]
            is False
            and hellaswag_hidden_innovation_bagged_gate["packet_contract"]["raw_scores_transmitted"] is False,
        ),
        (
            "hellaswag_hidden_innovation_eval_slice_stress_passes",
            bool(hellaswag_hidden_innovation_eval_slice_stress["pass_gate"]),
        ),
        (
            "hellaswag_hidden_innovation_eval_slice_stress_is_heldout_1024_rows",
            hellaswag_hidden_innovation_eval_slice_stress["headline"]["eval_slice_start"] >= 1024
            and hellaswag_hidden_innovation_eval_slice_stress["headline"]["eval_rows"] >= 1024,
        ),
        (
            "hellaswag_hidden_innovation_eval_slice_stress_beats_label_and_score_only",
            hellaswag_hidden_innovation_eval_slice_stress["headline"]["selected_minus_best_label_copy"] >= 0.02
            and hellaswag_hidden_innovation_eval_slice_stress["headline"][
                "paired_ci95_low_vs_best_label_copy"
            ]
            > 0.0
            and hellaswag_hidden_innovation_eval_slice_stress["headline"][
                "selected_minus_score_only_bagged_control"
            ]
            >= 0.02
            and hellaswag_hidden_innovation_eval_slice_stress["headline"][
                "paired_ci95_low_vs_score_only_bagged"
            ]
            > 0.0,
        ),
        (
            "hellaswag_hidden_innovation_eval_slice_stress_controls_and_jackknife",
            hellaswag_hidden_innovation_eval_slice_stress["headline"]["selected_minus_zero_hidden_control"] >= 0.02
            and hellaswag_hidden_innovation_eval_slice_stress["headline"]["wrong_example_hidden_control_accuracy"]
            <= hellaswag_hidden_innovation_eval_slice_stress["headline"]["best_label_copy_eval_accuracy"]
            and hellaswag_hidden_innovation_eval_slice_stress["headline"][
                "candidate_roll_hidden_control_accuracy"
            ]
            <= hellaswag_hidden_innovation_eval_slice_stress["headline"]["best_label_copy_eval_accuracy"]
            and hellaswag_hidden_innovation_eval_slice_stress["headline"]["jackknife_pass_count"]
            == hellaswag_hidden_innovation_eval_slice_stress["headline"]["jackknife_row_count"],
        ),
        (
            "hellaswag_hidden_innovation_eval_slice_stress_source_private_packet",
            hellaswag_hidden_innovation_eval_slice_stress["packet_contract"]["raw_payload_bytes"] == 2
            and hellaswag_hidden_innovation_eval_slice_stress["packet_contract"]["framed_record_bytes"] == 5
            and hellaswag_hidden_innovation_eval_slice_stress["packet_contract"]["source_text_exposed"] is False
            and hellaswag_hidden_innovation_eval_slice_stress["packet_contract"]["source_kv_exposed"] is False
            and hellaswag_hidden_innovation_eval_slice_stress["packet_contract"][
                "raw_hidden_vector_transmitted"
            ]
            is False
            and hellaswag_hidden_innovation_eval_slice_stress["packet_contract"]["raw_scores_transmitted"] is False,
        ),
        (
            "hellaswag_hidden_innovation_multi_slice_stress_passes",
            bool(hellaswag_hidden_innovation_multi_slice_stress["pass_gate"]),
        ),
        (
            "hellaswag_hidden_innovation_multi_slice_stress_has_3_contiguous_slices",
            hellaswag_hidden_innovation_multi_slice_stress["headline"]["slice_count"] >= 3
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["pass_slice_count"]
            == hellaswag_hidden_innovation_multi_slice_stress["headline"]["slice_count"]
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["total_eval_rows"] >= 3072
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["contiguous_validation_prefix"]
            is True
            and hellaswag_hidden_innovation_eval_slice_stress_2048_3072["headline"]["eval_slice_start"]
            == 2048,
        ),
        (
            "hellaswag_hidden_innovation_multi_slice_stress_has_4_contiguous_slices",
            hellaswag_hidden_innovation_multi_slice_stress["headline"]["slice_count"] >= 4
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["pass_slice_count"]
            == hellaswag_hidden_innovation_multi_slice_stress["headline"]["slice_count"]
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["total_eval_rows"] >= 4096
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["contiguous_validation_prefix"]
            is True,
        ),
        (
            "hellaswag_hidden_innovation_multi_slice_stress_has_5_contiguous_slices",
            hellaswag_hidden_innovation_multi_slice_stress["headline"]["slice_count"] >= 5
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["pass_slice_count"]
            == hellaswag_hidden_innovation_multi_slice_stress["headline"]["slice_count"]
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["total_eval_rows"] >= 5120
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["contiguous_validation_prefix"]
            is True,
        ),
        (
            "hellaswag_hidden_innovation_multi_slice_stress_has_6_contiguous_slices",
            hellaswag_hidden_innovation_multi_slice_stress["headline"]["slice_count"] >= 6
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["pass_slice_count"]
            == hellaswag_hidden_innovation_multi_slice_stress["headline"]["slice_count"]
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["total_eval_rows"] >= 6144
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["contiguous_validation_prefix"]
            is True,
        ),
        (
            "hellaswag_hidden_innovation_multi_slice_stress_has_7_contiguous_slices",
            hellaswag_hidden_innovation_multi_slice_stress["headline"]["slice_count"] >= 7
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["pass_slice_count"]
            == hellaswag_hidden_innovation_multi_slice_stress["headline"]["slice_count"]
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["total_eval_rows"] >= 7168
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["contiguous_validation_prefix"]
            is True,
        ),
        (
            "hellaswag_hidden_innovation_multi_slice_stress_has_8_contiguous_slices",
            hellaswag_hidden_innovation_multi_slice_stress["headline"]["slice_count"] >= 8
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["pass_slice_count"]
            == hellaswag_hidden_innovation_multi_slice_stress["headline"]["slice_count"]
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["total_eval_rows"] >= 8192
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["contiguous_validation_prefix"]
            is True,
        ),
        (
            "hellaswag_hidden_innovation_multi_slice_stress_has_9_contiguous_slices",
            hellaswag_hidden_innovation_multi_slice_stress["headline"]["slice_count"] >= 9
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["pass_slice_count"]
            == hellaswag_hidden_innovation_multi_slice_stress["headline"]["slice_count"]
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["total_eval_rows"] >= 9216
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["contiguous_validation_prefix"]
            is True,
        ),
        (
            "hellaswag_hidden_innovation_terminal_tail_soft_fail_recorded",
            hellaswag_hidden_innovation_terminal_tail["headline"]["eval_slice_start"] == 9216
            and hellaswag_hidden_innovation_terminal_tail["headline"]["eval_slice_end_exclusive"] == 10042
            and hellaswag_hidden_innovation_terminal_tail["headline"]["eval_rows"] == 826
            and hellaswag_hidden_innovation_terminal_tail["headline"]["terminal_tail_slice"] is True
            and hellaswag_hidden_innovation_terminal_tail["pass_gate"] is False
            and hellaswag_hidden_innovation_terminal_tail["headline"]["selected_minus_best_label_copy"] >= 0.02
            and hellaswag_hidden_innovation_terminal_tail["headline"]["paired_ci95_low_vs_best_label_copy"] > 0.0
            and hellaswag_hidden_innovation_terminal_tail["headline"]["jackknife_pass_count"]
            < hellaswag_hidden_innovation_terminal_tail["headline"]["jackknife_row_count"],
        ),
        (
            "hellaswag_hidden_innovation_full_validation_not_overclaimed",
            hellaswag_hidden_innovation_full_validation_multi_slice["pass_gate"] is False
            and hellaswag_hidden_innovation_full_validation_multi_slice["headline"]["slice_count"] == 10
            and hellaswag_hidden_innovation_full_validation_multi_slice["headline"]["pass_slice_count"] == 9
            and hellaswag_hidden_innovation_full_validation_multi_slice["headline"]["total_eval_rows"] == 10042
            and hellaswag_hidden_innovation_full_validation_multi_slice["headline"][
                "weighted_selected_eval_accuracy"
            ]
            > hellaswag_hidden_innovation_full_validation_multi_slice["headline"][
                "weighted_best_label_copy_eval_accuracy"
            ]
            and hellaswag_hidden_innovation_full_validation_multi_slice["headline"][
                "contiguous_validation_prefix"
            ]
            is True,
        ),
        (
            "hellaswag_hidden_innovation_multi_slice_stress_beats_label_score_zero",
            hellaswag_hidden_innovation_multi_slice_stress["headline"]["min_delta_vs_best_label_copy"] >= 0.02
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["min_ci95_low_vs_best_label_copy"]
            > 0.0
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["min_delta_vs_score_only_bagged"]
            >= 0.02
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["min_ci95_low_vs_score_only_bagged"]
            > 0.0
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["min_delta_vs_zero_hidden"] >= 0.02,
        ),
        (
            "hellaswag_hidden_innovation_multi_slice_stress_controls_and_jackknife",
            hellaswag_hidden_innovation_multi_slice_stress["headline"][
                "corrupted_hidden_controls_below_label_copy"
            ]
            is True
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["jackknife_slice_pass_count"]
            == hellaswag_hidden_innovation_multi_slice_stress["headline"]["slice_count"],
        ),
        (
            "hellaswag_hidden_innovation_multi_slice_stress_source_private_packet",
            hellaswag_hidden_innovation_multi_slice_stress["headline"]["source_private_packet"] is True
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["raw_payload_bytes"] == 2
            and hellaswag_hidden_innovation_multi_slice_stress["headline"]["framed_record_bytes"] == 5,
        ),
        (
            "hellaswag_anchor_relative_hidden_innovation_multi_slice_recorded",
            hellaswag_anchor_relative_hidden_innovation_multi_slice["headline"]["slice_count"] >= 5
            and hellaswag_anchor_relative_hidden_innovation_multi_slice["headline"]["total_eval_rows"] >= 5120
            and hellaswag_anchor_relative_hidden_innovation_multi_slice["headline"][
                "contiguous_validation_prefix"
            ]
            is True,
        ),
        (
            "hellaswag_anchor_relative_hidden_innovation_common_basis_demoted",
            hellaswag_anchor_relative_hidden_innovation_multi_slice["pass_gate"] is False
            and hellaswag_anchor_relative_hidden_innovation_multi_slice["headline"][
                "weighted_delta_vs_best_label_copy"
            ]
            > 0.0
            and hellaswag_anchor_relative_hidden_innovation_multi_slice["headline"][
                "min_delta_vs_best_label_copy"
            ]
            < 0.02
            and hellaswag_anchor_relative_hidden_innovation_multi_slice["headline"][
                "weighted_delta_vs_dense_hidden_innovation"
            ]
            < -0.02,
        ),
        (
            "hellaswag_anchor_relative_hidden_innovation_source_private_packet",
            hellaswag_anchor_relative_hidden_innovation_multi_slice["headline"]["source_private_packet"] is True
            and hellaswag_anchor_relative_hidden_innovation_multi_slice["headline"]["raw_payload_bytes"] == 2
            and hellaswag_anchor_relative_hidden_innovation_multi_slice["headline"]["framed_record_bytes"] == 5,
        ),
        (
            "hellaswag_pq_hidden_innovation_codec_not_overclaimed",
            hellaswag_pq_hidden_innovation_codec["pass_gate"] is False
            and hellaswag_pq_hidden_innovation_codec["headline"]["default_delta_vs_packet_only"] < 0.0
            and hellaswag_pq_hidden_innovation_codec["headline"]["best_scout_delta_vs_packet_only"] < 0.010
            and hellaswag_pq_hidden_innovation_codec["headline"][
                "best_scout_ci95_low_vs_packet_only"
            ]
            <= 0.0,
        ),
        (
            "hellaswag_pq_hidden_innovation_codec_source_private_packet",
            hellaswag_pq_hidden_innovation_codec["packet_contract"]["raw_payload_bytes"] == 1
            and hellaswag_pq_hidden_innovation_codec["packet_contract"]["framed_record_bytes"] == 4
            and hellaswag_pq_hidden_innovation_codec["packet_contract"]["source_text_exposed"] is False
            and hellaswag_pq_hidden_innovation_codec["packet_contract"]["source_kv_exposed"] is False
            and hellaswag_pq_hidden_innovation_codec["packet_contract"]["raw_hidden_vector_transmitted"]
            is False
            and hellaswag_pq_hidden_innovation_codec["packet_contract"]["raw_scores_transmitted"] is False,
        ),
        (
            "hellaswag_pq_hidden_innovation_codec_controls_collapse",
            hellaswag_pq_hidden_innovation_codec["headline"]["control_separation_gate"] is False
            and hellaswag_pq_hidden_innovation_codec["headline"]["control_max_delta_vs_packet_only"] <= 0.0
            and any(
                row["name"] == "candidate_only_code"
                and row["delta_vs_packet_only"] == 0.0
                for row in hellaswag_pq_hidden_innovation_codec["control_rows"]
            ),
        ),
        ("hellaswag_repair_systems_acceptance_card_passes", bool(hellaswag_repair_systems_acceptance["pass_gate"])),
        (
            "hellaswag_repair_systems_acceptance_method_promoted",
            hellaswag_repair_systems_acceptance["headline"]["method_gate_pass"] is True
            and hellaswag_repair_systems_acceptance["headline"]["best_delta_vs_source_label_copy"] >= 0.02
            and hellaswag_repair_systems_acceptance["headline"]["best_repair_row_id"]
            == "hidden_innovation_repair",
        ),
        (
            "hellaswag_repair_systems_acceptance_trained_label_control_clears",
            hellaswag_repair_systems_acceptance["headline"]["trained_label_copy_control_rows"] >= 1
            and hellaswag_repair_systems_acceptance["headline"]["best_delta_vs_trained_label_copy"] >= 0.02
            and all(
                row.get("delta_vs_trained_label_copy") is None
                or row["delta_vs_trained_label_copy"] >= 0.02
                or not row["method_gate_pass"]
                for row in hellaswag_repair_systems_acceptance["rows"]
            ),
        ),
        (
            "hellaswag_repair_systems_acceptance_audit_passes",
            hellaswag_repair_systems_acceptance["headline"]["systems_audit_pass"] is True
            and all(row["systems_audit_pass"] for row in hellaswag_repair_systems_acceptance["rows"]),
        ),
        (
            "hellaswag_repair_systems_acceptance_native_queue_blocked",
            hellaswag_repair_systems_acceptance["headline"]["native_queue_allowed"] is False
            and hellaswag_repair_systems_acceptance["headline"]["native_ready"] is False,
        ),
        (
            "hellaswag_repair_systems_acceptance_has_byte_latency_exposure_fields",
            all(
                row["raw_payload_bytes"] is not None
                and row["framed_record_bytes"] is not None
                and row["batch64_cacheline_bytes_per_request"] is not None
                and row["source_text_exposed"] is False
                and row["source_kv_exposed"] is False
                for row in hellaswag_repair_systems_acceptance["rows"]
            ),
        ),
        ("mac_packet_ring_transport_passes", bool(mac_packet_ring["pass_gate"])),
        ("serving_slo_envelope_passes", bool(serving_slo["pass_gate"])),
        ("systems_rate_assumption_frontier_passes", bool(systems_rate_assumption["pass_gate"])),
        ("cross_benchmark_systems_comparator_passes", bool(cross_benchmark_systems["pass_gate"])),
        (
            "cross_benchmark_systems_two_headline_rows",
            cross_benchmark_systems["headline"]["headline_eligible_benchmarks"] >= 2
            and cross_benchmark_systems["headline"]["diagnostic_benchmarks"] >= 1,
        ),
        (
            "cross_benchmark_systems_qjl_floor_above_50x",
            cross_benchmark_systems["headline"]["min_qjl_1bit_ratio_vs_framed"] >= 50.0,
        ),
        (
            "cross_benchmark_systems_native_nonclaim",
            cross_benchmark_systems["headline"]["native_systems_complete"] is False
            and any(
                check["check"] == "native_baseline_non_claims_explicit" and check["pass"]
                for check in cross_benchmark_systems["checks"]
            ),
        ),
        (
            "native_readiness_ledger_scopes_native_blocker",
            native_readiness["headline"]["native_ready"] is False
            and native_readiness["headline"]["pending_native_rows"] >= 5
            and native_readiness["headline"]["local_measured_rows"] >= 3,
        ),
        ("native_systems_benchmark_plan_passes", bool(native_systems_plan["pass_gate"])),
        (
            "native_systems_benchmark_plan_nonclaim",
            native_systems_plan["headline"]["native_systems_complete"] is False
            and any(
                check["check"] == "native_win_non_claims_recorded" and check["pass"]
                for check in native_systems_plan["checks"]
            ),
        ),
        (
            "native_systems_benchmark_plan_has_required_baselines",
            native_systems_plan["headline"]["required_baseline_count"] >= 10
            and native_systems_plan["headline"]["serving_substrates"] == ["vLLM", "SGLang"],
        ),
        (
            "native_systems_benchmark_plan_has_hardware_metrics",
            {"ttft_ms_p50", "tpot_ms_p50", "goodput_requests_per_s", "peak_gpu_memory_gb",
             "hbm_read_bytes_per_request", "pcie_or_nvlink_rx_bytes_per_request", "source_kv_exposed"}
            <= {row["metric"] for row in native_systems_plan["required_metrics"]},
        ),
        (
            "candidate_local_live_rows_9_of_9",
            candidate_competitor["headline"]["live_pass_rows"] == candidate_competitor["headline"]["live_rows"] == 9,
        ),
        (
            "candidate_local_cross_family_6_of_6",
            candidate_cross_family["headline"]["live_cross_family_pass_rows"]
            == candidate_cross_family["headline"]["live_cross_family_rows"]
            == 6,
        ),
        (
            "candidate_local_no_source_text_or_kv",
            not candidate_systems["headline"]["source_text_exposed"]
            and not candidate_systems["headline"]["source_kv_exposed"],
        ),
        ("candidate_local_systems_boundary_passes", bool(candidate_systems["headline"]["pass_gate"])),
        (
            "candidate_local_kv_proxy_ratio_above_1000x",
            candidate_systems["headline"]["min_kv_native_proxy_record_ratio_vs_live"] >= 1000.0,
        ),
        (
            "candidate_local_threshold_0_48_clean_9_of_9",
            candidate_threshold["headline"]["live_threshold_0_48_clean_rows"]
            == candidate_threshold["headline"]["live_threshold_0_48_rows"]
            == 9,
        ),
        (
            "candidate_local_rr_has_no_all_row_clean_threshold",
            not candidate_threshold["headline"]["rr_clean_threshold_range"]["exists"],
        ),
        (
            "candidate_local_random_sign_has_no_all_row_clean_threshold",
            not candidate_threshold["headline"]["random_rotation_sign_clean_threshold_range"]["exists"],
        ),
        ("candidate_local_margin_atlas_passes", bool(candidate_margin["headline"]["pass_gate"])),
        (
            "candidate_local_margin_beats_best_control_by_2x",
            candidate_margin["headline"]["live_matched_positive_margin_rate"]
            >= 2.0 * candidate_margin["headline"]["live_best_control_positive_margin_rate"],
        ),
        (
            "train_receiver_permuted_null_gap_3_seed_cross_family_pass",
            train_receiver_headline["all_seed_cross_family_pass"],
        ),
        (
            "train_receiver_permuted_null_gap_6_of_6_cross_rows_pass",
            train_receiver_headline["cross_pass_rows"] == train_receiver_headline["cross_row_count"] == 6,
        ),
        (
            "train_receiver_permuted_null_gap_controls_clean",
            train_receiver_headline["max_cross_best_control_accuracy"]
            <= train_receiver_headline["max_cross_target_accuracy"] + 0.03,
        ),
        (
            "train_receiver_permuted_null_gap_ci_positive",
            train_receiver_headline["min_passing_cross_ci95_low_vs_target"] > 0.15,
        ),
        (
            "procrustes_margin_control_matches_matched",
            candidate_margin["headline"]["procrustes_best_control_positive_margin_rate"]
            >= candidate_margin["headline"]["procrustes_matched_positive_margin_rate"],
        ),
        (
            "train_sender_packet_builder_3_seed_cross_family_pass",
            train_sender_packet_builder_headline["all_seed_cross_family_pass"],
        ),
        ("train_sender_packet_builder_9_of_9_rows_pass", train_sender_packet_builder_headline["pass_rows"] == 9),
        (
            "train_sender_packet_builder_beats_live_base",
            train_sender_packet_builder_headline["min_candidate_minus_base"] >= 0.10,
        ),
        (
            "train_sender_packet_builder_controls_clean",
            train_sender_packet_builder_headline["max_best_control_accuracy"]
            <= train_sender_packet_builder_headline["max_target_accuracy"] + 0.03,
        ),
        ("train_sender_packet_builder_rate_has_cross_family_pass", train_sender_rate_headline["all_seed_cross_family_pass"]),
        ("train_sender_packet_builder_rate_has_multiple_pass_rows", train_sender_rate_headline["pass_rows"] >= 6),
        (
            "source_prioritized_packet_builder_loo_3_seed_cross_family_pass",
            loo_packet_builder_headline["all_seed_cross_family_pass"],
        ),
        ("source_prioritized_packet_builder_loo_9_of_9_rows_pass", loo_packet_builder_headline["pass_rows"] == 9),
        ("source_prioritized_packet_builder_loo_beats_live_base", loo_packet_builder_headline["min_candidate_minus_base"] >= 0.10),
        (
            "source_prioritized_packet_builder_loo_controls_clean",
            loo_packet_builder_headline["max_best_control_accuracy"] <= loo_packet_builder_headline["max_target_accuracy"] + 0.03,
        ),
        (
            "public_packet_builder_3_seed_cross_family_pass",
            public_packet_builder_headline["all_seed_cross_family_pass"],
        ),
        ("public_packet_builder_9_of_9_rows_pass", public_packet_builder_headline["pass_rows"] == 9),
        ("public_packet_builder_beats_live_base", public_packet_builder_headline["min_candidate_minus_base"] >= 0.20),
        (
            "public_packet_builder_controls_clean",
            public_packet_builder_headline["max_best_control_accuracy"] <= public_packet_builder_headline["max_target_accuracy"] + 0.03,
        ),
        ("rate_frontier_passes", bool(rate["pass_gate"])),
        ("matched_byte_text_stays_at_target", rate["headline"]["matched_byte_text_at_packet_accuracy_max"] <= 0.25),
        ("packet_beats_query_aware_text_by_7x", rate["headline"]["packet_vs_query_aware_oracle_compression_min"] >= 7.0),
        ("kv_cache_qjl_lower_bound_above_1000x", kv["headline"]["min_non_packet_qjl_1bit_bytes_vs_packet"] >= 1000.0),
        ("coded_label_risk_passes", bool(coded["pass_gate"])),
        (
            "composed_label_code_order_stress_passes",
            bool(coded["by_transform"]["label_code_order_composed"]["pass_gate"]),
        ),
        ("endpoint_core_uncertainty_passes", bool(endpoint_core["pass_gate"])),
        ("endpoint_holdout_uncertainty_passes", bool(endpoint_holdout["pass_gate"])),
        ("ledger_has_paper_ready_rows", len(ledger["paper_ready_rows"]) >= 3),
    ]
    return [{"check": name, "pass": bool(value)} for name, value in checks]


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Source-Private ICLR Evidence Bundle",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- created UTC: `{payload['created_utc']}`",
        f"- current readiness: `{payload['readiness']}`",
        "",
        "## Technical Contributions",
        "",
        "| Contribution | Status | Headline evidence | Main metric | Remaining gap |",
        "|---|---|---|---|---|",
    ]
    for row in payload["contribution_rows"]:
        lines.append(
            "| "
            f"{row['contribution']} | {row['status']} | {row['headline_evidence']} | "
            f"{row['main_metric']} | {row['remaining_gap']} |"
        )
    lines.extend(
        [
            "",
            "## Pass Checks",
            "",
            "| Check | Pass |",
            "|---|---|",
        ]
    )
    for check in payload["pass_checks"]:
        lines.append(f"| `{check['check']}` | `{check['pass']}` |")
    lines.extend(
        [
            "",
            "## Novelty Matrix",
            "",
            "| Comparison | Source | Communicated object | Source-private | Internals? | Extreme rate? | Controls? | Paper role |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in payload["novelty_matrix"]:
        lines.append(
            "| "
            f"{row['comparison']} | {row['source']} | {row['communicated_object']} | "
            f"{row['source_private']} | {row['requires_model_internals']} | "
            f"{row['extreme_byte_rate']} | {row['source_destroying_controls']} | {row['paper_role']} |"
        )
    lines.extend(
        [
            "",
            "## Reproduction Commands",
            "",
            "```bash",
            *payload["reproduction_commands"],
            "```",
            "",
            "## Remaining ICLR Risks",
            "",
        ]
    )
    for risk in payload["remaining_iclr_risks"]:
        lines.append(f"- {risk}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_bundle(*, output_dir: pathlib.Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _artifact_status()
    missing = [name for name, row in artifacts.items() if not row["exists"]]
    if missing:
        raise FileNotFoundError(f"missing required artifacts: {', '.join(missing)}")

    rate = _read_json(ROOT / REQUIRED_ARTIFACTS["rate_frontier"])
    kv = _read_json(ROOT / REQUIRED_ARTIFACTS["kv_cache_baseline"])
    coded = _read_json(ROOT / REQUIRED_ARTIFACTS["coded_label_risk"])
    ledger = _read_json(ROOT / REQUIRED_ARTIFACTS["pass_fail_ledger"])
    endpoint_core = _read_json(ROOT / REQUIRED_ARTIFACTS["endpoint_uncertainty_core"])
    endpoint_holdout = _read_json(ROOT / REQUIRED_ARTIFACTS["endpoint_uncertainty_holdout"])
    candidate_competitor = _read_json(ROOT / REQUIRED_ARTIFACTS["candidate_local_competitor_basis"])
    candidate_cross_family = _read_json(ROOT / REQUIRED_ARTIFACTS["candidate_local_cross_family"])
    candidate_systems = _read_json(ROOT / REQUIRED_ARTIFACTS["candidate_local_systems_boundary"])
    candidate_threshold = _read_json(ROOT / REQUIRED_ARTIFACTS["candidate_local_threshold_frontier"])
    candidate_margin = _read_json(ROOT / REQUIRED_ARTIFACTS["candidate_local_margin_atlas"])
    public_packet_builder_runs = [
        _read_json(ROOT / REQUIRED_ARTIFACTS["candidate_conditioned_packet_builder_seed47"]),
        _read_json(ROOT / REQUIRED_ARTIFACTS["candidate_conditioned_packet_builder_seed53"]),
        _read_json(ROOT / REQUIRED_ARTIFACTS["candidate_conditioned_packet_builder_seed59"]),
    ]
    loo_packet_builder_runs = [
        _read_json(ROOT / REQUIRED_ARTIFACTS["source_prioritized_packet_builder_loo_seed47"]),
        _read_json(ROOT / REQUIRED_ARTIFACTS["source_prioritized_packet_builder_loo_seed53"]),
        _read_json(ROOT / REQUIRED_ARTIFACTS["source_prioritized_packet_builder_loo_seed59"]),
    ]
    train_sender_packet_builder_runs = [
        _read_json(ROOT / REQUIRED_ARTIFACTS["train_sender_packet_builder_seed47"]),
        _read_json(ROOT / REQUIRED_ARTIFACTS["train_sender_packet_builder_seed53"]),
        _read_json(ROOT / REQUIRED_ARTIFACTS["train_sender_packet_builder_seed59"]),
    ]
    train_sender_rate_runs = [_read_json(ROOT / REQUIRED_ARTIFACTS["train_sender_packet_builder_rate"])]
    train_receiver_runs = [
        _read_json(ROOT / REQUIRED_ARTIFACTS["train_receiver_permuted_null_gap_seed47"]),
        _read_json(ROOT / REQUIRED_ARTIFACTS["train_receiver_permuted_null_gap_seed53"]),
        _read_json(ROOT / REQUIRED_ARTIFACTS["train_receiver_permuted_null_gap_seed59"]),
    ]
    train_donor_antishuffle_runs = [
        _read_json(ROOT / REQUIRED_ARTIFACTS["train_donor_antishuffle_seed47_n128"]),
        _read_json(ROOT / REQUIRED_ARTIFACTS["train_donor_antishuffle_seed53_n128"]),
        _read_json(ROOT / REQUIRED_ARTIFACTS["train_donor_antishuffle_seed59_n128"]),
    ]
    train_donor_antishuffle_n512_runs = [
        _read_json(ROOT / REQUIRED_ARTIFACTS["train_donor_antishuffle_seed47_n512"]),
        _read_json(ROOT / REQUIRED_ARTIFACTS["train_donor_antishuffle_seed53_n512"]),
        _read_json(ROOT / REQUIRED_ARTIFACTS["train_donor_antishuffle_seed59_n512"]),
    ]
    train_donor_locked_frontier = _read_json(ROOT / REQUIRED_ARTIFACTS["train_donor_locked_rate_frontier"])
    train_donor_stable_gap = _read_json(ROOT / REQUIRED_ARTIFACTS["train_donor_stable_gap_selector"])
    train_donor_fixed12_eval = _read_json(ROOT / REQUIRED_ARTIFACTS["train_donor_fixed12_eval_audit"])
    arc_contract = _read_json(ROOT / REQUIRED_ARTIFACTS["arc_challenge_bridge_contract"])
    arc_fixed_validation = _read_json(ROOT / REQUIRED_ARTIFACTS["arc_challenge_fixed_packet_validation"])
    arc_fixed_test = _read_json(ROOT / REQUIRED_ARTIFACTS["arc_challenge_fixed_packet_test"])
    arc_seed_validation = _read_json(ROOT / REQUIRED_ARTIFACTS["arc_challenge_seed_stability_validation"])
    arc_seed_test = _read_json(ROOT / REQUIRED_ARTIFACTS["arc_challenge_seed_stability_test"])
    arc_common_validation = _read_json(ROOT / REQUIRED_ARTIFACTS["arc_challenge_common_basis_validation"])
    arc_common_test = _read_json(ROOT / REQUIRED_ARTIFACTS["arc_challenge_common_basis_test"])
    arc_common_seed_validation = _read_json(ROOT / REQUIRED_ARTIFACTS["arc_challenge_common_basis_seed_validation"])
    arc_common_seed_test = _read_json(ROOT / REQUIRED_ARTIFACTS["arc_challenge_common_basis_seed_test"])
    arc_anchor_relative_seed_validation = _read_json(
        ROOT / REQUIRED_ARTIFACTS["arc_challenge_anchor_relative_seed_validation"]
    )
    arc_anchor_relative_seed_test = _read_json(
        ROOT / REQUIRED_ARTIFACTS["arc_challenge_anchor_relative_seed_test"]
    )
    arc_anchor_id_shuffle_validation = _read_json(
        ROOT / REQUIRED_ARTIFACTS["arc_challenge_anchor_id_shuffle_validation"]
    )
    arc_anchor_id_shuffle_test = _read_json(ROOT / REQUIRED_ARTIFACTS["arc_challenge_anchor_id_shuffle_test"])
    arc_anchor_value_shuffle_validation = _read_json(
        ROOT / REQUIRED_ARTIFACTS["arc_challenge_anchor_value_shuffle_validation"]
    )
    arc_anchor_value_shuffle_test = _read_json(ROOT / REQUIRED_ARTIFACTS["arc_challenge_anchor_value_shuffle_test"])
    arc_random_anchors_validation = _read_json(ROOT / REQUIRED_ARTIFACTS["arc_challenge_random_anchors_validation"])
    arc_random_anchors_test = _read_json(ROOT / REQUIRED_ARTIFACTS["arc_challenge_random_anchors_test"])
    arc_source_latent_validation = _read_json(ROOT / REQUIRED_ARTIFACTS["arc_challenge_source_latent_endpoint_validation"])
    arc_systems_trace = _read_json(ROOT / REQUIRED_ARTIFACTS["arc_challenge_systems_trace"])
    sciq_contract = _read_json(ROOT / REQUIRED_ARTIFACTS["sciq_bridge_contract"])
    sciq_fixed_validation = _read_json(ROOT / REQUIRED_ARTIFACTS["sciq_fixed_packet_validation"])
    openbookqa_contract = _read_json(ROOT / REQUIRED_ARTIFACTS["openbookqa_bridge_contract"])
    openbookqa_fixed_validation = _read_json(ROOT / REQUIRED_ARTIFACTS["openbookqa_fixed_packet_validation"])
    openbookqa_fixed_test_4b = _read_json(ROOT / REQUIRED_ARTIFACTS["openbookqa_fixed_packet_test_4b"])
    openbookqa_seed_validation_3b = _read_json(
        ROOT / REQUIRED_ARTIFACTS["openbookqa_seed_stability_validation_3b"]
    )
    openbookqa_seed_test_3b = _read_json(ROOT / REQUIRED_ARTIFACTS["openbookqa_seed_stability_test_3b"])
    openbookqa_receiver_headroom = _read_json(ROOT / REQUIRED_ARTIFACTS["openbookqa_receiver_headroom"])
    commonsenseqa_contract = _read_json(ROOT / REQUIRED_ARTIFACTS["commonsenseqa_bridge_contract"])
    commonsenseqa_fixed_validation_12b = _read_json(
        ROOT / REQUIRED_ARTIFACTS["commonsenseqa_fixed_packet_validation_12b"]
    )
    commonsenseqa_seed_validation_2b_strict = _read_json(
        ROOT / REQUIRED_ARTIFACTS["commonsenseqa_seed_validation_2b_strict"]
    )
    commonsenseqa_seed_validation_2b_gap001 = _read_json(
        ROOT / REQUIRED_ARTIFACTS["commonsenseqa_seed_validation_2b_gap001"]
    )
    hellaswag_contract = _read_json(ROOT / REQUIRED_ARTIFACTS["hellaswag_bridge_contract"])
    hellaswag_fixed_validation1024_2b = _read_json(
        ROOT / REQUIRED_ARTIFACTS["hellaswag_fixed_packet_validation1024_2b"]
    )
    hellaswag_seed_validation1024_2b_5seed = _read_json(
        ROOT / REQUIRED_ARTIFACTS["hellaswag_seed_validation1024_2b_5seed"]
    )
    hellaswag_control_suite = _read_json(ROOT / REQUIRED_ARTIFACTS["hellaswag_control_suite"])
    hellaswag_score_packet_headroom = _read_json(ROOT / REQUIRED_ARTIFACTS["hellaswag_score_packet_headroom"])
    hellaswag_public_receiver_repair = _read_json(ROOT / REQUIRED_ARTIFACTS["hellaswag_public_receiver_repair"])
    hellaswag_train_source_score_repair = _read_json(
        ROOT / REQUIRED_ARTIFACTS["hellaswag_train_source_score_repair"]
    )
    hellaswag_hidden_summary_repair = _read_json(
        ROOT / REQUIRED_ARTIFACTS["hellaswag_hidden_summary_repair"]
    )
    hellaswag_hidden_innovation_repair = _read_json(
        ROOT / REQUIRED_ARTIFACTS["hellaswag_hidden_innovation_repair"]
    )
    hellaswag_hidden_innovation_stability = _read_json(
        ROOT / REQUIRED_ARTIFACTS["hellaswag_hidden_innovation_stability"]
    )
    hellaswag_hidden_innovation_train_sample_stress = _read_json(
        ROOT / REQUIRED_ARTIFACTS["hellaswag_hidden_innovation_train_sample_stress"]
    )
    hellaswag_hidden_innovation_bagged_gate = _read_json(
        ROOT / REQUIRED_ARTIFACTS["hellaswag_hidden_innovation_bagged_gate"]
    )
    hellaswag_hidden_innovation_eval_slice_stress = _read_json(
        ROOT / REQUIRED_ARTIFACTS["hellaswag_hidden_innovation_eval_slice_stress"]
    )
    hellaswag_hidden_innovation_eval_slice_stress_2048_3072 = _read_json(
        ROOT / REQUIRED_ARTIFACTS["hellaswag_hidden_innovation_eval_slice_stress_2048_3072"]
    )
    hellaswag_hidden_innovation_multi_slice_stress = _read_json(
        ROOT / REQUIRED_ARTIFACTS["hellaswag_hidden_innovation_multi_slice_stress"]
    )
    hellaswag_hidden_innovation_terminal_tail = _read_json(
        ROOT / REQUIRED_ARTIFACTS["hellaswag_hidden_innovation_terminal_tail"]
    )
    hellaswag_hidden_innovation_full_validation_multi_slice = _read_json(
        ROOT / REQUIRED_ARTIFACTS["hellaswag_hidden_innovation_full_validation_multi_slice"]
    )
    hellaswag_anchor_relative_hidden_innovation_multi_slice = _read_json(
        ROOT / REQUIRED_ARTIFACTS["hellaswag_anchor_relative_hidden_innovation_multi_slice"]
    )
    hellaswag_repair_systems_acceptance = _read_json(
        ROOT / REQUIRED_ARTIFACTS["hellaswag_repair_systems_acceptance"]
    )
    hellaswag_pq_hidden_innovation_codec = _read_json(
        ROOT / REQUIRED_ARTIFACTS["hellaswag_pq_hidden_innovation_codec"]
    )
    mac_packet_ring = _read_json(ROOT / REQUIRED_ARTIFACTS["mac_packet_ring_transport"])
    serving_slo = _read_json(ROOT / REQUIRED_ARTIFACTS["serving_slo_envelope"])
    systems_rate_assumption = _read_json(ROOT / REQUIRED_ARTIFACTS["systems_rate_assumption_frontier"])
    cross_benchmark_systems = _read_json(ROOT / REQUIRED_ARTIFACTS["cross_benchmark_systems_comparator"])
    native_readiness = _read_json(ROOT / REQUIRED_ARTIFACTS["native_readiness_ledger"])
    native_systems_plan = _read_json(ROOT / REQUIRED_ARTIFACTS["native_systems_benchmark_plan"])
    public_packet_builder_headline = _packet_builder_headline(public_packet_builder_runs)
    loo_packet_builder_headline = _packet_builder_headline(loo_packet_builder_runs)
    train_sender_packet_builder_headline = _packet_builder_headline(train_sender_packet_builder_runs)
    train_sender_rate_headline = _packet_builder_headline(train_sender_rate_runs)
    train_receiver_headline = _train_receiver_headline(train_receiver_runs)
    train_donor_antishuffle_headline = _packet_builder_cross_family_headline(train_donor_antishuffle_runs)
    train_donor_antishuffle_n512_headline = _packet_builder_cross_family_headline(
        train_donor_antishuffle_n512_runs
    )

    contribution_rows = _contribution_rows(
        rate=rate,
        kv=kv,
        coded=coded,
        ledger=ledger,
        endpoint_core=endpoint_core,
        endpoint_holdout=endpoint_holdout,
        candidate_competitor=candidate_competitor,
        candidate_cross_family=candidate_cross_family,
        candidate_systems=candidate_systems,
        candidate_threshold=candidate_threshold,
        candidate_margin=candidate_margin,
        train_sender_packet_builder_headline=train_sender_packet_builder_headline,
        train_sender_rate_headline=train_sender_rate_headline,
        public_packet_builder_headline=public_packet_builder_headline,
        loo_packet_builder_headline=loo_packet_builder_headline,
        train_receiver_headline=train_receiver_headline,
        train_donor_antishuffle_headline=train_donor_antishuffle_headline,
        train_donor_antishuffle_n512_headline=train_donor_antishuffle_n512_headline,
        train_donor_locked_frontier=train_donor_locked_frontier,
        train_donor_stable_gap=train_donor_stable_gap,
        train_donor_fixed12_eval=train_donor_fixed12_eval,
        arc_contract=arc_contract,
        arc_fixed_validation=arc_fixed_validation,
        arc_fixed_test=arc_fixed_test,
        arc_seed_validation=arc_seed_validation,
        arc_seed_test=arc_seed_test,
        arc_common_validation=arc_common_validation,
        arc_common_test=arc_common_test,
        arc_common_seed_validation=arc_common_seed_validation,
        arc_common_seed_test=arc_common_seed_test,
        arc_anchor_relative_seed_validation=arc_anchor_relative_seed_validation,
        arc_anchor_relative_seed_test=arc_anchor_relative_seed_test,
        arc_anchor_id_shuffle_validation=arc_anchor_id_shuffle_validation,
        arc_anchor_id_shuffle_test=arc_anchor_id_shuffle_test,
        arc_anchor_value_shuffle_validation=arc_anchor_value_shuffle_validation,
        arc_anchor_value_shuffle_test=arc_anchor_value_shuffle_test,
        arc_random_anchors_validation=arc_random_anchors_validation,
        arc_random_anchors_test=arc_random_anchors_test,
        arc_source_latent_validation=arc_source_latent_validation,
        arc_systems_trace=arc_systems_trace,
        sciq_contract=sciq_contract,
        sciq_fixed_validation=sciq_fixed_validation,
        openbookqa_contract=openbookqa_contract,
        openbookqa_fixed_validation=openbookqa_fixed_validation,
        openbookqa_fixed_test_4b=openbookqa_fixed_test_4b,
        openbookqa_seed_validation_3b=openbookqa_seed_validation_3b,
        openbookqa_seed_test_3b=openbookqa_seed_test_3b,
        openbookqa_receiver_headroom=openbookqa_receiver_headroom,
        commonsenseqa_contract=commonsenseqa_contract,
        commonsenseqa_fixed_validation_12b=commonsenseqa_fixed_validation_12b,
        commonsenseqa_seed_validation_2b_strict=commonsenseqa_seed_validation_2b_strict,
        commonsenseqa_seed_validation_2b_gap001=commonsenseqa_seed_validation_2b_gap001,
        hellaswag_contract=hellaswag_contract,
        hellaswag_fixed_validation1024_2b=hellaswag_fixed_validation1024_2b,
        hellaswag_seed_validation1024_2b_5seed=hellaswag_seed_validation1024_2b_5seed,
        hellaswag_control_suite=hellaswag_control_suite,
        hellaswag_score_packet_headroom=hellaswag_score_packet_headroom,
        hellaswag_public_receiver_repair=hellaswag_public_receiver_repair,
        hellaswag_train_source_score_repair=hellaswag_train_source_score_repair,
        hellaswag_hidden_summary_repair=hellaswag_hidden_summary_repair,
        hellaswag_hidden_innovation_repair=hellaswag_hidden_innovation_repair,
        hellaswag_hidden_innovation_stability=hellaswag_hidden_innovation_stability,
        hellaswag_hidden_innovation_train_sample_stress=hellaswag_hidden_innovation_train_sample_stress,
        hellaswag_hidden_innovation_bagged_gate=hellaswag_hidden_innovation_bagged_gate,
        hellaswag_hidden_innovation_eval_slice_stress=hellaswag_hidden_innovation_eval_slice_stress,
        hellaswag_hidden_innovation_eval_slice_stress_2048_3072=hellaswag_hidden_innovation_eval_slice_stress_2048_3072,
        hellaswag_hidden_innovation_multi_slice_stress=hellaswag_hidden_innovation_multi_slice_stress,
        hellaswag_hidden_innovation_terminal_tail=hellaswag_hidden_innovation_terminal_tail,
        hellaswag_hidden_innovation_full_validation_multi_slice=hellaswag_hidden_innovation_full_validation_multi_slice,
        hellaswag_anchor_relative_hidden_innovation_multi_slice=hellaswag_anchor_relative_hidden_innovation_multi_slice,
        hellaswag_repair_systems_acceptance=hellaswag_repair_systems_acceptance,
        hellaswag_pq_hidden_innovation_codec=hellaswag_pq_hidden_innovation_codec,
        mac_packet_ring=mac_packet_ring,
        cross_benchmark_systems=cross_benchmark_systems,
        native_readiness=native_readiness,
        native_systems_plan=native_systems_plan,
    )
    pass_checks = _pass_checks(
        artifacts=artifacts,
        rate=rate,
        kv=kv,
        coded=coded,
        ledger=ledger,
        endpoint_core=endpoint_core,
        endpoint_holdout=endpoint_holdout,
        candidate_competitor=candidate_competitor,
        candidate_cross_family=candidate_cross_family,
        candidate_systems=candidate_systems,
        candidate_threshold=candidate_threshold,
        candidate_margin=candidate_margin,
        train_sender_packet_builder_headline=train_sender_packet_builder_headline,
        train_sender_rate_headline=train_sender_rate_headline,
        public_packet_builder_headline=public_packet_builder_headline,
        loo_packet_builder_headline=loo_packet_builder_headline,
        train_receiver_headline=train_receiver_headline,
        train_donor_antishuffle_headline=train_donor_antishuffle_headline,
        train_donor_antishuffle_n512_headline=train_donor_antishuffle_n512_headline,
        train_donor_locked_frontier=train_donor_locked_frontier,
        train_donor_stable_gap=train_donor_stable_gap,
        train_donor_fixed12_eval=train_donor_fixed12_eval,
        arc_contract=arc_contract,
        arc_fixed_validation=arc_fixed_validation,
        arc_fixed_test=arc_fixed_test,
        arc_seed_validation=arc_seed_validation,
        arc_seed_test=arc_seed_test,
        arc_common_validation=arc_common_validation,
        arc_common_test=arc_common_test,
        arc_common_seed_validation=arc_common_seed_validation,
        arc_common_seed_test=arc_common_seed_test,
        arc_anchor_relative_seed_validation=arc_anchor_relative_seed_validation,
        arc_anchor_relative_seed_test=arc_anchor_relative_seed_test,
        arc_anchor_id_shuffle_validation=arc_anchor_id_shuffle_validation,
        arc_anchor_id_shuffle_test=arc_anchor_id_shuffle_test,
        arc_anchor_value_shuffle_validation=arc_anchor_value_shuffle_validation,
        arc_anchor_value_shuffle_test=arc_anchor_value_shuffle_test,
        arc_random_anchors_validation=arc_random_anchors_validation,
        arc_random_anchors_test=arc_random_anchors_test,
        arc_source_latent_validation=arc_source_latent_validation,
        arc_systems_trace=arc_systems_trace,
        sciq_contract=sciq_contract,
        sciq_fixed_validation=sciq_fixed_validation,
        openbookqa_contract=openbookqa_contract,
        openbookqa_fixed_validation=openbookqa_fixed_validation,
        openbookqa_fixed_test_4b=openbookqa_fixed_test_4b,
        openbookqa_seed_validation_3b=openbookqa_seed_validation_3b,
        openbookqa_seed_test_3b=openbookqa_seed_test_3b,
        openbookqa_receiver_headroom=openbookqa_receiver_headroom,
        commonsenseqa_contract=commonsenseqa_contract,
        commonsenseqa_fixed_validation_12b=commonsenseqa_fixed_validation_12b,
        commonsenseqa_seed_validation_2b_strict=commonsenseqa_seed_validation_2b_strict,
        commonsenseqa_seed_validation_2b_gap001=commonsenseqa_seed_validation_2b_gap001,
        hellaswag_contract=hellaswag_contract,
        hellaswag_fixed_validation1024_2b=hellaswag_fixed_validation1024_2b,
        hellaswag_seed_validation1024_2b_5seed=hellaswag_seed_validation1024_2b_5seed,
        hellaswag_control_suite=hellaswag_control_suite,
        hellaswag_score_packet_headroom=hellaswag_score_packet_headroom,
        hellaswag_public_receiver_repair=hellaswag_public_receiver_repair,
        hellaswag_train_source_score_repair=hellaswag_train_source_score_repair,
        hellaswag_hidden_summary_repair=hellaswag_hidden_summary_repair,
        hellaswag_hidden_innovation_repair=hellaswag_hidden_innovation_repair,
        hellaswag_hidden_innovation_stability=hellaswag_hidden_innovation_stability,
        hellaswag_hidden_innovation_train_sample_stress=hellaswag_hidden_innovation_train_sample_stress,
        hellaswag_hidden_innovation_bagged_gate=hellaswag_hidden_innovation_bagged_gate,
        hellaswag_hidden_innovation_eval_slice_stress=hellaswag_hidden_innovation_eval_slice_stress,
        hellaswag_hidden_innovation_eval_slice_stress_2048_3072=hellaswag_hidden_innovation_eval_slice_stress_2048_3072,
        hellaswag_hidden_innovation_multi_slice_stress=hellaswag_hidden_innovation_multi_slice_stress,
        hellaswag_hidden_innovation_terminal_tail=hellaswag_hidden_innovation_terminal_tail,
        hellaswag_hidden_innovation_full_validation_multi_slice=hellaswag_hidden_innovation_full_validation_multi_slice,
        hellaswag_anchor_relative_hidden_innovation_multi_slice=hellaswag_anchor_relative_hidden_innovation_multi_slice,
        hellaswag_repair_systems_acceptance=hellaswag_repair_systems_acceptance,
        hellaswag_pq_hidden_innovation_codec=hellaswag_pq_hidden_innovation_codec,
        mac_packet_ring=mac_packet_ring,
        serving_slo=serving_slo,
        systems_rate_assumption=systems_rate_assumption,
        cross_benchmark_systems=cross_benchmark_systems,
        native_readiness=native_readiness,
        native_systems_plan=native_systems_plan,
    )
    payload = {
        "gate": "source_private_iclr_evidence_bundle",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "readiness": "COLM is now strong around fixed-byte source-private packets, ARC-Challenge/OpenBookQA public-basis endpoints, the OpenBookQA packet/target receiver-fusion row, and byte/exposure systems accounting. ICLR remains blocked by robustness rather than by total absence of a positive receiver: the OpenBookQA receiver improves over packet-only on held-out test, but strict per-seed CI/control stability, ARC replication, native NVIDIA systems baselines, and a less label-copy-like common-basis or learned connector are still missing. HellaSwag is now a diagnostic/headroom and negative-ablation surface rather than a current receiver-improvement headline.",
        "pass_gate": all(check["pass"] for check in pass_checks),
        "pass_checks": pass_checks,
        "artifact_status": artifacts,
        "contribution_rows": contribution_rows,
        "novelty_matrix": NOVELTY_MATRIX,
        "candidate_conditioned_packet_builder_headline": public_packet_builder_headline,
        "source_prioritized_packet_builder_loo_headline": loo_packet_builder_headline,
        "train_sender_packet_builder_headline": train_sender_packet_builder_headline,
        "train_sender_packet_builder_rate_headline": train_sender_rate_headline,
        "train_receiver_permuted_null_gap_headline": train_receiver_headline,
        "train_donor_antishuffle_headline": train_donor_antishuffle_headline,
        "train_donor_antishuffle_n512_headline": train_donor_antishuffle_n512_headline,
        "train_donor_locked_rate_frontier_headline": train_donor_locked_frontier["policies"],
        "train_donor_stable_gap_selector_headline": train_donor_stable_gap["policies"],
        "train_donor_fixed12_eval_headline": train_donor_fixed12_eval["headline"],
        "arc_challenge_bridge_contract_headline": {
            "pass_gate": arc_contract["pass_gate"],
            "public_benchmark_result_ready": arc_contract["public_benchmark_result_ready"],
            "local_checks": arc_contract["checks"],
            "official_rows": {
                split: summary["n"] for split, summary in arc_contract["official_summaries"].items()
            },
        },
        "arc_challenge_fixed_packet_validation_headline": arc_fixed_validation["headline"],
        "arc_challenge_fixed_packet_test_headline": arc_fixed_test["headline"],
        "arc_challenge_seed_stability_validation_headline": arc_seed_validation["aggregate"],
        "arc_challenge_seed_stability_test_headline": arc_seed_test["aggregate"],
        "arc_challenge_common_basis_validation_headline": arc_common_validation["headline"],
        "arc_challenge_common_basis_test_headline": arc_common_test["headline"],
        "arc_challenge_common_basis_seed_validation_headline": arc_common_seed_validation["aggregate"],
        "arc_challenge_common_basis_seed_test_headline": arc_common_seed_test["aggregate"],
        "arc_challenge_anchor_relative_seed_validation_headline": arc_anchor_relative_seed_validation["aggregate"],
        "arc_challenge_anchor_relative_seed_test_headline": arc_anchor_relative_seed_test["aggregate"],
        "arc_challenge_anchor_id_shuffle_validation_headline": arc_anchor_id_shuffle_validation["aggregate"],
        "arc_challenge_anchor_id_shuffle_test_headline": arc_anchor_id_shuffle_test["aggregate"],
        "arc_challenge_anchor_value_shuffle_validation_headline": arc_anchor_value_shuffle_validation["aggregate"],
        "arc_challenge_anchor_value_shuffle_test_headline": arc_anchor_value_shuffle_test["aggregate"],
        "arc_challenge_random_anchors_validation_headline": arc_random_anchors_validation["aggregate"],
        "arc_challenge_random_anchors_test_headline": arc_random_anchors_test["aggregate"],
        "arc_challenge_source_latent_endpoint_validation_headline": arc_source_latent_validation["headline"],
        "arc_challenge_systems_trace_headline": arc_systems_trace["headline"],
        "sciq_bridge_contract_headline": {
            "pass_gate": sciq_contract["pass_gate"],
            "local_checks": sciq_contract["checks"],
            "official_rows": {
                split: summary["n"] for split, summary in sciq_contract["official_summaries"].items()
            },
        },
        "sciq_fixed_packet_validation_headline": sciq_fixed_validation["headline"],
        "openbookqa_bridge_contract_headline": {
            "pass_gate": openbookqa_contract["pass_gate"],
            "local_checks": openbookqa_contract["checks"],
            "official_rows": {
                split: summary["n"] for split, summary in openbookqa_contract["official_summaries"].items()
            },
        },
        "openbookqa_fixed_packet_validation_headline": openbookqa_fixed_validation["headline"],
        "openbookqa_fixed_packet_test_4b_headline": openbookqa_fixed_test_4b["headline"],
        "openbookqa_seed_stability_validation_3b_headline": openbookqa_seed_validation_3b["aggregate"],
        "openbookqa_seed_stability_test_3b_headline": openbookqa_seed_test_3b["aggregate"],
        "openbookqa_receiver_headroom_headline": openbookqa_receiver_headroom["headline"],
        "commonsenseqa_bridge_contract_headline": {
            "pass_gate": commonsenseqa_contract["pass_gate"],
            "local_checks": commonsenseqa_contract["checks"],
            "labeled_rows": {
                split: summary["n"] for split, summary in commonsenseqa_contract["labeled_summaries"].items()
            },
        },
        "commonsenseqa_fixed_packet_validation_12b_headline": commonsenseqa_fixed_validation_12b["headline"],
        "commonsenseqa_seed_stability_validation_2b_strict_headline": commonsenseqa_seed_validation_2b_strict[
            "aggregate"
        ],
        "commonsenseqa_seed_stability_validation_2b_gap001_headline": commonsenseqa_seed_validation_2b_gap001[
            "aggregate"
        ],
        "hellaswag_bridge_contract_headline": {
            "pass_gate": hellaswag_contract["pass_gate"],
            "local_checks": hellaswag_contract["checks"],
            "labeled_rows": {
                split: summary["n"] for split, summary in hellaswag_contract["labeled_summaries"].items()
            },
        },
        "hellaswag_fixed_packet_validation1024_2b_headline": hellaswag_fixed_validation1024_2b["headline"],
        "hellaswag_seed_stability_validation1024_2b_headline": hellaswag_seed_validation1024_2b_5seed[
            "aggregate"
        ],
        "hellaswag_control_suite_headline": hellaswag_control_suite["headline"],
        "hellaswag_score_packet_headroom_headline": hellaswag_score_packet_headroom["headline"],
        "hellaswag_public_receiver_repair_headline": hellaswag_public_receiver_repair["headline"],
        "hellaswag_train_source_score_repair_headline": hellaswag_train_source_score_repair["headline"],
        "hellaswag_hidden_summary_repair_headline": {
            **hellaswag_hidden_summary_repair["headline"],
            "pass_gate": hellaswag_hidden_summary_repair["pass_gate"],
            "selected_layer": hellaswag_hidden_summary_repair["hidden_model_selection"]["selected_layer"],
            "selected_ridge_alpha": hellaswag_hidden_summary_repair["hidden_model_selection"]["selected_ridge"],
            "selected_packet_raw_bytes": hellaswag_hidden_summary_repair["packet_contract"]["raw_payload_bytes"],
            "selected_packet_framed_bytes": hellaswag_hidden_summary_repair["packet_contract"][
                "framed_record_bytes"
            ],
            "total_wall_time_sec": hellaswag_hidden_summary_repair["timing"]["total_seconds"],
        },
        "hellaswag_hidden_innovation_repair_headline": {
            **hellaswag_hidden_innovation_repair["headline"],
            "pass_gate": hellaswag_hidden_innovation_repair["pass_gate"],
            "selected_packet_raw_bytes": hellaswag_hidden_innovation_repair["packet_contract"][
                "raw_payload_bytes"
            ],
            "selected_packet_framed_bytes": hellaswag_hidden_innovation_repair["packet_contract"][
                "framed_record_bytes"
            ],
            "total_wall_time_sec": hellaswag_hidden_innovation_repair["timing"]["total_seconds"],
        },
        "hellaswag_hidden_innovation_stability_headline": {
            **hellaswag_hidden_innovation_stability["headline"],
            "pass_gate": hellaswag_hidden_innovation_stability["pass_gate"],
            "selected_packet_raw_bytes": hellaswag_hidden_innovation_stability["packet_contract"][
                "raw_payload_bytes"
            ],
            "selected_packet_framed_bytes": hellaswag_hidden_innovation_stability["packet_contract"][
                "framed_record_bytes"
            ],
            "unrestricted_diagnostic": hellaswag_hidden_innovation_stability[
                "unrestricted_model_selection_diagnostic"
            ],
            "total_wall_time_sec": hellaswag_hidden_innovation_stability["timing"]["total_seconds"],
        },
        "hellaswag_hidden_innovation_train_sample_stress_headline": {
            **hellaswag_hidden_innovation_train_sample_stress["headline"],
            "pass_gate": hellaswag_hidden_innovation_train_sample_stress["pass_gate"],
            "selected_packet_raw_bytes": hellaswag_hidden_innovation_train_sample_stress["packet_contract"][
                "raw_payload_bytes"
            ],
            "selected_packet_framed_bytes": hellaswag_hidden_innovation_train_sample_stress["packet_contract"][
                "framed_record_bytes"
            ],
            "total_wall_time_sec": hellaswag_hidden_innovation_train_sample_stress["timing"]["total_seconds"],
        },
        "hellaswag_hidden_innovation_bagged_gate_headline": {
            **hellaswag_hidden_innovation_bagged_gate["headline"],
            "pass_gate": hellaswag_hidden_innovation_bagged_gate["pass_gate"],
            "jackknife_summary": hellaswag_hidden_innovation_bagged_gate["jackknife_summary"],
            "selected_packet_raw_bytes": hellaswag_hidden_innovation_bagged_gate["packet_contract"][
                "raw_payload_bytes"
            ],
            "selected_packet_framed_bytes": hellaswag_hidden_innovation_bagged_gate["packet_contract"][
                "framed_record_bytes"
            ],
            "total_wall_time_sec": hellaswag_hidden_innovation_bagged_gate["timing"]["total_seconds"],
        },
        "hellaswag_hidden_innovation_eval_slice_stress_headline": {
            **hellaswag_hidden_innovation_eval_slice_stress["headline"],
            "pass_gate": hellaswag_hidden_innovation_eval_slice_stress["pass_gate"],
            "slice_path": hellaswag_hidden_innovation_eval_slice_stress["slice_metadata"]["slice_path"],
            "total_wall_time_sec": hellaswag_hidden_innovation_eval_slice_stress["timing"]["total_seconds"],
        },
        "hellaswag_hidden_innovation_eval_slice_stress_2048_3072_headline": {
            **hellaswag_hidden_innovation_eval_slice_stress_2048_3072["headline"],
            "pass_gate": hellaswag_hidden_innovation_eval_slice_stress_2048_3072["pass_gate"],
            "slice_path": hellaswag_hidden_innovation_eval_slice_stress_2048_3072["slice_metadata"][
                "slice_path"
            ],
            "total_wall_time_sec": hellaswag_hidden_innovation_eval_slice_stress_2048_3072["timing"][
                "total_seconds"
            ],
        },
        "hellaswag_hidden_innovation_multi_slice_stress_headline": {
            **hellaswag_hidden_innovation_multi_slice_stress["headline"],
            "pass_gate": hellaswag_hidden_innovation_multi_slice_stress["pass_gate"],
            "slice_artifacts": hellaswag_hidden_innovation_multi_slice_stress["slice_artifacts"],
        },
        "hellaswag_hidden_innovation_terminal_tail_headline": {
            **hellaswag_hidden_innovation_terminal_tail["headline"],
            "pass_gate": hellaswag_hidden_innovation_terminal_tail["pass_gate"],
            "slice_path": hellaswag_hidden_innovation_terminal_tail["slice_metadata"]["slice_path"],
            "total_wall_time_sec": hellaswag_hidden_innovation_terminal_tail["timing"]["total_seconds"],
        },
        "hellaswag_hidden_innovation_full_validation_multi_slice_headline": {
            **hellaswag_hidden_innovation_full_validation_multi_slice["headline"],
            "pass_gate": hellaswag_hidden_innovation_full_validation_multi_slice["pass_gate"],
            "slice_artifacts": hellaswag_hidden_innovation_full_validation_multi_slice["slice_artifacts"],
        },
        "hellaswag_anchor_relative_hidden_innovation_multi_slice_headline": {
            **hellaswag_anchor_relative_hidden_innovation_multi_slice["headline"],
            "pass_gate": hellaswag_anchor_relative_hidden_innovation_multi_slice["pass_gate"],
            "slice_artifacts": hellaswag_anchor_relative_hidden_innovation_multi_slice["slice_artifacts"],
        },
        "hellaswag_repair_systems_acceptance_headline": hellaswag_repair_systems_acceptance["headline"],
        "hellaswag_pq_hidden_innovation_codec_headline": {
            **hellaswag_pq_hidden_innovation_codec["headline"],
            "pass_gate": hellaswag_pq_hidden_innovation_codec["pass_gate"],
            "packet_contract": hellaswag_pq_hidden_innovation_codec["packet_contract"],
            "pass_rule": hellaswag_pq_hidden_innovation_codec["pass_rule"],
            "systems_packet_sideband": hellaswag_pq_hidden_innovation_codec["systems_packet_sideband"],
        },
        "mac_packet_ring_transport_headline": mac_packet_ring["headline"],
        "serving_slo_envelope_headline": serving_slo["headline"],
        "systems_rate_assumption_frontier_headline": systems_rate_assumption["headline"],
        "cross_benchmark_systems_comparator_headline": cross_benchmark_systems["headline"],
        "native_readiness_ledger_headline": native_readiness["headline"],
        "native_systems_benchmark_plan_headline": native_systems_plan["headline"],
        "reproduction_commands": REPRODUCTION_COMMANDS,
        "remaining_iclr_risks": [
            "Production serving TTFT/TPOT/throughput on NVIDIA GPUs is still missing; the cross-benchmark systems comparator and native benchmark plan define the table, but they are not native serving results.",
            "ARC-Challenge, OpenBookQA, and HellaSwag now have seed-stable public-basis endpoints, but the source scorer is still a local Qwen log-likelihood bridge rather than a learned hidden-state communication endpoint.",
            "HellaSwag anchored hidden-innovation repair passes cached split stability but fails the fresh 2027 train-row-sample stress; the bagged gate rescues several cached/slice gates, but the final terminal tail validation[9216:10042] soft-fails the strict jackknife gate and the later PQ hidden-code gate fails versus packet-only. HellaSwag must therefore be framed as diagnostic/headroom plus negative-ablation evidence, not as a current receiver-improvement headline.",
            "The HellaSwag anchor-relative/common-basis hidden-innovation variant preserves only a small weighted lift over label-copy and score-only controls and fails all five strict slices, so the paper cannot yet claim a robust shared-coordinate mechanism.",
            "The HellaSwag PQ hidden-code gate also fails: the predeclared default is below packet-only and the best diagnostic scout is below the +0.010 promotion bar with CI95 low exactly zero, so Mac-local hidden-code/codebook widening is cut.",
            "HellaSwag source top-2 oracle headroom is still large after the hidden-innovation repair, so reviewers will expect ablations explaining why the repair recovers only part of the headroom.",
            "The train-only public receiver HellaSwag repair probe also fails below source-label copy, so the next repair gate needs train-split source scores or hidden source summaries rather than public lexical features alone.",
            "The 512-row train-source-score HellaSwag repair probe also fails below source-label copy, so score-shape repair is weakened unless a substantially richer train-source feature family is introduced.",
            "The 512-row train-source-hidden HellaSwag label-copy repair fails below source-label copy, while hidden-innovation residual denoising succeeds; the paper must explain this distinction clearly.",
            "The HellaSwag repair acceptance card now promotes only the hidden-innovation row locally and still blocks native-queue promotion until native systems evidence exists.",
            "SciQ is documented as a benchmark-selection limitation because same-byte answer-text saturates despite strong source signal.",
            "CommonsenseQA confirms non-science source signal, but same-byte text is still too close under the strict text-margin gate.",
            "Anchor-coordinate controls show ID/value mismatch collapse, but random shared anchors pass; do not claim semantic train-anchor superiority.",
            "The direct Qwen-hidden to BGE residual endpoint failed validation, so deeper latent endpoints need better common-basis learning before promotion.",
            "The top live method is candidate-local and thresholded; the paper must show the threshold frontier and controls clearly.",
            "Same-family structured-text controls remain unpromoted and should be framed as a limitation or cut from headline claims.",
            "The headline method is protocol/candidate-side-information communication, not universal semantic latent transfer.",
            "Simple learned cross-family masked-innovation receivers failed; a future shared-dictionary/crosscoder method needs feature knockout before promotion.",
            "The final paper must show text relay catches up at higher byte budgets to avoid unfair-baseline criticism.",
        ],
    }

    (output_dir / "iclr_evidence_bundle.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(output_dir / "iclr_evidence_bundle.md", payload)
    _write_csv(output_dir / "novelty_matrix.csv", NOVELTY_MATRIX)
    _write_csv(output_dir / "contribution_matrix.csv", contribution_rows)
    commands_path = output_dir / "reproduce_iclr_evidence_bundle.sh"
    commands_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(REPRODUCTION_COMMANDS) + "\n",
        encoding="utf-8",
    )
    commands_path.chmod(commands_path.stat().st_mode | stat.S_IXUSR)

    artifacts_to_hash = [
        "iclr_evidence_bundle.json",
        "iclr_evidence_bundle.md",
        "novelty_matrix.csv",
        "contribution_matrix.csv",
        "reproduce_iclr_evidence_bundle.sh",
        "manifest.json",
        "manifest.md",
    ]
    manifest = {
        "command": "./venv_arm64/bin/python scripts/build_source_private_iclr_evidence_bundle.py --output-dir "
        + str(output_dir.relative_to(ROOT) if output_dir.is_relative_to(ROOT) else output_dir),
        "artifacts": artifacts_to_hash,
        "artifact_sha256": {
            name: _sha256_file(output_dir / name)
            for name in artifacts_to_hash
            if name not in {"manifest.json", "manifest.md"}
        },
        "pass_gate": payload["pass_gate"],
        "python": sys.version,
        "script_sha256": _sha256_file(pathlib.Path(__file__)),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Source-Private ICLR Evidence Bundle Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- contributions: `{len(contribution_rows)}`",
                f"- novelty comparisons: `{len(NOVELTY_MATRIX)}`",
                "",
                "## Artifacts",
                "",
                *[f"- `{name}`" for name in artifacts_to_hash],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = build_bundle(output_dir=output_dir)
    if not payload["pass_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
