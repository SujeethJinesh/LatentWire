from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_INPUT_PATHS = {
    "evidence_table": ROOT
    / "results/latentwire_colm_v2_iclr_evidence_table_20260504/evidence_table.json",
    "live_branch_triage": ROOT
    / "results/iclr_colm_v2_live_branch_triage_20260504/live_branch_triage.json",
    "conditional_pq_status": ROOT
    / "results/source_private_conditional_pq_iclr_colm_v2_status_20260504/conditional_pq_iclr_colm_v2_status.json",
    "hellaswag_fixed_hybrid": ROOT
    / "results/source_private_hellaswag_fixed_hybrid_full_validation_gate_20260503_validation0_10042/hellaswag_fixed_hybrid_full_validation_gate.json",
    "openbookqa_receiver_headroom": ROOT
    / "results/source_private_openbookqa_receiver_headroom_gate_20260502/openbookqa_receiver_headroom_gate.json",
    "systems_boundary": ROOT
    / "results/source_private_systems_boundary_figure_table_split_20260504/systems_boundary_figure_data.json",
}

DEFAULT_OUTPUT_DIR = ROOT / "results/latentwire_colm_v2_review_packet_20260504"
DEFAULT_PAPER_PATH = ROOT / "paper/latentwire_colm_v2_review_packet_20260504.md"


BASELINE_MATRIX = [
    {
        "category": "dense_cache_transfer",
        "baseline": "Cache-to-Cache (C2C)",
        "source": "https://openreview.net/forum?id=LeatkxrBCi",
        "what_transfers": "projected source KV cache fused into target KV cache",
        "source_private": "no, dense source cache state crosses the interface",
        "byte_regime": "high bandwidth KV/cache state",
        "receiver_training": "learned projection/gating",
        "target_access": "target internals",
        "included_in_current_eval": "systems-boundary comparator only",
        "latentwire_distinction": (
            "LatentWire transmits tiny source-private packets and evaluates utility per byte; "
            "it does not claim raw-accuracy or native serving wins over C2C."
        ),
        "still_needed": "direct native C2C run on matching models/tasks for any stronger claim",
    },
    {
        "category": "dense_cache_transfer",
        "baseline": "Latent Space Communication via K-V Cache Alignment",
        "source": "https://arxiv.org/abs/2601.06123",
        "what_transfers": "aligned prefix K/V cache from source model into target model decoding",
        "source_private": "no, aligned cache state crosses the boundary",
        "byte_regime": "dense K/V cache state",
        "receiver_training": "cache alignment map",
        "target_access": "target cache path",
        "included_in_current_eval": "related-work boundary",
        "latentwire_distinction": (
            "K/V cache alignment is a direct dense latent-communication competitor; "
            "LatentWire must claim only low-rate packet transfer unless direct cache "
            "alignment rows are run."
        ),
        "still_needed": "direct comparison if claiming broad latent communication novelty",
    },
    {
        "category": "kv_cache_serving",
        "baseline": "DroidSpeak",
        "source": "https://arxiv.org/abs/2411.02820",
        "what_transfers": "compatible KV caches across distributed LLM nodes",
        "source_private": "no, cache state is reused",
        "byte_regime": "KV/cache state",
        "receiver_training": "serving/runtime method",
        "target_access": "compatible architecture cache path",
        "included_in_current_eval": "related-work and systems-boundary comparator",
        "latentwire_distinction": (
            "DroidSpeak is cache reuse; LatentWire studies source-private task packets "
            "with wrong-row and source-choice controls."
        ),
        "still_needed": "native serving comparison only after GPU setup",
    },
    {
        "category": "kv_cache_serving",
        "baseline": "KVCOMM",
        "source": "https://arxiv.org/abs/2510.12872",
        "what_transfers": "cross-context KV-cache information for multi-agent inference",
        "source_private": "no, cache information crosses the agent boundary",
        "byte_regime": "KV/cache state",
        "receiver_training": "training-free cache communication",
        "target_access": "cache-level access",
        "included_in_current_eval": "related-work and systems-boundary comparator",
        "latentwire_distinction": (
            "KVCOMM is a high-bandwidth cache communication regime; LatentWire is "
            "a low-byte packet protocol with destructive source-private controls."
        ),
        "still_needed": "native baseline if claiming runtime superiority",
    },
    {
        "category": "kv_cache_serving",
        "baseline": "RelayCaching",
        "source": "https://arxiv.org/abs/2603.13289",
        "what_transfers": "decoding-time KV caches for collaborative generation",
        "source_private": "no, produced cache state is reused",
        "byte_regime": "KV/cache state",
        "receiver_training": "serving/runtime method",
        "target_access": "cache-level access",
        "included_in_current_eval": "related-work and systems-boundary comparator",
        "latentwire_distinction": (
            "RelayCaching accelerates cache reuse; LatentWire evaluates whether compact "
            "source evidence causes target utility without exposing cache/text."
        ),
        "still_needed": "native serving baseline for latency or throughput claims",
    },
    {
        "category": "kv_cache_reuse",
        "baseline": "CacheGen / LMCache / CacheBlend-style reuse",
        "source": "https://arxiv.org/abs/2310.07240",
        "what_transfers": "compressed, streamed, or blended reusable prompt/cache state",
        "source_private": "no, cache or prompt-derived state is reused",
        "byte_regime": "compressed cache/prompt state",
        "receiver_training": "system-dependent",
        "target_access": "serving cache path",
        "included_in_current_eval": "systems-boundary comparator",
        "latentwire_distinction": (
            "Cache-reuse systems optimize serving reuse for existing context; LatentWire "
            "sends small task-level source evidence packets across model boundaries."
        ),
        "still_needed": "native vLLM/SGLang/LMCache run for throughput claims",
    },
    {
        "category": "activation_communication",
        "baseline": "Communicating Activations Between Language Model Agents",
        "source": "https://arxiv.org/abs/2501.14082",
        "what_transfers": "activation vectors between language-model agents",
        "source_private": "limited; activation vectors are exposed",
        "byte_regime": "dense activations",
        "receiver_training": "method-dependent",
        "target_access": "activation-level access",
        "included_in_current_eval": "related-work boundary",
        "latentwire_distinction": (
            "LatentWire emphasizes source-private low-byte packets, explicit packet bytes, "
            "and destructive controls rather than generic activation exchange."
        ),
        "still_needed": "cite and distinguish in related-work matrix",
    },
    {
        "category": "activation_communication",
        "baseline": "CIPHER / Let Models Speak Ciphers",
        "source": "https://arxiv.org/abs/2310.06272",
        "what_transfers": "embedding-level messages between debating LLM agents",
        "source_private": "partial; embedding messages are exposed to peers",
        "byte_regime": "embedding vectors or continuous messages",
        "receiver_training": "method-dependent",
        "target_access": "embedding/message interface",
        "included_in_current_eval": "related-work boundary",
        "latentwire_distinction": (
            "CIPHER pressures broad latent-message novelty; LatentWire's narrower claim "
            "is byte-counted source-private packets with destructive controls."
        ),
        "still_needed": "avoid claiming generic embedding communication novelty",
    },
    {
        "category": "latent_translation",
        "baseline": "Direct Semantic Communication via Vector Translation",
        "source": "https://arxiv.org/abs/2511.03945",
        "what_transfers": "translated semantic vectors",
        "source_private": "partial; dense vectors may expose source state",
        "byte_regime": "dense vector communication",
        "receiver_training": "translation model",
        "target_access": "representation-level access",
        "included_in_current_eval": "related-work boundary",
        "latentwire_distinction": (
            "LatentWire requires low-byte source-private packet accounting and source-choice "
            "destructive controls."
        ),
        "still_needed": "verify final citation metadata before camera-ready",
    },
    {
        "category": "latent_translation",
        "baseline": "InterLat",
        "source": "https://arxiv.org/abs/2511.09149",
        "what_transfers": "intermediate latent representations",
        "source_private": "partial; dense latent state may cross",
        "byte_regime": "latent vector state",
        "receiver_training": "method-dependent",
        "target_access": "latent-level access",
        "included_in_current_eval": "related-work boundary",
        "latentwire_distinction": (
            "LatentWire claims only packetized, byte-counted, source-private transfer "
            "under destructive controls."
        ),
        "still_needed": "verify final citation metadata before camera-ready",
    },
    {
        "category": "soft_prompting",
        "baseline": "Prefix tuning",
        "source": "https://aclanthology.org/2021.acl-long.353/",
        "what_transfers": "task-specific continuous prefix parameters",
        "source_private": "not a source-to-target communication protocol",
        "byte_regime": "learned prompt parameters",
        "receiver_training": "trained prefix",
        "target_access": "target model prompt interface",
        "included_in_current_eval": "conceptual baseline",
        "latentwire_distinction": (
            "Prefix tuning adapts one model; LatentWire tests row-specific source evidence "
            "packets with wrong-row and source-choice controls."
        ),
        "still_needed": "baseline wording in related work",
    },
    {
        "category": "context_compression",
        "baseline": "Gist tokens",
        "source": "https://proceedings.neurips.cc/paper_files/paper/2023/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract-Conference.html",
        "what_transfers": "learned summary tokens for prompt compression",
        "source_private": "no, compression is of visible context",
        "byte_regime": "soft/visible compressed context",
        "receiver_training": "trained compression behavior",
        "target_access": "target prompt/context path",
        "included_in_current_eval": "conceptual baseline",
        "latentwire_distinction": (
            "Gist compresses a model's context; LatentWire transfers hidden source evidence "
            "between models with source-private controls."
        ),
        "still_needed": "same-byte visible text baseline remains the local control",
    },
    {
        "category": "context_compression",
        "baseline": "LLMLingua / LongLLMLingua",
        "source": "https://arxiv.org/abs/2310.05736; https://arxiv.org/abs/2310.06839",
        "what_transfers": "compressed visible prompt text or retained prompt tokens",
        "source_private": "no, visible prompt content is compressed and exposed",
        "byte_regime": "compressed text/token context",
        "receiver_training": "prompt compression model",
        "target_access": "target text prompt interface",
        "included_in_current_eval": "same-byte visible text is the local control",
        "latentwire_distinction": (
            "LLMLingua compresses visible context; LatentWire tests whether opaque "
            "source-private packets add utility beyond same-byte visible text."
        ),
        "still_needed": "optional stronger visible-text compression baseline for ICLR",
    },
    {
        "category": "query_bottleneck",
        "baseline": "BLIP-2 / Q-Former",
        "source": "https://arxiv.org/abs/2301.12597",
        "what_transfers": "query bottleneck between vision encoder and language model",
        "source_private": "not a source-to-target LLM communication protocol",
        "byte_regime": "learned query token bottleneck",
        "receiver_training": "trained Q-Former",
        "target_access": "frozen LM interface",
        "included_in_current_eval": "architectural precedent",
        "latentwire_distinction": (
            "LatentWire's novelty is not query bottlenecks; it is packetized source-private "
            "model-to-model transfer plus destructive controls."
        ),
        "still_needed": "cite as precedent, not baseline claim",
    },
    {
        "category": "query_bottleneck",
        "baseline": "Flamingo / Perceiver Resampler",
        "source": "https://arxiv.org/abs/2204.14198",
        "what_transfers": "fixed latent resampler tokens into LM cross-attention",
        "source_private": "not source-private model-to-model communication",
        "byte_regime": "latent token bottleneck",
        "receiver_training": "trained multimodal bridge",
        "target_access": "LM cross-attention path",
        "included_in_current_eval": "architectural precedent",
        "latentwire_distinction": (
            "LatentWire differs by source-private packets, byte accounting, and controls "
            "against source-choice artifacts."
        ),
        "still_needed": "cite as precedent, not novelty",
    },
    {
        "category": "query_bottleneck",
        "baseline": "Perceiver IO",
        "source": "https://arxiv.org/abs/2107.14795",
        "what_transfers": "latent query array for generic input/output mapping",
        "source_private": "not a source-to-target LLM communication protocol",
        "byte_regime": "latent array",
        "receiver_training": "trained architecture",
        "target_access": "model architecture",
        "included_in_current_eval": "architectural precedent",
        "latentwire_distinction": (
            "Latent arrays are prior art; LatentWire's claim is communication protocol "
            "evaluation under strict controls."
        ),
        "still_needed": "cite as motivation only",
    },
    {
        "category": "common_basis",
        "baseline": "Sparse Crosscoders",
        "source": "https://transformer-circuits.pub/2024/crosscoders/",
        "what_transfers": "shared and private features across activation spaces",
        "source_private": "feature analysis, not packet transfer",
        "byte_regime": "sparse feature coordinates if packetized",
        "receiver_training": "trained crosscoder",
        "target_access": "activation datasets",
        "included_in_current_eval": "related-work boundary",
        "latentwire_distinction": (
            "Common bases alone are not novel; LatentWire must show sparse features cause "
            "source-private downstream utility under atom-shuffle and wrong-row controls."
        ),
        "still_needed": "future SRP method should compare directly if using crosscoders",
    },
    {
        "category": "common_basis",
        "baseline": "Universal Sparse Autoencoders",
        "source": "https://arxiv.org/abs/2502.03714",
        "what_transfers": "cross-model sparse concept basis",
        "source_private": "feature analysis, not packet transfer",
        "byte_regime": "sparse coordinates if packetized",
        "receiver_training": "trained SAE",
        "target_access": "activation datasets",
        "included_in_current_eval": "related-work boundary",
        "latentwire_distinction": (
            "LatentWire uses common-basis work only as a possible packet representation; "
            "novelty requires byte-counted causal packet utility."
        ),
        "still_needed": "future SRP baseline if PCA succeeds",
    },
    {
        "category": "common_basis",
        "baseline": "Transcoders / sparse feature circuits",
        "source": "https://arxiv.org/abs/2406.11944",
        "what_transfers": "sparse feature-to-feature computation approximating model internals",
        "source_private": "analysis method, not direct packet transfer",
        "byte_regime": "sparse feature coordinates if packetized",
        "receiver_training": "trained transcoder",
        "target_access": "activation datasets and internal features",
        "included_in_current_eval": "related-work boundary",
        "latentwire_distinction": (
            "Transcoders can define interpretable atoms, but LatentWire must prove that "
            "packetized atoms causally improve target behavior under strict controls."
        ),
        "still_needed": "strong future baseline for behavior-transcoder packet branch",
    },
    {
        "category": "common_basis",
        "baseline": "SAEBench and SAE non-canonicity checks",
        "source": "https://proceedings.mlr.press/v267/karvonen25a.html",
        "what_transfers": "benchmarking and reliability checks for sparse feature dictionaries",
        "source_private": "analysis/evaluation method",
        "byte_regime": "not a communication protocol",
        "receiver_training": "SAE training/evaluation",
        "target_access": "activation datasets",
        "included_in_current_eval": "related-work boundary",
        "latentwire_distinction": (
            "SAE quality and feature stability are prerequisites if Sparse Resonance "
            "Packets use sparse atoms; they are not themselves the communication result."
        ),
        "still_needed": "use if upgrading from PCA/SVD to SAE/crosscoder packets",
    },
    {
        "category": "quantization",
        "baseline": "TurboQuant",
        "source": "https://arxiv.org/abs/2504.19874",
        "what_transfers": "low-bit quantized vectors/KV state",
        "source_private": "no, quantized source state still exposes state",
        "byte_regime": "low-bit dense vector/KV compression",
        "receiver_training": "quantization/calibration",
        "target_access": "state/vector path",
        "included_in_current_eval": "systems byte floor comparator",
        "latentwire_distinction": (
            "TurboQuant reduces dense state cost; LatentWire sends task-level packets "
            "and must avoid unmeasured throughput claims."
        ),
        "still_needed": "native comparison after NVIDIA hardware",
    },
    {
        "category": "quantization",
        "baseline": "KVQuant",
        "source": "https://arxiv.org/abs/2401.18079",
        "what_transfers": "low-bit KV cache for long-context serving",
        "source_private": "no, cache is still exposed/reused",
        "byte_regime": "low-bit KV cache",
        "receiver_training": "quantization/calibration",
        "target_access": "cache path",
        "included_in_current_eval": "systems byte floor comparator",
        "latentwire_distinction": (
            "KVQuant compresses one model's cache; LatentWire transfers compact source "
            "evidence across models."
        ),
        "still_needed": "native comparison after NVIDIA hardware",
    },
    {
        "category": "quantization",
        "baseline": "KIVI / QJL low-bit KV sketches",
        "source": "https://arxiv.org/abs/2402.02750; https://arxiv.org/abs/2504.19874",
        "what_transfers": "low-bit KV/cache values or sketch-corrected quantized vectors",
        "source_private": "no, dense state is compressed rather than hidden",
        "byte_regime": "low-bit dense cache/vector state",
        "receiver_training": "quantization/calibration or training-free compression",
        "target_access": "cache/vector path",
        "included_in_current_eval": "systems byte floor comparator",
        "latentwire_distinction": (
            "These methods shrink dense state; LatentWire transfers task-level packets "
            "and should compare byte floors without claiming native speed."
        ),
        "still_needed": "native low-bit cache comparison for ICLR systems claims",
    },
    {
        "category": "kv_cache_compression",
        "baseline": "H2O",
        "source": "https://arxiv.org/abs/2306.14048",
        "what_transfers": "retained heavy-hitter KV cache tokens",
        "source_private": "no, same-model cache retention",
        "byte_regime": "pruned KV cache",
        "receiver_training": "training-free cache policy",
        "target_access": "same-model cache path",
        "included_in_current_eval": "related-work boundary",
        "latentwire_distinction": (
            "H2O reduces one model's cache memory; LatentWire sends source-private "
            "cross-model evidence packets."
        ),
        "still_needed": "long-context reviewer baseline, not direct COLM_v2 competitor",
    },
    {
        "category": "kv_cache_compression",
        "baseline": "SnapKV",
        "source": "https://arxiv.org/abs/2404.14469",
        "what_transfers": "selected KV cache tokens before generation",
        "source_private": "no, same-model cache compression",
        "byte_regime": "pruned KV cache",
        "receiver_training": "training-free cache policy",
        "target_access": "same-model cache path",
        "included_in_current_eval": "related-work boundary",
        "latentwire_distinction": (
            "SnapKV is same-model long-context compression; LatentWire is source-to-target "
            "packet communication."
        ),
        "still_needed": "systems appendix comparator if adding long-context tasks",
    },
    {
        "category": "kv_cache_compression",
        "baseline": "Quest",
        "source": "https://arxiv.org/abs/2406.10774",
        "what_transfers": "query-aware sparse KV pages for long-context inference",
        "source_private": "no, same-model cache sparsity",
        "byte_regime": "sparse KV page loading",
        "receiver_training": "training-free cache selection",
        "target_access": "same-model cache path",
        "included_in_current_eval": "related-work boundary",
        "latentwire_distinction": (
            "Quest optimizes attention cache loading; LatentWire optimizes byte-limited "
            "model-to-model evidence transfer."
        ),
        "still_needed": "native systems appendix if reviewers demand long-context baselines",
    },
    {
        "category": "kv_cache_compression",
        "baseline": "KVzip",
        "source": "https://arxiv.org/abs/2505.23416",
        "what_transfers": "query-agnostic compressed KV cache with reconstruction objective",
        "source_private": "no, same-model cache compression",
        "byte_regime": "compressed KV cache",
        "receiver_training": "compression/reconstruction method",
        "target_access": "same-model cache path",
        "included_in_current_eval": "related-work boundary",
        "latentwire_distinction": (
            "KVzip is cache storage compression; LatentWire is packetized cross-model "
            "communication with source-private controls."
        ),
        "still_needed": "appendix context if broad systems framing expands",
    },
    {
        "category": "serving_substrate",
        "baseline": "vLLM / PagedAttention",
        "source": "https://dl.acm.org/doi/10.1145/3600006.3613165",
        "what_transfers": "paged KV-cache serving substrate",
        "source_private": "not a communication method",
        "byte_regime": "serving memory-management baseline",
        "receiver_training": "runtime system",
        "target_access": "serving engine",
        "included_in_current_eval": "native-system blocker only",
        "latentwire_distinction": (
            "vLLM is a required runtime baseline for TTFT/TPOT/goodput/HBM once GPU "
            "measurements are available."
        ),
        "still_needed": "native NVIDIA run before any latency/HBM claim",
    },
    {
        "category": "serving_substrate",
        "baseline": "SGLang / RadixAttention",
        "source": "https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html",
        "what_transfers": "structured LM serving runtime with KV reuse",
        "source_private": "not a communication method",
        "byte_regime": "serving memory-management baseline",
        "receiver_training": "runtime system",
        "target_access": "serving engine",
        "included_in_current_eval": "native-system blocker only",
        "latentwire_distinction": (
            "SGLang is a second serving substrate for native claims; Mac byte accounting "
            "cannot replace it."
        ),
        "still_needed": "native NVIDIA run before any latency/HBM claim",
    },
    {
        "category": "local_control",
        "baseline": "source-index / source-label copy",
        "source": "internal destructive control",
        "what_transfers": "the source's preferred answer index or label",
        "source_private": "yes but artifact-prone",
        "byte_regime": "1-2 bytes",
        "receiver_training": "none or simple receiver",
        "target_access": "answer-choice interface",
        "included_in_current_eval": "yes",
        "latentwire_distinction": (
            "LatentWire must beat this to show more than answer-choice copying."
        ),
        "still_needed": "keep in all packet tables",
    },
    {
        "category": "local_control",
        "baseline": "source-rank / source-score quantization",
        "source": "internal destructive control",
        "what_transfers": "source ranking or quantized score vector",
        "source_private": "yes but may reveal source choice behavior",
        "byte_regime": "few bytes",
        "receiver_training": "none or simple receiver",
        "target_access": "answer-choice interface",
        "included_in_current_eval": "yes",
        "latentwire_distinction": (
            "LatentWire must exceed score/rank packet controls to claim richer communication."
        ),
        "still_needed": "include in claim audit",
    },
    {
        "category": "local_control",
        "baseline": "same-byte visible text",
        "source": "internal destructive control",
        "what_transfers": "human-readable text within the same byte budget",
        "source_private": "no, visible text crosses the boundary",
        "byte_regime": "same byte budget as packet",
        "receiver_training": "none or same receiver",
        "target_access": "prompt text",
        "included_in_current_eval": "yes",
        "latentwire_distinction": (
            "LatentWire packets must justify why opaque source-private bytes are useful "
            "relative to visible text at the same byte budget."
        ),
        "still_needed": "include in strict-control table",
    },
    {
        "category": "local_control",
        "baseline": "wrong-row / shuffled / target-derived packet",
        "source": "internal destructive control",
        "what_transfers": "noncausal or target-only packet with matched format",
        "source_private": "yes",
        "byte_regime": "same packet budget",
        "receiver_training": "same receiver",
        "target_access": "same packet interface",
        "included_in_current_eval": "yes",
        "latentwire_distinction": (
            "Passing these controls is the main evidence that the packet is row-specific "
            "source communication rather than receiver artifact."
        ),
        "still_needed": "keep as hard gate for ICLR",
    },
]


CLAIM_AUDIT = [
    {
        "claim": "LatentWire_v2 provides a source-private packet-transfer evaluation framework.",
        "support_level": "supported_for_colm_v2",
        "safe_wording": (
            "We introduce a source-private, byte-accounted evaluation framework with "
            "matched, wrong-row, shuffled, target-derived, and same-byte controls."
        ),
        "evidence": "evidence_table, live_branch_triage, OpenBookQA hardening artifact",
        "reviewer_risk": "low if framed as framework plus strict controls",
    },
    {
        "claim": "Tiny packets can show narrow utility under strict controls.",
        "support_level": "supported_narrowly",
        "safe_wording": (
            "On selected source-private rows, fixed-byte or conditional packet methods "
            "show positive utility with paired uncertainty, but the effect is not broad."
        ),
        "evidence": "conditional PQ status and HellaSwag fixed hybrid validation",
        "reviewer_risk": "medium; emphasize narrow scope and controls",
    },
    {
        "claim": "Many apparent wins collapse under source-choice or wrong-row controls.",
        "support_level": "strongly_supported",
        "safe_wording": (
            "OpenBookQA hardening shows that a matched receiver's apparent advantage is "
            "nearly matched by the same-source-choice wrong-row control."
        ),
        "evidence": "OpenBookQA receiver headroom gate",
        "reviewer_risk": "low; this is an honest negative result",
    },
    {
        "claim": "LatentWire is more byte-efficient than dense cache transfer regimes.",
        "support_level": "supported_as_byte_accounting_only",
        "safe_wording": (
            "The communicated packet object is far smaller than dense source-state or "
            "KV/cache floors, but native runtime and task-utility superiority are unmeasured."
        ),
        "evidence": "systems boundary artifact",
        "reviewer_risk": "medium; avoid latency/energy/HBM claims",
    },
    {
        "claim": "LatentWire beats C2C.",
        "support_level": "not_supported",
        "safe_wording": (
            "LatentWire studies a different low-rate source-private point in the design "
            "space and uses C2C as a high-bandwidth baseline/contrast."
        ),
        "evidence": "no direct native C2C run",
        "reviewer_risk": "high; do not claim",
    },
    {
        "claim": "Sparse Resonance Packets are a broad positive ICLR method.",
        "support_level": "not_supported_yet",
        "safe_wording": (
            "Sparse Resonance Packets remain the ICLR direction; current evidence motivates "
            "the next tokenwise/source-causal gate but does not establish broad utility."
        ),
        "evidence": "live branch triage blocker rows",
        "reviewer_risk": "high; keep out of COLM_v2 headline",
    },
]


PROTOCOL_CONTROLS_FIGURE_ROWS = [
    {
        "stage": "source_private_source",
        "object": "source model hidden/candidate evidence",
        "review_question": "Is information computed by a separate source model?",
        "control_or_check": "answer-key-forbidden source path",
    },
    {
        "stage": "packet_encoder",
        "object": "low-byte packet",
        "review_question": "How many bytes cross the model boundary?",
        "control_or_check": "raw/framed/cacheline/DMA byte accounting",
    },
    {
        "stage": "receiver",
        "object": "target-side packet use",
        "review_question": "Does the target improve because of the matched source row?",
        "control_or_check": "target-only, zero-source, target-derived packet",
    },
    {
        "stage": "row_causality",
        "object": "matched versus nonmatched source packet",
        "review_question": "Is the packet carrying row-specific evidence?",
        "control_or_check": "wrong-row, source-row shuffle, same-source-choice wrong-row",
    },
    {
        "stage": "choice_artifact",
        "object": "answer-choice preference",
        "review_question": "Is the receiver just copying the source's favorite option?",
        "control_or_check": "source-index, source-rank, source-score, candidate roll",
    },
    {
        "stage": "packet_semantics",
        "object": "packet atoms/coefficients",
        "review_question": "Do the packet contents matter?",
        "control_or_check": "atom shuffle, coefficient corruption, random same-byte packet",
    },
    {
        "stage": "text_baseline",
        "object": "visible text at same byte budget",
        "review_question": "Does opaque packet transfer beat tiny visible prompts?",
        "control_or_check": "same-byte visible text",
    },
]


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return "; ".join(_fmt(item) for item in value)
    return str(value)


def _safe_md(value: Any) -> str:
    return _fmt(value).replace("|", "\\|").replace("\n", " ")


def _load_inputs(input_paths: dict[str, pathlib.Path]) -> dict[str, dict[str, Any]]:
    return {key: _read_json(path) for key, path in input_paths.items()}


def _input_manifest(input_paths: dict[str, pathlib.Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key, path in input_paths.items():
        rows.append(
            {
                "key": key,
                "path": _repo_path(path),
                "sha256": _sha256_file(path),
            }
        )
    return rows


def _result_row(section: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "section": section,
        "branch": row.get("branch"),
        "status": row.get("status"),
        "score": row.get("score"),
        "baseline": row.get("baseline"),
        "delta": row.get("delta"),
        "ci95_low": row.get("ci95_low"),
        "record_bytes": row.get("record_bytes"),
        "artifact": row.get("artifact"),
        "decision": row.get("decision"),
        "evidence": row.get("evidence"),
    }


def _main_results(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section, key in [
        ("colm_v2_core", "colm_v2_core_rows"),
        ("colm_v2_supporting", "colm_v2_supporting_rows"),
    ]:
        rows.extend(_result_row(section, row) for row in evidence.get(key, []))
    return rows


def _negative_results(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [_result_row("iclr_blocker", row) for row in evidence.get("iclr_blocker_rows", [])]


def _openbookqa_controls(openbookqa: dict[str, Any]) -> list[dict[str, Any]]:
    headline = openbookqa.get("headline", {})
    default_seed = headline.get("default_seed")
    per_seed = openbookqa.get("per_seed", [])
    seed_row = next(
        (row for row in per_seed if row.get("seed") == default_seed),
        per_seed[0] if per_seed else {},
    )
    metrics = seed_row.get("condition_metrics", {})
    matched = metrics.get("matched_source_private_packet", {})
    matched_acc = matched.get("receiver_accuracy")
    rows: list[dict[str, Any]] = []
    for condition, row in sorted(metrics.items()):
        accuracy = row.get("receiver_accuracy")
        rows.append(
            {
                "benchmark": "OpenBookQA",
                "seed": seed_row.get("seed"),
                "condition": condition,
                "receiver_accuracy": accuracy,
                "matched_minus_condition": (
                    matched_acc - accuracy
                    if isinstance(matched_acc, (int, float))
                    and isinstance(accuracy, (int, float))
                    else None
                ),
                "base_accuracy": row.get("base_accuracy"),
                "target_public_accuracy": row.get("target_public_accuracy"),
                "receiver_minus_base": row.get("receiver_minus_base"),
                "override_rate": row.get("override_rate"),
                "help_count": row.get("help_count"),
                "harm_count": row.get("harm_count"),
                "ci95_low_vs_base": row.get("paired_ci95_vs_base", {}).get("ci95_low"),
                "ci95_high_vs_base": row.get("paired_ci95_vs_base", {}).get("ci95_high"),
            }
        )
    return rows


def _systems_accounting(systems: dict[str, Any]) -> list[dict[str, Any]]:
    selected_keys = [
        "row_group",
        "method",
        "communicated_object",
        "raw_bytes",
        "framed_bytes",
        "cacheline_bytes",
        "batch64_bytes",
        "source_private",
        "source_text_exposed",
        "source_hidden_or_score_vector_exposed",
        "source_kv_exposed",
        "source_packet_cached",
        "source_scoring_included",
        "native_measured",
        "native_claim_allowed",
        "measurement_status",
        "claim_allowed",
        "overclaim_guard",
        "source_url",
    ]
    rows: list[dict[str, Any]] = []
    for row in systems.get("rows", []):
        rows.append({key: row.get(key) for key in selected_keys})
    return rows


def _evidence_ladder(
    main_results: list[dict[str, Any]], negative_results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    ladder: list[dict[str, Any]] = []
    for tier, rows in [
        ("1_core_colm_v2_positive", [r for r in main_results if r["section"] == "colm_v2_core"]),
        (
            "2_guardrail_or_support",
            [r for r in main_results if r["section"] == "colm_v2_supporting"],
        ),
        ("3_iclr_blocker_or_negative", negative_results),
    ]:
        for row in rows:
            ladder.append(
                {
                    "tier": tier,
                    "branch": row.get("branch"),
                    "status": row.get("status"),
                    "score": row.get("score"),
                    "baseline": row.get("baseline"),
                    "delta": row.get("delta"),
                    "ci95_low": row.get("ci95_low"),
                    "record_bytes": row.get("record_bytes"),
                    "artifact": row.get("artifact"),
                }
            )
    return ladder


def _contribution_table(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in evidence.get("current_contributions", []):
        rows.append(
            {
                "contribution": item.get("name"),
                "status": item.get("status"),
                "needs_work": item.get("gap") or item.get("needs_work"),
                "colm_v2_role": _contribution_role(item.get("name", "")),
            }
        )
    return rows


def _contribution_role(name: str) -> str:
    if "control" in name or "controls" in name:
        return "core evaluation contribution"
    if "byte" in name or "systems" in name:
        return "systems framing contribution"
    if "conditional" in name or "packet" in name:
        return "narrow positive-method evidence"
    if "negative" in name or "failure" in name:
        return "claim-boundary evidence"
    return "supporting paper contribution"


def build_review_packet(
    input_paths: dict[str, pathlib.Path] | None = None,
) -> dict[str, Any]:
    paths = input_paths or DEFAULT_INPUT_PATHS
    inputs = _load_inputs(paths)
    evidence = inputs["evidence_table"]
    triage = inputs["live_branch_triage"]
    systems = inputs["systems_boundary"]
    openbookqa = inputs["openbookqa_receiver_headroom"]

    main_results = _main_results(evidence)
    negative_results = _negative_results(evidence)
    strict_controls = _openbookqa_controls(openbookqa)
    systems_accounting = _systems_accounting(systems)
    evidence_ladder = _evidence_ladder(main_results, negative_results)

    review_packet = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "readiness": evidence["readiness"],
        "story": evidence["story"],
        "submission_gap": {
            "colm_v2": (
                "review-facing packaging: final tables, figures, baseline matrix, "
                "claim audit, and artifact manifest"
            ),
            "iclr": triage["submission_gap"],
        },
        "paper_decision": evidence["paper_decision"],
        "next_exact_gate": triage["next_exact_gate"],
        "claim_boundaries": evidence["claim_boundaries"],
        "contribution_table": _contribution_table(evidence),
        "main_results": main_results,
        "strict_controls": strict_controls,
        "systems_headline": systems.get("headline", {}),
        "systems_accounting": systems_accounting,
        "baseline_matrix": BASELINE_MATRIX,
        "negative_results": negative_results,
        "claim_audit": CLAIM_AUDIT,
        "figure_data": {
            "evidence_ladder": evidence_ladder,
            "protocol_controls": PROTOCOL_CONTROLS_FIGURE_ROWS,
            "systems_boundary_rows": systems_accounting,
        },
        "reproducibility": {
            "input_manifest": _input_manifest(paths),
            "generated_tables": [
                "main_results.csv",
                "strict_controls.csv",
                "systems_accounting.csv",
                "baseline_matrix.csv",
                "claim_audit.csv",
                "negative_results.csv",
                "figure_data_evidence_ladder.csv",
                "figure_data_protocol_controls.csv",
                "figure_data_systems_boundary.csv",
            ],
            "local_venv": "./venv_arm64",
            "scratch_policy": ".debug/ only; not checked in",
            "no_ssh": True,
        },
        "reviewer_packet_status": {
            "colm_v2_review_ready_after_human_paper_pass": True,
            "iclr_positive_method_ready": False,
            "do_not_claim_better_than_c2c": True,
            "direct_native_systems_claims_supported": False,
        },
    }
    return review_packet


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key)) for key in fieldnames})


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_safe_md(row.get(column)) for column in columns) + " |")
    return lines


def render_markdown(packet: dict[str, Any]) -> str:
    lines: list[str] = [
        "# LatentWire COLM_v2 Review Packet",
        "",
        f"- created UTC: `{packet['created_utc']}`",
        f"- COLM_v2 readiness: `{packet['readiness']['colm_v2']}`",
        f"- ICLR readiness: `{packet['readiness']['iclr']}`",
        "",
        "## Current Story",
        "",
        packet["story"],
        "",
        "## Exact Submission Gaps",
        "",
        f"- COLM_v2: {packet['submission_gap']['colm_v2']}",
        f"- ICLR: {packet['submission_gap']['iclr']}",
        "",
        "## Current Technical Contributions",
        "",
        *_markdown_table(
            packet["contribution_table"],
            ["contribution", "status", "needs_work", "colm_v2_role"],
        ),
        "",
        "## Main Results",
        "",
        "Rows that can appear in the scoped workshop story, with guarded wording.",
        "",
        *_markdown_table(
            packet["main_results"],
            [
                "section",
                "branch",
                "status",
                "score",
                "baseline",
                "delta",
                "ci95_low",
                "record_bytes",
                "decision",
            ],
        ),
        "",
        "## Strict Controls",
        "",
        "OpenBookQA hardening is included because it is the cleanest warning that source-choice artifacts can mimic packet gains.",
        "",
        *_markdown_table(
            packet["strict_controls"],
            [
                "benchmark",
                "seed",
                "condition",
                "receiver_accuracy",
                "matched_minus_condition",
                "base_accuracy",
                "target_public_accuracy",
                "override_rate",
                "help_count",
                "harm_count",
            ],
        ),
        "",
        "## Systems Boundary",
        "",
        packet["systems_headline"].get("claim_scope", ""),
        "",
        *_markdown_table(
            packet["systems_accounting"],
            [
                "row_group",
                "method",
                "communicated_object",
                "raw_bytes",
                "framed_bytes",
                "cacheline_bytes",
                "batch64_bytes",
                "source_private",
                "source_kv_exposed",
                "native_measured",
                "claim_allowed",
            ],
        ),
        "",
        "## Baseline Matrix",
        "",
        *_markdown_table(
            packet["baseline_matrix"],
            [
                "category",
                "baseline",
                "what_transfers",
                "source_private",
                "byte_regime",
                "included_in_current_eval",
                "latentwire_distinction",
                "still_needed",
                "source",
            ],
        ),
        "",
        "## Negative Results And Saturation",
        "",
        *_markdown_table(
            packet["negative_results"],
            [
                "branch",
                "status",
                "score",
                "baseline",
                "delta",
                "ci95_low",
                "record_bytes",
                "decision",
            ],
        ),
        "",
        "## Reviewer Claim Audit",
        "",
        *_markdown_table(
            packet["claim_audit"],
            ["claim", "support_level", "safe_wording", "evidence", "reviewer_risk"],
        ),
        "",
        "## Figure Data",
        "",
        "- `figure_data_evidence_ladder.csv`: evidence ladder for core, guardrail, and blocker rows.",
        "- `figure_data_protocol_controls.csv`: protocol/control schematic data.",
        "- `figure_data_systems_boundary.csv`: byte/exposure rows for systems boundary plots.",
        "",
        "## Reproducibility Manifest",
        "",
        *_markdown_table(packet["reproducibility"]["input_manifest"], ["key", "path", "sha256"]),
        "",
        "## Next Exact Gate",
        "",
        f"- name: `{packet['next_exact_gate']['name']}`",
        f"- primary path: {packet['next_exact_gate']['primary_path']}",
        f"- fallback path: {packet['next_exact_gate']['fallback_path']}",
        f"- pass bar: {packet['next_exact_gate']['pass_bar']}",
        "",
        "## Claim Boundaries",
        "",
    ]
    for boundary in packet["claim_boundaries"]:
        lines.append(f"- {boundary}")
    lines.append("")
    return "\n".join(lines)


def write_outputs(packet: dict[str, Any], output_dir: pathlib.Path, paper_path: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "review_packet.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = render_markdown(packet)
    (output_dir / "review_packet.md").write_text(markdown, encoding="utf-8")
    paper_path.write_text(markdown, encoding="utf-8")

    _write_csv(output_dir / "contribution_table.csv", packet["contribution_table"])
    _write_csv(output_dir / "main_results.csv", packet["main_results"])
    _write_csv(output_dir / "strict_controls.csv", packet["strict_controls"])
    _write_csv(output_dir / "systems_accounting.csv", packet["systems_accounting"])
    _write_csv(output_dir / "baseline_matrix.csv", packet["baseline_matrix"])
    _write_csv(output_dir / "claim_audit.csv", packet["claim_audit"])
    _write_csv(output_dir / "negative_results.csv", packet["negative_results"])
    _write_csv(
        output_dir / "figure_data_evidence_ladder.csv",
        packet["figure_data"]["evidence_ladder"],
    )
    _write_csv(
        output_dir / "figure_data_protocol_controls.csv",
        packet["figure_data"]["protocol_controls"],
    )
    _write_csv(
        output_dir / "figure_data_systems_boundary.csv",
        packet["figure_data"]["systems_boundary_rows"],
    )

    manifest = {
        "created_utc": packet["created_utc"],
        "packet_json": "review_packet.json",
        "packet_markdown": "review_packet.md",
        "paper_markdown": _repo_path(paper_path),
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
        "input_manifest": packet["reproducibility"]["input_manifest"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_lines = [
        "# LatentWire COLM_v2 Review Packet Manifest",
        "",
        f"- created UTC: `{manifest['created_utc']}`",
        f"- paper markdown: `{manifest['paper_markdown']}`",
        "",
        "## Outputs",
        "",
    ]
    for item in manifest["outputs"]:
        manifest_lines.append(f"- `{item}`")
    manifest_lines.extend(["", "## Inputs", ""])
    manifest_lines.extend(_markdown_table(manifest["input_manifest"], ["key", "path", "sha256"]))
    manifest_lines.append("")
    (output_dir / "manifest.md").write_text("\n".join(manifest_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-path", type=pathlib.Path, default=DEFAULT_PAPER_PATH)
    args = parser.parse_args()

    packet = build_review_packet()
    write_outputs(packet, args.output_dir, args.paper_path)


if __name__ == "__main__":
    main()
