from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import json
import torch

import latent_bridge.ablation_sweep as sweep
import latent_bridge.calibrate as calibrate


class _FakeTokenizer:
    def __init__(self, lengths_by_prompt: dict[str, int]) -> None:
        self.pad_token_id = None
        self.eos_token = "<eos>"
        self.lengths_by_prompt = lengths_by_prompt

    def __call__(self, batch, return_tensors="pt", padding=None, truncation=None, max_length=None):
        if isinstance(batch, str):
            batch = [batch]
        seq_len = max_length or max(self.lengths_by_prompt.values())
        data = []
        masks = []
        for prompt in batch:
            real_len = min(self.lengths_by_prompt[prompt], seq_len)
            data.append(list(range(1, real_len + 1)) + [0] * (seq_len - real_len))
            masks.append([1] * real_len + [0] * (seq_len - real_len))
        tensor = torch.tensor(data, dtype=torch.long)
        mask = torch.tensor(masks, dtype=torch.long)

        class _Encoded(dict):
            def __init__(self, input_ids, attention_mask):
                super().__init__(input_ids=input_ids, attention_mask=attention_mask)
                self.input_ids = input_ids
                self.attention_mask = attention_mask

            def to(self, device):
                self.input_ids = self.input_ids.to(device)
                self["input_ids"] = self.input_ids
                self.attention_mask = self.attention_mask.to(device)
                self["attention_mask"] = self.attention_mask
                return self

        return _Encoded(tensor, mask)


class _FakeCache:
    def __init__(self, layers):
        self.layers = layers

    def to_legacy_cache(self):
        return self.layers


class _FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, **enc):
        self.calls += 1
        input_ids = enc["input_ids"]
        batch, seq = input_ids.shape
        layers = []
        for layer_idx in range(2):
            values = input_ids.to(torch.float32).view(batch, 1, seq, 1)
            K = values + float(layer_idx * 100)
            V = values + float(layer_idx * 1000)
            layers.append((K, V, None))
        return SimpleNamespace(past_key_values=_FakeCache(layers))


class _FakeAttentionModel:
    def __call__(self, **enc):
        input_ids = enc["input_ids"]
        batch, seq = input_ids.shape
        attn = torch.zeros(batch, 2, seq, seq, dtype=torch.float32)
        for batch_idx in range(batch):
            valid_len = int(enc["attention_mask"][batch_idx].sum().item())
            if valid_len <= 1:
                continue
            attn[batch_idx, 0, valid_len - 1, 0] = 1.0
            attn[batch_idx, 1, valid_len - 1, valid_len - 2] = 1.0
        return SimpleNamespace(attentions=(attn,))


def test_collect_kvs_masks_padding_tokens_and_concatenates_valid_positions() -> None:
    model = _FakeModel()
    tokenizer = _FakeTokenizer({"a": 4, "b": 2, "c": 1})
    prompts = ["a", "b", "c"]

    kvs = calibrate.collect_kvs(
        model,
        tokenizer,
        prompts,
        max_length=4,
        batch_size=2,
        device="cpu",
    )

    assert len(kvs) == 2
    for K, V in kvs:
        assert K.shape == (7, 1, 1, 1)
        assert V.shape == (7, 1, 1, 1)
        assert K.dtype == torch.float32
        assert V.dtype == torch.float32

    # The padded positions for prompts "b" and "c" are skipped entirely.
    assert kvs[0][0].flatten().tolist() == [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 1.0]
    assert kvs[0][1].flatten().tolist() == [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 1.0]


def test_collect_aligned_kv_pairs_truncates_to_shorter_valid_length() -> None:
    src_model = _FakeModel()
    tgt_model = _FakeModel()
    src_tokenizer = _FakeTokenizer({"a": 4, "b": 3})
    tgt_tokenizer = _FakeTokenizer({"a": 2, "b": 5})
    prompts = ["a", "b"]

    src_kvs, tgt_kvs = calibrate.collect_aligned_kv_pairs(
        src_model,
        src_tokenizer,
        tgt_model,
        tgt_tokenizer,
        prompts,
        max_length=5,
        batch_size=2,
        device="cpu",
    )

    assert len(src_kvs) == 2
    assert len(tgt_kvs) == 2
    for pair in src_kvs + tgt_kvs:
        K, V = pair
        assert K.shape == (5, 1, 1, 1)
        assert V.shape == (5, 1, 1, 1)

    assert src_kvs[0][0].flatten().tolist() == [1.0, 2.0, 1.0, 2.0, 3.0]
    assert tgt_kvs[0][0].flatten().tolist() == [1.0, 2.0, 1.0, 2.0, 3.0]


def test_collect_aligned_kv_pairs_uses_source_reasoning_prompt() -> None:
    src_model = _FakeModel()
    tgt_model = _FakeModel()
    prompt = "a"
    source_prompt = calibrate._source_reasoning_prompt(prompt, "cot")
    src_tokenizer = _FakeTokenizer({source_prompt: 4})
    tgt_tokenizer = _FakeTokenizer({prompt: 2})

    src_kvs, tgt_kvs = calibrate.collect_aligned_kv_pairs(
        src_model,
        src_tokenizer,
        tgt_model,
        tgt_tokenizer,
        [prompt],
        max_length=5,
        batch_size=1,
        device="cpu",
        source_reasoning_mode="cot",
    )

    assert src_kvs[0][0].shape == (2, 1, 1, 1)
    assert tgt_kvs[0][0].shape == (2, 1, 1, 1)


def test_collect_group_attention_templates_supports_peak_mode() -> None:
    model = _FakeAttentionModel()
    tokenizer = _FakeTokenizer({"a": 4, "b": 4})

    templates = calibrate.collect_group_attention_templates(
        model,
        tokenizer,
        ["a", "b"],
        max_length=4,
        batch_size=2,
        device="cpu",
        kv_heads=2,
        group_count=2,
        bins=4,
        template_mode="peak",
    )

    assert len(templates) == 1
    layer0 = templates[0]
    assert layer0.shape == (2, 4)
    assert torch.allclose(layer0[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(layer0[1], torch.tensor([0.0, 0.0, 1.0, 0.0]))


def test_calibrate_config_helpers_parse_and_batch() -> None:
    assert calibrate.torch_dtype("float32") is torch.float32
    assert calibrate.torch_dtype("float16") is torch.float16
    assert list(calibrate.batched([1, 2, 3, 4], 3)) == [[1, 2, 3], [4]]


def test_calibrate_parse_args_supports_unit_tested_ablation_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "reduced_rank",
            "--alignment-rank",
            "2",
            "--rotation",
            "dct",
            "--layer-pairing",
            "random",
            "--whitening",
            "--target-whitening",
            "--source-reasoning-mode",
            "cot",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "reduced_rank"
    assert args.alignment_rank == 2
    assert args.rotation == "dct"
    assert args.layer_pairing == "random"
    assert args.whitening is True
    assert args.target_whitening is True
    assert args.source_reasoning_mode == "cot"


def test_calibrate_parse_args_accepts_grouped_alignment(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "grouped_auto",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "grouped_auto"


def test_calibrate_parse_args_accepts_grouped_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "grouped_transport",
            "--transport-residual-rank",
            "8",
            "--transport-temperature",
            "0.5",
            "--transport-sinkhorn-iters",
            "12",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "grouped_transport"
    assert args.transport_residual_rank == 8
    assert args.transport_temperature == 0.5
    assert args.transport_sinkhorn_iters == 12


def test_calibrate_parse_args_accepts_grouped_permutation(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "grouped_permutation",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "grouped_permutation"


def test_calibrate_parse_args_accepts_grouped_signature_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "grouped_signature_transport",
            "--transport-signature-rank",
            "6",
            "--transport-signature-weight",
            "0.25",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "grouped_signature_transport"
    assert args.transport_signature_rank == 6
    assert args.transport_signature_weight == 0.25


def test_calibrate_parse_args_accepts_grouped_subspace_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "grouped_subspace_transport",
            "--transport-signature-rank",
            "4",
            "--transport-signature-weight",
            "0.1",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "grouped_subspace_transport"
    assert args.transport_signature_rank == 4
    assert args.transport_signature_weight == 0.1


def test_calibrate_parse_args_accepts_grouped_canonical_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "grouped_canonical_transport",
            "--canonical-subspace-rank",
            "6",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "grouped_canonical_transport"
    assert args.canonical_subspace_rank == 6


def test_calibrate_parse_args_accepts_grouped_covariance_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "grouped_covariance_transport",
            "--transport-signature-weight",
            "0.1",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "grouped_covariance_transport"


def test_calibrate_parse_args_accepts_grouped_template_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "grouped_template_transport",
            "--transport-template-bins",
            "32",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "grouped_template_transport"
    assert args.transport_template_bins == 32
    assert args.transport_signature_weight == 0.0


def test_calibrate_parse_args_accepts_grouped_template_subspace_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "grouped_template_subspace_transport",
            "--transport-template-bins",
            "48",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "grouped_template_subspace_transport"
    assert args.transport_template_bins == 48


def test_calibrate_parse_args_accepts_broadcast_template_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "broadcast_template_transport",
            "--transport-template-bins",
            "40",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "broadcast_template_transport"
    assert args.transport_template_bins == 40


def test_calibrate_parse_args_accepts_broadcast_template_ot_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "broadcast_template_ot_transport",
            "--transport-template-bins",
            "32",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "broadcast_template_ot_transport"
    assert args.transport_template_bins == 32


def test_calibrate_parse_args_accepts_broadcast_peak_template_ot_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--alignment",
            "broadcast_peak_template_ot_transport",
            "--transport-template-bins",
            "24",
        ],
    )

    args = calibrate.parse_args()
    assert args.alignment == "broadcast_peak_template_ot_transport"
    assert args.transport_template_bins == 24


def test_calibrate_parse_args_supports_head_and_prequant_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--head-selection-topk",
            "1",
            "--head-selection-ratio",
            "0.5",
            "--head-selection-metric",
            "negative_error",
            "--pre-quant-rank",
            "64",
            "--pre-quant-shrinkage",
            "0.25",
            "--quantization-correction",
            "affine",
        ],
    )

    args = calibrate.parse_args()
    assert args.head_selection_topk == 1
    assert args.head_selection_ratio == 0.5
    assert args.head_selection_metric == "negative_error"
    assert args.pre_quant_rank == 64
    assert args.pre_quant_shrinkage == 0.25
    assert args.quantization_correction == "affine"


def test_calibrate_parse_args_accepts_ridge_quantization_correction(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--quantization-correction",
            "ridge",
        ],
    )

    args = calibrate.parse_args()

    assert args.quantization_correction == "ridge"


def test_calibrate_parse_args_accepts_learned_fusion_dropout(monkeypatch) -> None:
    monkeypatch.setattr(
        calibrate.sys,
        "argv",
        [
            "calibrate.py",
            "--source-model",
            "src",
            "--target-model",
            "tgt",
            "--calibration-file",
            "cal.txt",
            "--output",
            "out.pt",
            "--learned-fusion-dropout",
            "0.5",
        ],
    )

    args = calibrate.parse_args()
    assert args.learned_fusion_dropout == 0.5


def test_ablation_sweep_parse_accuracies_and_main_plumbing(monkeypatch, tmp_path) -> None:
    summary = """
    noise before
    === Summary ===
      target_alone: 0.250
      text_to_text: 0.500
      rotalign_kv: 0.750
    trailing noise
    """
    parsed = sweep.parse_accuracies(summary)
    assert parsed == {
        "target_alone": 0.25,
        "text_to_text": 0.5,
        "rotalign_kv": 0.75,
    }

    out_path = tmp_path / "results.jsonl"
    ckpt_dir = tmp_path / "ckpts"
    args = Namespace(
        source_model="src-model",
        target_model="tgt-model",
        calibration_file="cal.txt",
        eval_file="eval.jsonl",
        output=str(out_path),
        checkpoint_dir=str(ckpt_dir),
        rotations=["orthogonal"],
        alignments=["procrustes"],
        bits=[4],
        whiten=["on"],
        device="cpu",
        dtype="float32",
        head_selection_topks=[1],
        head_selection_ratios=[0.5],
        head_selection_metrics=["negative_error"],
        pre_quant_ranks=[64],
        pre_quant_shrinkages=[0.25],
        quantization_corrections=["affine"],
        source_reasoning_modes=["plain", "cot"],
        kv_transports=["k_only"],
    )
    monkeypatch.setattr(sweep, "parse_args", lambda: args)

    commands = []

    def fake_run_cmd(cmd):
        commands.append(cmd)
        if any(part.endswith("evaluate.py") for part in cmd):
            return """
            === Summary ===
              target_alone: 0.100
              text_to_text: 0.200
              rotalign_kv: 0.300
            """
        return "calibrated"

    monkeypatch.setattr(sweep, "run_cmd", fake_run_cmd)
    times = iter([10.0, 16.0, 20.0, 26.0, 30.0, 36.0, 40.0, 46.0, 50.0, 56.0])
    monkeypatch.setattr(sweep.time, "time", lambda: next(times))

    sweep.main()

    assert len(commands) == 4
    assert any(part.endswith("scripts/calibrate.py") for part in commands[0])
    assert any(part == "--whitening" for part in commands[0])
    assert any(part == "--head-selection-topk" for part in commands[0])
    assert any(part == "--head-selection-ratio" for part in commands[0])
    assert any(part == "--head-selection-metric" for part in commands[0])
    assert any(part == "--pre-quant-rank" for part in commands[0])
    assert any(part == "--quantization-correction" for part in commands[0])
    assert any(part == "--source-reasoning-mode" for part in commands[0])
    assert any(part.endswith("scripts/evaluate.py") for part in commands[1])
    assert any(part == "--kv-transport" for part in commands[1])
    assert any(part == "--source-reasoning-mode" for part in commands[1])

    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["rotation"] == "orthogonal"
    assert records[0]["alignment"] == "procrustes"
    assert records[0]["bits"] == 4
    assert records[0]["whitening"] is True
    assert records[0]["head_selection_topk"] == 1
    assert records[0]["head_selection_ratio"] == 0.5
    assert records[0]["head_selection_metric"] == "negative_error"
    assert records[0]["pre_quant_rank"] == 64
    assert records[0]["quantization_correction"] == "affine"
    assert records[0]["kv_transport"] == "k_only"
    assert records[0]["rotalign_kv"] == 0.3
    assert {record["source_reasoning_mode"] for record in records} == {"plain", "cot"}
