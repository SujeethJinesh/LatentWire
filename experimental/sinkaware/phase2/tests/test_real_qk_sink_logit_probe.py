import torch

from experimental.sinkaware.phase2.real_qk_sink_logit_probe import _evaluate


def test_qk_probe_detects_query_side_signal() -> None:
    samples = []
    for idx in range(120):
        hidden = torch.tensor([float(idx % 9), float((idx // 9) % 4), float(idx % 3)])
        position = torch.tensor([idx / 119.0])
        target = -0.3 * hidden[0] + 0.2 * hidden[1] + 0.01 * position[0]
        samples.append({"layer": 0, "hidden": hidden, "position": position, "target": target})

    result = _evaluate(samples, ranks=(1, 2, 4, 8))

    assert result["summary"]["rank2_hidden_plus_pos_r2"] > 0.90
