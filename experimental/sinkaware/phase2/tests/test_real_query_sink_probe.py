import torch

from experimental.sinkaware.phase2.real_query_sink_probe import _evaluate


def test_real_query_probe_promotes_hidden_plus_position_signal() -> None:
    samples = []
    for idx in range(120):
        hidden = torch.tensor([float(idx % 7), float((idx // 7) % 5)])
        position = torch.tensor([idx / 119.0])
        target = 0.2 * hidden[0] - 0.1 * hidden[1] + 0.05 * position[0]
        samples.append({"layer": 0, "hidden": hidden, "position": position, "target": target})

    result = _evaluate(samples, ranks=(1, 2, 4, 8))

    assert result["summary"]["rank2_hidden_plus_pos_r2"] > 0.95
