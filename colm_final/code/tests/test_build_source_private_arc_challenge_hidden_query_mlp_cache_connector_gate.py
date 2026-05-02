import numpy as np

from scripts import build_source_private_arc_challenge_hidden_query_mlp_cache_connector_gate as gate
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate


def _rows() -> list[arc_gate.ArcRow]:
    return [
        arc_gate.ArcRow(
            row_id="r0",
            content_id="c0",
            question="q0",
            choices=("a", "b", "c"),
            choice_labels=("A", "B", "C"),
            answer_index=1,
            answer_label="B",
        ),
        arc_gate.ArcRow(
            row_id="r1",
            content_id="c1",
            question="q1",
            choices=("d", "e"),
            choice_labels=("A", "B"),
            answer_index=0,
            answer_label="A",
        ),
    ]


def test_mlp_map_learns_synthetic_target_and_is_applyable() -> None:
    rng = np.random.default_rng(11)
    source = rng.normal(size=(32, 6))
    linear = rng.normal(size=(6, 4))
    target = np.tanh(source @ linear)
    mapper = gate._fit_mlp_map(
        source_features=source,
        target_features=target,
        fit_indices=np.arange(source.shape[0]),
        pca_dim=6,
        hidden_dim=12,
        weight_decay=0.0,
        seed=13,
        epochs=120,
        lr=0.02,
    )

    predicted = gate._apply_mlp_map(source, mapper)

    assert predicted.shape == target.shape
    assert mapper["pca_dim"] == 6
    assert mapper["hidden_dim"] == 12
    assert mapper["fit_loss_final"] < mapper["fit_loss_initial"]
    assert np.mean(gate.hq_gate._rowwise_cosine(predicted, target)) > 0.85


def test_rotate_blocks_moves_whole_examples_not_candidate_columns() -> None:
    rows = _rows()
    flat = np.arange(5 * 2, dtype=float).reshape(5, 2)

    rotated = gate._rotate_blocks(rows, flat)

    assert rotated.shape == flat.shape
    assert np.allclose(rotated[:2], flat[3:])
    assert np.allclose(rotated[2:], flat[:3])


def test_jsonable_mapper_excludes_weight_arrays() -> None:
    mapper = {
        "pca": {"singular_values_top8": [3.0, 2.0]},
        "target_mean": np.zeros(2),
        "target_scale": np.ones(2),
        "state": {"net.0.weight": np.zeros((2, 2))},
        "pca_dim": 2,
        "hidden_dim": 3,
        "weight_decay": 0.01,
        "fit_loss_initial": 1.0,
        "fit_loss_final": 0.5,
    }

    payload = gate._jsonable_mapper(mapper)

    assert "state" not in payload
    assert "target_mean" not in payload
    assert payload["pca_singular_values_top8"] == [3.0, 2.0]
    assert payload["weight_decay"] == 0.01
