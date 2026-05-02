from __future__ import annotations

import numpy as np
import pytest

from scripts import build_source_private_hellaswag_minimal_packet_compaction as gate


def test_minimal_payload_bytes_for_four_candidates() -> None:
    assert gate._minimal_payload_bits(4) == 2
    assert gate._minimal_payload_bytes(4) == 1
    assert gate._minimal_payload_bytes(257) == 2


def test_candidate_packet_roundtrip_and_rejects_invalid_id() -> None:
    predictions = np.asarray([0, 1, 2, 3], dtype=np.int64)
    encoded, decoded = gate._compact_predictions(predictions, candidate_count=4)

    assert encoded.dtype == np.uint8
    assert decoded.tolist() == predictions.tolist()
    with pytest.raises(ValueError):
        gate._encode_candidate(4, candidate_count=4)
    with pytest.raises(ValueError):
        gate._decode_candidate(4, candidate_count=4)


def test_packet_accounting_reduces_raw_and_framed_bytes() -> None:
    row = gate._packet_accounting(row_count=10, original_raw_bytes=2, candidate_count=4)

    assert row["compact_raw_payload_bytes_per_request"] == 1
    assert row["compact_framed_record_bytes_per_request"] == 4
    assert row["logical_original_raw_payload_bytes_total"] == 20
    assert row["logical_compact_raw_payload_bytes_total"] == 10
    assert row["logical_original_framed_record_bytes_total"] == 50
    assert row["logical_compact_framed_record_bytes_total"] == 40
