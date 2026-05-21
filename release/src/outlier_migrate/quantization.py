"""CPU helpers for simple symmetric INT4 dequantization."""

from __future__ import annotations

from collections.abc import Sequence


Matrix = list[list[float]]


def symmetric_int4_quantize_row(row: Sequence[float]) -> list[float]:
    """Quantize one row to signed INT4 levels and dequantize to floats."""

    if not row:
        return []
    scale = max(abs(float(value)) for value in row) / 7.0
    if scale == 0.0:
        return [0.0 for _ in row]
    return [_dequantize_value(float(value), scale) for value in row]


def quantize_matrix_per_output_channel(
    matrix: Sequence[Sequence[float]],
    *,
    protected_rows: set[int] | None = None,
    protected_cols: set[int] | None = None,
) -> Matrix:
    """Apply per-output-channel INT4 with protected rows and columns restored."""

    protected_rows = protected_rows or set()
    protected_cols = protected_cols or set()
    original = [[float(value) for value in row] for row in matrix]
    quantized = [symmetric_int4_quantize_row(row) for row in original]
    for row_index in protected_rows:
        if 0 <= row_index < len(original):
            quantized[row_index] = list(original[row_index])
    for row_index, row in enumerate(original):
        for col_index in protected_cols:
            if 0 <= col_index < len(row):
                quantized[row_index][col_index] = row[col_index]
    return quantized


def _dequantize_value(value: float, scale: float) -> float:
    level = round(value / scale)
    clipped = min(7, max(-7, level))
    return float(clipped * scale)
